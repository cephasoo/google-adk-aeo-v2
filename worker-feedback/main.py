# --- /worker-feedback/main.py ---
import functions_framework
from flask import jsonify
import vertexai
from vertexai.generative_models import GenerativeModel
import os
import json
import re
import uuid
import datetime
import certifi
import requests
from google.cloud import firestore, tasks_v2

# --- UNIFIED MODEL ADAPTER (The Brain Switch) ---
class UnifiedModel:
    def __init__(self, provider, model_name):
        self.provider = provider
        self.model_name = model_name
        if provider == "vertex_ai":
            vertexai.init(project=PROJECT_ID, location=LOCATION)
            from vertexai.generative_models import GenerativeModel, SafetySetting
            # Standard safety settings
            safety = [
                SafetySetting(category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH),
                SafetySetting(category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH),
                SafetySetting(category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH),
                SafetySetting(category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH),
            ]
            self._native_model = GenerativeModel(model_name, safety_settings=safety)
            print(f"✅ Loaded Unified Vertex Model: {model_name}")

    def generate_content(self, prompt, generation_config=None, max_retries=3):
        import time
        import random
        if self.provider == "vertex_ai":
            retries = 0
            while retries <= max_retries:
                try:
                    return self._native_model.generate_content(prompt, generation_config=generation_config)
                except Exception as e:
                    error_msg = str(e).lower()
                    if "429" in error_msg or "resource exhausted" in error_msg:
                        if retries < max_retries:
                            wait_time = (2 ** retries) + random.uniform(0, 1)
                            print(f"⚠️ Feedback 429 (Quota): Retrying in {wait_time:.2f}s...")
                            time.sleep(wait_time)
                            retries += 1
                            continue
                    raise e
        return None

# --- Config ---
PROJECT_ID = os.environ.get("PROJECT_ID")
LOCATION = os.environ.get("LOCATION")
MODEL_NAME = os.environ.get("MODEL_NAME")
N8N_PROPOSAL_WEBHOOK_URL = os.environ.get("N8N_PROPOSAL_WEBHOOK_URL") 
STORY_WORKER_URL = os.environ.get("STORY_WORKER_URL") 
QUEUE_NAME = "story-worker-queue"
FUNCTION_IDENTITY_EMAIL = os.environ.get("FUNCTION_IDENTITY_EMAIL")
# PROPOSAL_KEYWORDS = ["outline", "draft", "proposal", "story", "brief"]
INGEST_KNOWLEDGE_URL = os.environ.get("INGEST_KNOWLEDGE_URL")

unimodel = None
db = None
tasks_client = None

# --- Utils ---
def extract_json(text):
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match: raise ValueError(f"No JSON found")
    return json.loads(match.group(0))

def extract_url_from_text(text):
    url_pattern = r'(https?://[^\s<>|]+)'
    match = re.search(url_pattern, text)
    if match: return match.group(1)
    return None

def dispatch_task(payload, target_url):
    global tasks_client
    if tasks_client is None: tasks_client = tasks_v2.CloudTasksClient()
    parent = tasks_client.queue_path(PROJECT_ID, LOCATION, QUEUE_NAME)
    
    from google.protobuf import duration_pb2
    deadline = duration_pb2.Duration()
    deadline.FromSeconds(600)

    task = {
        "http_request": {
            "http_method": tasks_v2.HttpMethod.POST,
            "url": target_url,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(payload).encode(),
            "oidc_token": {"service_account_email": FUNCTION_IDENTITY_EMAIL}
        },
        "dispatch_deadline": deadline
    }
    tasks_client.create_task(request={"parent": parent, "task": task})

def tell_then_and_now_story(interlinked_concepts, tool_confirmation=None):
    if not tool_confirmation or not tool_confirmation.get("confirmed"): raise PermissionError("Wait for approval.")
    return unimodel.generate_content(f"Tell a 'Then and Now' story using: {interlinked_concepts}").text

def refine_proposal(topic, current_proposal, critique):
    return extract_json(unimodel.generate_content(f"REWRITE proposal... Draft: {json.dumps(current_proposal)}...").text)

# --- THE STATEFUL AND HARDENED FEEDBACK WORKER ---
@functions_framework.http
def process_feedback_logic(request):
    global unimodel, db
    if unimodel is None: 
        unimodel = UnifiedModel("vertex_ai", MODEL_NAME)
        db = firestore.Client()

    req = request.get_json(silent=True)
    if isinstance(req, list): req = req[0] # Add safety for n8n list payloads
    print(f"DEBUG: Feedback Worker received payload keys: {list(req.keys()) if req else 'None'}")
    session_id = req.get('session_id')
    user_feedback = req.get('feedback', '')
    images = req.get('images', []) # New sensory input array
    print(f"DEBUG: Feedback Worker images count: {len(images)}")
    
    doc_ref = db.collection('agent_sessions').document(session_id)
    session_doc = doc_ref.get()
    if not session_doc.exists: return jsonify({"error": "Session not found"}), 404
    session_data = session_doc.to_dict()
    slack_context = session_data.get('slack_context', {})

    # --- CALCULATE THE EXPIRATION TIMESTAMP ---
    expire_time = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=30)
    
    # --- ADK FIX 1: RESTORED DETERMINISTIC GUARDRAIL ---
    # Always check for a URL first. It's the fastest and most reliable signal.
    if extract_url_from_text(user_feedback):
        print("URL detected in feedback. Delegating to research worker immediately.")
        doc_ref.update({
            "status": "delegating_research",
            "last_updated": expire_time
            })
        # FIX: Persist the feedback into the events subcollection
        doc_ref.collection('events').add({
            "event_type": "user_feedback",
            "text": user_feedback,
            "images": images,
            "timestamp": datetime.datetime.now(datetime.timezone.utc)
        })
        dispatch_task({"session_id": session_id, "topic": session_data.get('topic'), "feedback_text": user_feedback, "slack_context": slack_context, "images": images}, STORY_WORKER_URL)
        return jsonify({"msg": "Delegated (URL detected)"}), 200

    # --- ADK FIX 2: CONTEXTUAL LLM TRIAGE ---
    # FIX: Read from 'events' subcollection instead of 'event_log' array
    events_ref = doc_ref.collection('events')
    
    # MEMORY EXPANSION: Removed .limit(5) to ensure context isn't lost during feedback loops
    recent_events_query = events_ref.order_by('timestamp', direction=firestore.Query.DESCENDING)


    # 2. Stream and then REVERSE the list to get chronological order (Oldest -> Newest)
    recent_events = [doc.to_dict() for doc in recent_events_query.stream()][::-1]
    
    # MEMORY EXPANSION: Removed char limits (was 500) to ensure full context for triage
    formatted_history = []
    for e in recent_events:
        etype = e.get('event_type', 'unknown')
        # Try all possible content fields
        content = e.get('text') or e.get('content') or e.get('payload') or str(e.get('data', ''))
        formatted_history.append(f"Turn ({etype}): {str(content)}")
    
    history_text = "\n".join(formatted_history)

    # --- ADK FIX 3: ROBUST APPROVAL DETECTION ---
    # Fast-path for single-word approvals
    tokens = re.sub(r'[^\w\s]', '', user_feedback).strip().lower().split()
    is_quick_approval = len(tokens) <= 3 and any(t in ["approved", "approve"] for t in tokens)

    if is_quick_approval:
        intent = "APPROVE"
        print(f"Quick Approval detected. Forcing APPROVE intent.")
    else:
        feedback_triage_prompt = f"""
        Analyze the user's latest message in the context of the conversation history. Classify the user's INTENT into one of three categories:

        1.  **APPROVE**: The user is explicitly confirming, finalizing, or "approving" the current result for the record or knowledge base.
            *   *Examples:* "I approve on the ideas in this conversation.", "Approved!", "This is perfect, save it.", "I like this, we are done."

        2.  **REFINE**: The user is asking for a change to the agent's current strategy or last structured proposal (e.g., a 'Then vs Now' draft).
            *   *Examples:* "Make it shorter", "Can you change the tone?", "Add a point about X"

        3.  **DELEGATE**: The user is asking a new factual question, confirming a proposed research direction, or asking a "meta" question. 
            *   **CRITICAL:** If the user is saying "Yes", "Proceed", "Looks good", or confirming a suggestion for *further research*, select **DELEGATE**.

        CONVERSATION HISTORY:
        {history_text}

        USER'S LATEST MESSAGE: "{user_feedback}"

        Respond with ONLY the category name (APPROVE, REFINE, or DELEGATE).
        """
        intent = unimodel.generate_content(feedback_triage_prompt).text.strip().upper()
    
    print(f"Feedback Triage classified intent as: {intent}")
    
    # --- 3. STATE-AWARE ROUTING LOGIC ---
    if intent == "DELEGATE":
        # This now handles all factual questions, meta-questions, and simple chat.
        print("Intent requires new research or is conversational. Delegating to story worker...")
        doc_ref.update({
            "status": "delegating_research",
            "last_updated": expire_time
            })
        # FIX: Persist the feedback into the events subcollection
        doc_ref.collection('events').add({
            "event_type": "user_feedback",
            "text": user_feedback,
            "images": req.get('images', []),
            "timestamp": datetime.datetime.now(datetime.timezone.utc)
        })
        # No-Strip: Pass full payload forward, but preserve mission topic
        payload = req.copy()
        # FIX: Ensure we pass the CURRENT turn's context, not the stale session context
        slack_context.update({
            "ts": req.get('slack_ts'),
            "thread_ts": req.get('slack_thread_ts')
        })
        payload.update({"topic": session_data.get('topic'), "feedback_text": user_feedback, "slack_context": slack_context})
        dispatch_task(payload, STORY_WORKER_URL)
        return jsonify({"msg": "Delegated to Research Worker"}), 200
        
    elif intent == "APPROVE":
        # Final safety check: We proceed if either quick-approval was hit OR LLM classified as APPROVE
        print(f"Executing APPROVE path for: {user_feedback}")

        ts = datetime.datetime.now(datetime.timezone.utc)
        user_event = {"event_type": "user_feedback", "text": user_feedback, "intent": "APPROVE", "timestamp": ts}

        # FIX: Fetch FULL history from subcollection to find the proposal
        full_history = [doc.to_dict() for doc in events_ref.order_by('timestamp').stream()]
        # --- ROBUST ARTIFACT EXTRACTION (Avoids Old "Then-and-Now" Rants) ---
        target_content = None
        
        # We go backwards. The FIRST high-fidelity thing we hit is our target.
        for event in reversed(full_history):
            etype = event.get('event_type')
            
            # CATEGORY A: Direct Artifacts (Text already exists)
            # This includes blog posts, topic clusters, and direct answers
            if etype in ['agent_answer', 'agent_proposal', 'loop_draft']:
                # Prefer text, then data/payload
                raw_data = event.get('proposal_data') or event.get('payload') or event.get('data')
                target_content = event.get('text') or event.get('content')
                
                if not target_content and raw_data:
                    target_content = json.dumps(raw_data, indent=2) if isinstance(raw_data, (dict, list)) else str(raw_data)
                
                if target_content:
                    print(f"Found Recent Artifact: {etype}")
                    break

            # CATEGORY B: Deferred Synthesis (Requires calling a tool)
            # Only synthesize if this was the most recent professional event
            elif etype == 'adk_request_confirmation' and event.get('payload'):
                print("Found Recent Request for Synthesis (Then-and-Now). Synthesizing...")
                target_content = tell_then_and_now_story(event['payload'], tool_confirmation={"confirmed": True})
                break

        # Pass 2: Social Fallback
        if not target_content:
            for event in reversed(full_history):
                if event.get('event_type') == 'agent_reply':
                    target_content = event.get('text') or event.get('content')
                    if target_content: break
        
        if not target_content:
            return jsonify({"msg": "Nothing to approve found."}), 404

        # --- RAG INGESTION ---
        if INGEST_KNOWLEDGE_URL:
             # Prepend Mission Context for better future retrieval
            rag_content = f"MISSION: {session_data.get('topic', 'General Inquiry')}\n\nCONTENT:\n{target_content}"
            dispatch_task({"session_id": session_id, "topic": session_data.get('topic'), "story": rag_content}, INGEST_KNOWLEDGE_URL)
        
        # FIX: Update parent status, but write events to subcollection
        doc_ref.update({
            "status": "completed", 
            "final_story": target_content,
            "last_updated": expire_time
        })
        
        events_ref.add(user_event)
        events_ref.add({"event_type": "final_output", "content": target_content, "timestamp": datetime.datetime.now(datetime.timezone.utc)})
        
        requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={"session_id": session_id, "proposal": [{"link": target_content}], "thread_ts": slack_context.get('ts'), "channel_id": slack_context.get('channel'), "is_final_story": True, "is_initial_post": False }, verify=certifi.where(), timeout=15)
        return jsonify({"msg": "Approved and Ingested"}), 200

    elif intent == "REFINE":
        # FIX: Fetch history from subcollection
        full_history = [doc.to_dict() for doc in events_ref.order_by('timestamp').stream()]
        
        # --- ROBUST ARTIFACT EXTRACTION (Same as APPROVE path) ---
        last_prop_data = None
        for event in reversed(full_history):
            etype = event.get('event_type')
            # Look for structured data in common keys
            data = event.get('proposal_data') or event.get('payload') or event.get('data')
            if data and etype in ['agent_proposal', 'loop_draft', 'adk_request_confirmation', 'agent_answer']:
                last_prop_data = data
                print(f"Refine: Found target artifact in '{etype}' event.")
                break
        
        if not last_prop_data:
            print("Refine intent found, but no usable technical artifact exists in history. Delegating...")
            # No-Strip: Pass full payload forward
            payload = req.copy()
            payload.update({"topic": session_data.get('topic'), "feedback_text": user_feedback})
            dispatch_task(payload, STORY_WORKER_URL)
            return jsonify({"msg": "Delegated (Refine Fallback)"}), 200

        # Optimization: If it's a dict/list, we refine it. If it's raw text, we treat it as the 'current_proposal'
        new_prop = refine_proposal(session_data.get('topic'), last_prop_data, user_feedback)
        new_id = f"approval_{uuid.uuid4().hex[:8]}"
        
        # FIX: Write to subcollection
        ts = datetime.datetime.now(datetime.timezone.utc)
        events_ref.add({"event_type": "user_feedback", "text": user_feedback, "intent": "REFINE", "timestamp": ts})
        events_ref.add({"event_type": "agent_proposal", "proposal_data": new_prop, "timestamp": ts})
        events_ref.add({"event_type": "adk_request_confirmation", "approval_id": new_id, "payload": new_prop['interlinked_concepts'], "timestamp": ts})
        
        doc_ref.update({"last_updated": expire_time})
        
        requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={
            "session_id": session_id, 
            "approval_id": new_id, 
            "proposal": new_prop['interlinked_concepts'], 
            "thread_ts": slack_context.get('ts'), 
            "channel_id": slack_context.get('channel'),
            "is_final_story": False,
            "is_initial_post": False
        }, verify=certifi.where(), timeout=15)

    return jsonify({"message": "Refinement processed."}), 200