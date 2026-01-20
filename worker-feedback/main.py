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

model = None
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
    return model.generate_content(f"Tell a 'Then and Now' story using: {interlinked_concepts}").text

def refine_proposal(topic, current_proposal, critique):
    return extract_json(model.generate_content(f"REWRITE proposal... Draft: {json.dumps(current_proposal)}...").text)

# --- THE STATEFUL AND HARDENED FEEDBACK WORKER ---
@functions_framework.http
def process_feedback_logic(request):
    global model, db
    if model is None: 
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        model = GenerativeModel(MODEL_NAME)
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

    # --- ADK FIX 3: STRICT VERBATIM APPROVAL ---
    # We strip common formatting and check for the literal "approved" (case-insensitive)
    sanitized_feedback = user_feedback.strip().strip('*_~').lower()
    is_strict_approval = sanitized_feedback == "approved"

    if is_strict_approval:
        intent = "APPROVE"
        print("Literal 'approved' detected. Forcing APPROVE intent.")
    else:
        feedback_triage_prompt = f"""
        Analyze the user's latest message in the context of the conversation history. Classify the user's INTENT into one of two categories:

        1.  **REFINE**: The user is asking for a change to the agent's current strategy or last structured proposal (e.g., a 'Then vs Now' draft).
            *   *Examples:* "Make it shorter", "Can you change the tone?", "Add a point about X, Why does X look like X when it should look like Y?"

        2.  **DELEGATE**: The user is asking a new factual question, confirming a proposed research direction, or asking a "meta" question about the agent's behavior. This category is for anything that requires the research agent to keep working.
            *   **CRITICAL INCLUSION:** If the user is saying "Yes", "Proceed", "Looks good", or confirming a suggestion for further research or development, select **DELEGATE**.
            *   *Examples:* "What's trending in social media?", "Yes, please do that research.", "Demystify job impact.", "Thanks!", "Makes sense."

        CONVERSATION HISTORY:
        {history_text}

        USER'S LATEST MESSAGE: "{user_feedback}"

        Respond with ONLY the category name (REFINE or DELEGATE).
        """
        intent = model.generate_content(feedback_triage_prompt).text.strip().upper()
    
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
        # Final safety check: We ONLY proceed with finalization/ingestion if it was a strict literal approval
        if not is_strict_approval:
            print("Internal Error: APPROVE intent reached without strict literal match. Falling back to DELEGATE.")
            # No-Strip: Pass full payload forward
            payload = req.copy()
            payload.update({"topic": session_data.get('topic'), "feedback_text": user_feedback})
            dispatch_task(payload, STORY_WORKER_URL)
            return jsonify({"msg": "Fallback to research (non-verbatim approval)"}), 200

        ts = datetime.datetime.now(datetime.timezone.utc)
        user_event = {"event_type": "user_feedback", "text": user_feedback, "intent": "APPROVE", "timestamp": ts}

        # FIX: Fetch FULL history from subcollection to find the proposal
        full_history = [doc.to_dict() for doc in events_ref.order_by('timestamp').stream()]
        target_content = None
        
        for event in reversed(full_history):
            if event.get('event_type') == 'adk_request_confirmation':
                target_content = tell_then_and_now_story(event['payload'], tool_confirmation={"confirmed": True})
                break
            elif event.get('event_type') in ['agent_answer', 'agent_proposal', 'agent_reply']:
                target_content = event.get('text') or event.get('content') or str(event.get('data', ''))
                break
        
        if not target_content:
            return jsonify({"msg": "Nothing to approve found."}), 404

        if INGEST_KNOWLEDGE_URL:
            dispatch_task({"session_id": session_id, "topic": session_data.get('topic'), "story": target_content}, INGEST_KNOWLEDGE_URL)
        
        # FIX: Update parent status, but write events to subcollection
        doc_ref.update({
            "status": "completed", 
            "final_story": target_content,
            "last_updated": expire_time
        })
        
        events_ref.add(user_event)
        events_ref.add({"event_type": "final_output", "content": target_content, "timestamp": datetime.datetime.now(datetime.timezone.utc)})
        
        requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={"session_id": session_id, "proposal": [{"link": target_content}], "thread_ts": slack_context.get('ts'), "channel_id": slack_context.get('channel'), "is_final_story": True, "is_initial_post": False }, verify=certifi.where())
        return jsonify({"msg": "Approved and Ingested"}), 200

    elif intent == "REFINE":
        # FIX: Fetch history from subcollection
        full_history = [doc.to_dict() for doc in events_ref.order_by('timestamp').stream()]
        last_prop = next((e for e in reversed(full_history) if e.get('proposal_data')), None)
        
        if not last_prop:
            print("Refine intent found, but no proposal exists. Delegating to story worker for clarification.")
            # No-Strip: Pass full payload forward
            payload = req.copy()
            payload.update({"topic": session_data.get('topic'), "feedback_text": user_feedback})
            dispatch_task(payload, STORY_WORKER_URL)
            return jsonify({"msg": "Delegated (Refine Fallback)"}), 200

        new_prop = refine_proposal(session_data.get('topic'), last_prop['proposal_data'], user_feedback)
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
        }, verify=certifi.where())

    return jsonify({"message": "Refinement processed."}), 200