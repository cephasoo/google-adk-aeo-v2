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
PROPOSAL_KEYWORDS = ["outline", "draft", "proposal", "story", "brief"]
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
    session_id = req['session_id']
    user_feedback = req['feedback']
    
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
        dispatch_task({"session_id": session_id, "topic": user_feedback, "slack_context": slack_context}, STORY_WORKER_URL)
        return jsonify({"msg": "Delegated (URL detected)"}), 200

    # --- ADK FIX 2: CONTEXTUAL LLM TRIAGE ---
    # If no URL, use the LLM to understand the nuanced intent.
    history_text = "\n".join([f"Turn ({e.get('event_type')}): {e.get('text', '')[:200]}" for e in session_data.get('event_log', [])[-5:]])
    feedback_triage_prompt = f"""
    Analyze the user's latest message in the context of the conversation history. Classify the user's INTENT into one of three categories:

    1.  **APPROVE**: The user is giving explicit approval to finalize the last output.
        *Examples:* "Looks good", "Perfect, proceed", "Yes, finalize this."

    2.  **REFINE**: The user is asking for a change to the agent's last structured proposal (e.g., a 'Then vs Now' draft).
        *Examples:* "Make it shorter", "Can you change the tone?", "Add a point about X."

    3.  **DELEGATE**: The user is asking a new factual question, a "meta" question about the agent's behavior, or a request that requires new research. This requires handing off to the specialist research agent.
        *Examples:* "What's trending in social media?", "Does that explain [feature X]?", "You hallucinated earlier.", "Can you summarize this job posting?", "Thanks!"

    CONVERSATION HISTORY:
    {history_text}

    USER'S LATEST MESSAGE: "{user_feedback}"

    Respond with ONLY the category name (APPROVE, REFINE, or DELEGATE).
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
        dispatch_task({"session_id": session_id, "topic": user_feedback, "slack_context": slack_context}, STORY_WORKER_URL)
        return jsonify({"msg": "Delegated to Research Worker"}), 200
        
    elif intent == "APPROVE":
        user_event = {"event_type": "user_feedback", "text": user_feedback, "intent": "APPROVE"}
        history = session_data.get('event_log', [])
        target_content = None
        
        for event in reversed(history):
            if event.get('event_type') == 'adk_request_confirmation':
                target_content = tell_then_and_now_story(event['payload'], tool_confirmation={"confirmed": True})
                break
            elif event.get('event_type') == 'agent_answer':
                target_content = event.get('text')
                break
        
        if not target_content:
            # (Fallback logic is correct)
            return jsonify({"msg": "Nothing to approve found."}), 404

        if INGEST_KNOWLEDGE_URL:
            dispatch_task({"session_id": session_id, "topic": session_data.get('topic'), "story": target_content}, INGEST_KNOWLEDGE_URL)
        
        doc_ref.update({
            "status": "completed", "final_story": target_content, 
            "event_log": firestore.ArrayUnion([user_event, {"event_type": "final_output", "content": target_content}]),
            "last_updated": expire_time
        })
        requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={"session_id": session_id, "proposal": [{"link": target_content}], "thread_ts": slack_context.get('ts'), "channel_id": slack_context.get('channel'), "is_final_story": True }, verify=True)
        return jsonify({"msg": "Approved and Ingested"}), 200

    elif intent == "REFINE":
        last_prop = next((e for e in reversed(session_data.get('event_log', [])) if e.get('proposal_data')), None)
        
        if not last_prop:
            # If there's no proposal to refine, the intent was misclassified.
            # The safest action is to delegate it to the powerful story worker to figure out.
            print("Refine intent found, but no proposal exists. Delegating to story worker for clarification.")
            dispatch_task({"session_id": session_id, "topic": user_feedback, "slack_context": slack_context}, STORY_WORKER_URL)
            return jsonify({"msg": "Delegated (Refine Fallback)"}), 200

        new_prop = refine_proposal(session_data.get('topic'), last_prop['proposal_data'], user_feedback)
        new_id = f"approval_{uuid.uuid4().hex[:8]}"
        
        doc_ref.update({
            "event_log": firestore.ArrayUnion([
                {"event_type": "user_feedback", "text": user_feedback, "intent": "REFINE"}, 
                {"event_type": "agent_proposal", "proposal_data": new_prop}, 
                {"event_type": "adk_request_confirmation", "approval_id": new_id, "payload": new_prop['interlinked_concepts']}
            ]),
            "last_updated": expire_time
        })
        requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={
            "session_id": session_id, 
            "approval_id": new_id, 
            "proposal": new_prop['interlinked_concepts'], 
            "thread_ts": slack_context.get('ts'), 
            "channel_id": slack_context.get('channel'),
            "is_final_story": False # <--- EXPLICITLY KEEP OPEN
        }, verify=True)

    return jsonify({"message": "Refinement processed."}), 200