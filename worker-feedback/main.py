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
import requests
from google.cloud import firestore, tasks_v2

# --- Config ---
PROJECT_ID = os.environ.get("PROJECT_ID")
LOCATION = os.environ.get("LOCATION")
MODEL_NAME = os.environ.get("MODEL_NAME")
N8N_PROPOSAL_WEBHOOK_URL = os.environ.get("N8N_PROPOSAL_WEBHOOK_URL") 
STORY_WORKER_URL = os.environ.get("STORY_WORKER_URL") # Needed for Delegation!
QUEUE_NAME = "story-worker-queue"
FUNCTION_IDENTITY_EMAIL = os.environ.get("FUNCTION_IDENTITY_EMAIL")
PROPOSAL_KEYWORDS = ["outline", "draft", "proposal", "story", "brief"]

model = None
db = None
tasks_client = None

# --- Utils ---
def extract_json(text): # ... (Same as above) ...
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match: raise ValueError(f"No JSON found")
    return json.loads(match.group(0))

def extract_url_from_text(text): # ... (Same as above) ...
    url_pattern = r'(https?://[^\s<>|]+)'
    match = re.search(url_pattern, text)
    if match: return match.group(1)
    return None

def dispatch_task(payload, target_url):
    global tasks_client
    if tasks_client is None: tasks_client = tasks_v2.CloudTasksClient()
    parent = tasks_client.queue_path(PROJECT_ID, LOCATION, QUEUE_NAME)
    task = {
        "http_request": {
            "http_method": tasks_v2.HttpMethod.POST,
            "url": target_url,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(payload).encode(),
            "oidc_token": {"service_account_email": FUNCTION_IDENTITY_EMAIL}
        }
    }
    tasks_client.create_task(request={"parent": parent, "task": task})

def classify_feedback_intent(feedback_text):
    global model
    prompt = f"Classify User Feedback: '{feedback_text}'. Respond ONLY with: APPROVE or REFINE."
    response = model.generate_content(prompt).text.strip().upper()
    if "APPROVE" in response: return "APPROVE"
    return "REFINE"

# --- Tools needed for final proposal steps ---
def tell_then_and_now_story(interlinked_concepts, tool_confirmation=None):
    if not tool_confirmation or not tool_confirmation.get("confirmed"): raise PermissionError("Wait for approval.")
    return model.generate_content(f"Tell a 'Then and Now' story using: {interlinked_concepts}").text

def refine_proposal(topic, current_proposal, critique):
    return extract_json(model.generate_content(f"REWRITE proposal... Draft: {json.dumps(current_proposal)}...").text)
def generate_comprehensive_answer(topic, context):
    return model.generate_content(f"Answer: {topic}. Context: {context}").text.strip()


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
    session_type = session_data.get('type', 'work') 

    # 1. Skill Router (Delegation) logic
    if session_type in ['social', 'work_answer']:
        url_in_feedback = extract_url_from_text(user_feedback)
        is_graduation = any(w in user_feedback.lower() for w in PROPOSAL_KEYWORDS)
        
        if url_in_feedback or is_graduation:
            print(f"Delegation Triggered.")
            target_topic = ""
            if url_in_feedback:
                target_topic = user_feedback
                doc_ref.update({"type": "work_answer", "status": "working_on_grounding"})
            else:
                history = session_data.get('event_log', [])
                history_text = "\n".join([f"{e.get('event_type')}: {e.get('text')}" for e in history[-5:]])
                ext_prompt = f"The user wants a proposal. HISTORY: {history_text}. USER: '{user_feedback}'. Identify SUBJECT MATTER."
                target_topic = model.generate_content(ext_prompt).text.strip()
                doc_ref.update({"type": "work_proposal", "topic": target_topic, "status": "awaiting_approval"})

            dispatch_task({"session_id": session_id, "topic": target_topic, "slack_context": slack_context}, STORY_WORKER_URL)
            return jsonify({"msg": "Delegated"}), 200

        # Normal Chat
        history = session_data.get('event_log', [])
        context_text = "\n".join([f"{e.get('event_type')}: {e.get('text')}" for e in history[-7:]])
        reply = model.generate_content(f"Reply to user in context:\n{context_text}\nUser: {user_feedback}").text.strip()
        doc_ref.update({"event_log": firestore.ArrayUnion([{"event_type": "user_feedback", "text": user_feedback}, {"event_type": "agent_reply", "text": reply}])})
        requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={"session_id": session_id, "type": "social", "message": reply, "thread_ts": slack_context.get('ts'), "channel_id": slack_context.get('channel')}, verify=True )
        return jsonify({"msg": "reply sent"}), 200

    # 2. Proposal Loop Logic (Approve/Refine/Question)
    intent = classify_feedback_intent(user_feedback)
    user_event = {"event_type": "user_feedback", "text": user_feedback, "intent": intent, "timestamp": str(datetime.datetime.now())}

    if intent == "APPROVE":
        pending_event = next((e for e in reversed(session_data.get('event_log', [])) if e.get('event_type') == 'adk_request_confirmation'), None)
        story = tell_then_and_now_story(pending_event['payload'], tool_confirmation={"confirmed": True})
        doc_ref.update({"status": "completed", "final_story": story, "event_log": firestore.ArrayUnion([user_event, {"event_type": "story", "content": story}])})
        requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={"session_id": session_id, "proposal": [{"link": story}], "thread_ts": slack_context.get('ts'), "channel_id": slack_context.get('channel'), "is_final_story": True}, verify=True )
        
    elif intent == "QUESTION":
        answer_text = generate_comprehensive_answer(user_feedback, "User pivoting to question.")
        doc_ref.update({"type": "work_answer", "status": "awaiting_feedback", "event_log": firestore.ArrayUnion([user_event, {"event_type": "agent_answer", "text": answer_text}])})
        requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={"session_id": session_id, "type": "social", "message": answer_text, "thread_ts": slack_context.get('ts'), "channel_id": slack_context.get('channel')}, verify=True )

    else: # REFINE
        last_prop = next((e for e in reversed(session_data.get('event_log', [])) if e.get('proposal_data')), None)
        new_prop = refine_proposal(session_data.get('topic'), last_prop['proposal_data'], user_feedback)
        new_id = f"approval_{uuid.uuid4().hex[:8]}"
        doc_ref.update({"event_log": firestore.ArrayUnion([user_event, {"event_type": "agent_proposal", "proposal_data": new_prop}, {"event_type": "adk_request_confirmation", "approval_id": new_id, "payload": new_prop['interlinked_concepts']}])})
        requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={"session_id": session_id, "approval_id": new_id, "proposal": new_prop['interlinked_concepts'], "thread_ts": slack_context.get('ts'), "channel_id": slack_context.get('channel')}, verify=True )

    return jsonify({"message": "Feedback processed."}), 200