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

def classify_feedback_intent(feedback_text):
    global model
    prompt = f"""
    Classify User Feedback: '{feedback_text}'. 
    Categories:
    1. APPROVE: User explicitly accepts the draft (e.g., "Looks good", "Proceed", "Yes").
    2. REFINE: User wants changes to the current draft (e.g., "Make it shorter", "Change the tone").
    3. QUESTION: User is asking a question or starting a new topic (e.g., "What is X?", "Can you identify themes?").
    
    Respond ONLY with the category name.
    """
    response = model.generate_content(prompt).text.strip().upper()
    if "APPROVE" in response: return "APPROVE"
    if "REFINE" in response: return "REFINE"
    return "QUESTION" # Default fallback

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
    
    # --- 0. PRIORITY CHECK: URL DELEGATION ---
    # Always check for URLs first. If found, delegate immediately.
    url_in_feedback = extract_url_from_text(user_feedback)
    if url_in_feedback:
        doc_ref.update({"type": "work_answer", "status": "working_on_grounding"})
        dispatch_task({"session_id": session_id, "topic": user_feedback, "slack_context": slack_context}, STORY_WORKER_URL)
        return jsonify({"msg": "Delegated (URL detected)"}), 200

    # --- 1. CLASSIFY INTENT ---
    intent = classify_feedback_intent(user_feedback)
    print(f"User Feedback Intent Classified as: {intent}")

    # --- 2. HANDLE QUESTIONS / CHAT ---
    if intent == "QUESTION":
        history = session_data.get('event_log', [])
        # Provide context so it can answer questions about the proposal
        context_text = "\n".join([f"{e.get('event_type')}: {e.get('text') or e.get('payload')}" for e in history[-5:]])
        
        answer_text = generate_comprehensive_answer(user_feedback, context_text)
        
        doc_ref.update({
            "type": "work_answer", 
            "status": "awaiting_feedback", 
            "event_log": firestore.ArrayUnion([
                {"event_type": "user_feedback", "text": user_feedback, "intent": "QUESTION"}, 
                {"event_type": "agent_answer", "text": answer_text}
            ])
        })
        requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={
            "session_id": session_id, 
            "type": "social", 
            "message": answer_text, 
            "thread_ts": slack_context.get('ts'), 
            "channel_id": slack_context.get('channel'),
            "is_final_story": False # <--- EXPLICITLY KEEP OPEN
        }, verify=True)
        return jsonify({"msg": "Question answered"}), 200

    # --- 3. HANDLE APPROVAL ---
    if intent == "APPROVE":
        user_event = {"event_type": "user_feedback", "text": user_feedback, "intent": intent, "timestamp": str(datetime.datetime.now())}
        history = session_data.get('event_log', [])
        target_content = None
        
        # Look backwards for content to approve
        for event in reversed(history):
            if event.get('event_type') == 'adk_request_confirmation':
                target_content = tell_then_and_now_story(event['payload'], tool_confirmation={"confirmed": True})
                break
            elif event.get('event_type') == 'agent_answer':
                target_content = event.get('text')
                break
        
        if not target_content:
            # Fallback if approval is premature
            requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={
                "session_id": session_id, 
                "type": "social", 
                "message": "I'm ready to write, but I couldn't find the draft. Can you ask me to generate it again?", 
                "thread_ts": slack_context.get('ts'), 
                "channel_id": slack_context.get('channel'),
                "is_final_story": False # <--- EXPLICITLY KEEP OPEN
            }, verify=True)
            return jsonify({"msg": "Nothing to approve found."}), 200

        # Ingest
        if INGEST_KNOWLEDGE_URL:
            dispatch_task({"session_id": session_id, "topic": session_data.get('topic'), "story": target_content}, INGEST_KNOWLEDGE_URL)
        
        # Update DB
        doc_ref.update({
            "status": "completed", 
            "final_story": target_content, 
            "event_log": firestore.ArrayUnion([user_event, {"event_type": "final_output", "content": target_content}])
        })

        # Notify Slack - THIS IS THE ONLY PLACE "is_final_story" IS TRUE
        requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={
            "session_id": session_id, 
            "proposal": [{"link": target_content}], 
            "thread_ts": slack_context.get('ts'), 
            "channel_id": slack_context.get('channel'), 
            "is_final_story": True 
        }, verify=True)
        
        return jsonify({"msg": "Approved and Ingested"}), 200

    # --- 4. HANDLE REFINEMENT / DELEGATION (Default) ---
    # This block handles "REFINE" or if keywords imply a new draft task
    is_graduation = any(w in user_feedback.lower() for w in PROPOSAL_KEYWORDS)
    
    if is_graduation:
        # User wants a new outline/draft -> Delegate to Story Worker
        doc_ref.update({"type": "work_proposal", "status": "awaiting_approval"})
        dispatch_task({"session_id": session_id, "topic": user_feedback, "slack_context": slack_context}, STORY_WORKER_URL)
        return jsonify({"msg": "Delegated (Keywords detected)"}), 200
    
    else:
        # Standard Refinement (Edit the existing JSON/Proposal)
        last_prop = next((e for e in reversed(session_data.get('event_log', [])) if e.get('proposal_data')), None)
        
        if not last_prop:
            # Fallback if no proposal exists to refine -> Treat as Question/Chat
            # We recurse to standard chat logic manually here
            answer_text = generate_comprehensive_answer(user_feedback, "User wants refinement but no proposal exists.")
            requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={
                "session_id": session_id, 
                "type": "social", 
                "message": answer_text, 
                "thread_ts": slack_context.get('ts'), 
                "channel_id": slack_context.get('channel'),
                "is_final_story": False # <--- EXPLICITLY KEEP OPEN
            }, verify=True)
            return jsonify({"msg": "Fallback chat"}), 200

        new_prop = refine_proposal(session_data.get('topic'), last_prop['proposal_data'], user_feedback)
        new_id = f"approval_{uuid.uuid4().hex[:8]}"
        
        doc_ref.update({
            "event_log": firestore.ArrayUnion([
                {"event_type": "user_feedback", "text": user_feedback, "intent": "REFINE"}, 
                {"event_type": "agent_proposal", "proposal_data": new_prop}, 
                {"event_type": "adk_request_confirmation", "approval_id": new_id, "payload": new_prop['interlinked_concepts']}
            ])
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