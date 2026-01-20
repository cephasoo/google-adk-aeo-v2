# --- /dispatcher-story/main.py ---
import functions_framework
from flask import jsonify
import vertexai
from vertexai.generative_models import GenerativeModel
import os
import json
import uuid
import re
import certifi
import requests
from google.cloud import firestore, tasks_v2
import datetime

# --- Configuration ---
PROJECT_ID = os.environ.get("PROJECT_ID")
LOCATION = os.environ.get("LOCATION")
MODEL_NAME = os.environ.get("MODEL_NAME")
N8N_PROPOSAL_WEBHOOK_URL = os.environ.get("N8N_PROPOSAL_WEBHOOK_URL") 
QUEUE_NAME = "story-worker-queue"

# Defines where the work goes (The Private Worker URL)
STORY_WORKER_URL = os.environ.get("STORY_WORKER_URL") 
FUNCTION_IDENTITY_EMAIL = os.environ.get("FUNCTION_IDENTITY_EMAIL")

# --- Global Clients ---
model = None
db = None
tasks_client = None

# --- Minimal Utils Needed for Dispatch ---
def extract_url_from_text(text):
    """Finds and cleans the first URL in a string."""
    url_pattern = r'(https?://[^\s<>|]+)'
    match = re.search(url_pattern, text)
    if match: return match.group(1)
    return None

def dispatch_task(payload, target_url):
    global tasks_client
    if tasks_client is None: tasks_client = tasks_v2.CloudTasksClient()
    parent = tasks_client.queue_path(PROJECT_ID, LOCATION, QUEUE_NAME)

    # --- PRODUCTION HARDENING ---
    # Add a 10-minute deadline to the task.
    # This tells Cloud Tasks to stop retrying after 600 seconds.
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

# --- The Function ---
@functions_framework.http
def start_story_workflow(request):
    global model, db
    if model is None: 
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        model = GenerativeModel(MODEL_NAME)
    if db is None: db = firestore.Client()

    request_json = request.get_json(silent=True)
    if isinstance(request_json, list): request_json = request_json[0]
    text_input = request_json.get('topic')
    images = request_json.get('images', []) # Expecting a list of strings (URLs or Base64)
    slack_context = { 
        "ts": request_json.get('slack_ts'), 
        "thread_ts": request_json.get('slack_thread_ts'), 
        "channel": request_json.get('slack_channel') 
    }

    if not text_input and not images: return jsonify({"error": "Invalid request"}), 400
    
    session_id = str(uuid.uuid4())

    # --- CALCULATE THE EXPIRATION TIMESTAMP ---
    expire_time = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=30)
    
    # 1. Deterministic Routing (Regex)
    has_url = extract_url_from_text(text_input or "") is not None
    intent = "SOCIAL" 
    
    if has_url or images:
        print(f"Routing: URL or Image detected -> Forcing WORK mode.")
        intent = "WORK"
    else:
        # 2. AI Triage (Only if no URL/Image)
        triage_prompt = f"""
        Classify the user's intent based on the definitions below.
        
        DEFINITIONS:
        - WORK: Requests for information, research, news, trends, 'viral' topics, strategies, drafts, analysis, or review.
          (e.g., "What is trending?", "Tell me about X", "Draft a story", "Who is...", "Viral in Nigeria").
        - SOCIAL: Pure small talk, greetings, compliments, or gratitude WITHOUT an information request.
          (e.g., "Hi", "Thanks", "Good job", "You are funny", "God bless you").
          
        USER INPUT: "{text_input}"
        
        RESPOND ONLY WITH "SOCIAL" OR "WORK".
        """
        intent = model.generate_content(triage_prompt).text.strip().upper()
    
    if "SOCIAL" in intent:
        social_prompt = f"User: '{text_input}'. Respond naturally/wittily. Keep it brief."
        reply = model.generate_content(social_prompt).text.strip()
        
        # --- FIX: WRITE TO SUBCOLLECTION (Prevent 1MB Crash) ---
        session_ref = db.collection('agent_sessions').document(session_id)
        
        # 1. Set Parent Metadata (No 'event_log' array)
        session_ref.set({
            "status": "completed", 
            "type": "social", 
            "topic": "Social", 
            "slack_context": slack_context,
            "last_updated": expire_time
        })

        # 2. Write Events to Subcollection
        batch = db.batch()
        ts = datetime.datetime.now(datetime.timezone.utc)
        
        event_1 = {"event_type": "social", "text": text_input, "timestamp": ts}
        event_2 = {"event_type": "agent_reply", "text": reply, "timestamp": ts}
        
        # Use .document() to generate auto-ID
        batch.set(session_ref.collection('events').document(), event_1)
        batch.set(session_ref.collection('events').document(), event_2)
        batch.commit()
        
        # Send Reply via Webhook (Adding timeout to prevent hanging)
        try:
            requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={"session_id": session_id, "type": "social", "message": reply, "channel_id": slack_context['channel'], "thread_ts": slack_context['ts']}, verify=certifi.where(), timeout=10)
        except Exception as e:
            print(f"Warning: Failed to send webhook to n8n: {str(e)}")
            
        return jsonify({"msg": "Accepted", "session_id": session_id}), 200
    else:
        # Dispatch to Worker (No-Strip Policy: Copy full payload)
        payload = request_json.copy()
        payload.update({
             "session_id": session_id,
             "topic": text_input, # Ensure sanitized or original topic is clear
             "slack_context": slack_context
        })
        
        session_ref = db.collection('agent_sessions').document(session_id)
        session_ref.set({
             "status": "starting", 
             "type": "work_answer", 
             "topic": text_input,
             "images": images,
             "slack_context": slack_context,
             "last_updated": expire_time
        })

        # --- PERSIST INITIAL EVENT ---
        # Fixed: Ensure the first 'WORK' turn is also in the events subcollection
        session_ref.collection('events').add({
            "event_type": "user_request",
            "text": text_input,
            "images": images,
            "timestamp": datetime.datetime.now(datetime.timezone.utc)
        })
        
        dispatch_task(payload, STORY_WORKER_URL)
        return jsonify({"type": "work", "msg": "Accepted", "session_id": session_id}), 202
