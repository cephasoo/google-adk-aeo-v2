# --- /dispatcher-story/main.py ---
import functions_framework
from flask import jsonify
import os
import json
import uuid
from google.cloud import firestore, tasks_v2
import datetime

# --- Configuration ---
PROJECT_ID = os.environ.get("PROJECT_ID")
LOCATION = os.environ.get("LOCATION")
QUEUE_NAME = "story-worker-queue"
STORY_WORKER_URL = os.environ.get("STORY_WORKER_URL") 
FUNCTION_IDENTITY_EMAIL = os.environ.get("FUNCTION_IDENTITY_EMAIL")

# --- Global Client ---
db = None
tasks_client = None

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
    print(f"TELEMETRY: Dispatching to Target URL: {target_url}")
    response = tasks_client.create_task(request={"parent": parent, "task": task})
    print(f"TELEMETRY: Task Created Successfully. Name: {response.name}")


@functions_framework.http
def start_story_workflow(request):
    global db
    if db is None: db = firestore.Client()

    request_json = request.get_json(silent=True)
    if not request_json: return jsonify({"error": "No JSON payload"}), 400
    if isinstance(request_json, list): request_json = request_json[0]
    
    text_input = request_json.get('topic') or request_json.get('text')
    images = request_json.get('images', [])
    slack_context = { 
        "ts": request_json.get('slack_ts') or request_json.get('ts'), 
        "thread_ts": request_json.get('slack_thread_ts') or request_json.get('thread_ts'), 
        "channel": request_json.get('slack_channel') or request_json.get('channel')
    }

    if not text_input and not images: 
        return jsonify({"error": "Invalid request"}), 400
    
    session_id = str(uuid.uuid4())
    expire_time = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=30)
    
    # Minimalist Dispatch: Hand off everything to the worker
    payload = request_json.copy()
    payload.update({
         "session_id": session_id,
         "topic": text_input,
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

    session_ref.collection('events').add({
        "event_type": "user_request",
        "text": text_input,
        "images": images,
        "timestamp": datetime.datetime.now(datetime.timezone.utc)
    })
    
    dispatch_task(payload, STORY_WORKER_URL)
    return jsonify({"type": "work", "msg": "Accepted", "session_id": session_id}), 202
