# --- /dispatcher-feedback/main.py ---
import functions_framework
from flask import jsonify
import os
import json
from google.cloud import tasks_v2

PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
LOCATION = os.environ.get("GCP_LOCATION")
QUEUE_NAME = "story-worker-queue"
FEEDBACK_WORKER_URL = os.environ.get("FEEDBACK_WORKER_URL")
FUNCTION_IDENTITY_EMAIL = os.environ.get("FUNCTION_IDENTITY_EMAIL")

tasks_client = None

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

@functions_framework.http
def handle_feedback_workflow(request):
    req = request.get_json(silent=True)
    # Simple pass-through
    dispatch_task({"session_id": req['session_id'], "feedback": req['feedback']}, FEEDBACK_WORKER_URL)
    return jsonify({"msg": "Feedback accepted"}), 202