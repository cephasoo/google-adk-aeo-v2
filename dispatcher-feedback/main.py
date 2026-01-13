# --- /dispatcher-feedback/main.py ---
import functions_framework
from flask import jsonify
import os
import json
from google.cloud import tasks_v2

PROJECT_ID = os.environ.get("PROJECT_ID")
LOCATION = os.environ.get("LOCATION")
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
    if isinstance(req, list): req = req[0]
    
    # NORMALIZE: Ensure worker-compatible context exists without stripping original payload
    if 'slack_context' not in req:
        req['slack_context'] = {
            "ts": req.get('slack_ts'),
            "thread_ts": req.get('slack_thread_ts'),
            "channel": req.get('slack_channel')
        }
        
    dispatch_task(req, FEEDBACK_WORKER_URL)
    return jsonify({"msg": "Feedback accepted"}), 202