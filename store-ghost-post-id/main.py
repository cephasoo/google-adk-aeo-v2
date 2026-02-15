# --- /worker-story/store_ghost_post_id.py ---
import functions_framework
from flask import jsonify
from google.cloud import firestore
import datetime
import os

# --- Configuration ---
PROJECT_ID = os.environ.get("PROJECT_ID")

# Initialize Firestore
db = firestore.Client(project=PROJECT_ID)

@functions_framework.http
def store_ghost_post_id(request):
    """
    Webhook endpoint for N8N to send Ghost post ID after creation.
    Called by N8N after successfully creating a Ghost CMS draft.
    
    Expected payload:
    {
        "session_id": "uuid-string",
        "ghost_post_id": "ghost-post-id-string"
    }
    """
    # CORS handling for N8N webhook
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)
    
    headers = {'Access-Control-Allow-Origin': '*'}
    
    try:
        # Parse request data
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "No JSON payload provided"}), 400, headers
        
        # N8N sends payload as array [{...}], extract first element
        if isinstance(data, list):
            if len(data) == 0:
                return jsonify({"error": "Empty payload array"}), 400, headers
            data = data[0]
        
        session_id = data.get('session_id')
        ghost_post_id = data.get('ghost_post_id')
        
        # Validate required fields
        if not session_id or not ghost_post_id:
            return jsonify({
                "error": "Missing required fields",
                "required": ["session_id", "ghost_post_id"],
                "received": list(data.keys())
            }), 400, headers
        
        # Store in Firestore
        session_ref = db.collection('agent_sessions').document(session_id)
        
        # Check if session exists
        session_doc = session_ref.get()
        if not session_doc.exists:
            return jsonify({
                "error": f"Session not found: {session_id}"
            }), 404, headers
        
        # Update session with Ghost post ID
        session_ref.update({
            "ghost_post_id": ghost_post_id,
            "ghost_post_created_at": datetime.datetime.now(datetime.timezone.utc),
            "last_updated": datetime.datetime.now(datetime.timezone.utc)
        })
        
        print(f"✅ Stored Ghost post ID: {ghost_post_id} for session: {session_id}")
        
        return jsonify({
            "msg": "Ghost post ID stored successfully",
            "session_id": session_id,
            "ghost_post_id": ghost_post_id
        }), 200, headers
        
    except Exception as e:
        print(f"⚠️ Failed to store Ghost post ID: {e}")
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500, headers
