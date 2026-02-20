# --- /worker-story/store_ghost_post_id.py ---
import functions_framework
from flask import jsonify
from google.cloud import firestore, secretmanager
import datetime
import os
import jwt
import re

# --- Configuration ---
PROJECT_ID = os.environ.get("PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT")
_SECRET_CACHE = {}

def get_project_id():
    global PROJECT_ID
    if PROJECT_ID:
        return PROJECT_ID
    try:
        # standard Cloud Run environment variable
        PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if PROJECT_ID:
            return PROJECT_ID
            
        import google.auth
        _, PROJECT_ID = google.auth.default()
        return PROJECT_ID
    except Exception:
        return None

# Lazy Client Initialization
_db = None
_secret_client = None

def get_db():
    global _db
    if _db is None:
        _db = firestore.Client(project=get_project_id())
    return _db

def get_secret_client():
    global _secret_client
    if _secret_client is None:
        _secret_client = secretmanager.SecretManagerServiceClient()
    return _secret_client

# Helpers
def get_secret(secret_id):
    """Unified secret retrieval with env fallback and global caching."""
    if secret_id in _SECRET_CACHE:
        return _SECRET_CACHE[secret_id]
        
    pid = get_project_id()
    if not pid:
        # Fallback to ENV immediately if no project context
        val = os.environ.get(secret_id.upper().replace("-", "_"))
        if val: _SECRET_CACHE[secret_id] = val
        return val

    try:
        name = f"projects/{pid}/secrets/{secret_id}/versions/latest"
        client = get_secret_client()
        response = client.access_secret_version(request={"name": name})
        val = response.payload.data.decode("UTF-8")
        _SECRET_CACHE[secret_id] = val
        return val
    except Exception as e:
        print(f"Secret {secret_id} failed: {e}. Falling back to ENV.")
        val = os.environ.get(secret_id.upper().replace("-", "_"))
        if val: _SECRET_CACHE[secret_id] = val
        return val

def verify_ghost_jwt(auth_header):
    """Verifies HS256 JWT using hex-decoded Ghost Admin Secret."""
    if not auth_header or not auth_header.startswith("Ghost "):
        return False, "Missing or invalid Authorization header format"
    
    token = auth_header.split(" ")[1]
    
    # Get Admin API Key
    api_key = get_secret("ghost-admin-api-key")
    if not api_key:
        return False, "Server misconfiguration: Secret key unknown"
    
    try:
        # Extract KID from header for verification
        unverified_header = jwt.get_unverified_header(token)
        token_kid = unverified_header.get("kid")
        
        id_part, secret_part = api_key.split(':')
        
        if token_kid != id_part:
            return False, f"Token 'kid' mismatch. Expected {id_part}"
            
        # Hex-decode the secret as per Ghost spec
        secret_bytes = bytes.fromhex(secret_part)
        
        # Verify JWT (HS256)
        jwt.decode(token, secret_bytes, algorithms=["HS256"], audience="/v4/admin/")
        return True, None
    except jwt.ExpiredSignatureError:
        return False, "Token expired"
    except jwt.InvalidTokenError as e:
        return False, f"Invalid token: {str(e)}"
    except Exception as e:
        return False, f"Verification error: {str(e)}"

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
            'Access-Control-Allow-Headers': 'Content-Type, Authorization',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)
    
    headers = {'Access-Control-Allow-Origin': '*'}
    
    start_time = datetime.datetime.now()
    
    # --- SECURITY: JWT Verification ---
    auth_header = request.headers.get("Authorization")
    is_valid, error = verify_ghost_jwt(auth_header)
    
    auth_done = datetime.datetime.now()
    print(f"⏱️ AUTH CHECK: {(auth_done - start_time).total_seconds()}s")
    
    if not is_valid:
        print(f"🔒 UNAUTHORIZED: {error}")
        return jsonify({"error": "Unauthorized", "details": error}), 401, headers
    
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
        
        # Store in Firestore (Standardized Resilience)
        db = get_db()
        session_ref = db.collection('agent_sessions').document(session_id)
        
        # ADK FIX: Standardize on .set(merge=True) to ensure ID is recorded 
        # even if root document was recently deleted (Shadow Session).
        session_ref.set({
            "ghost_post_id": ghost_post_id,
            "ghost_post_created_at": datetime.datetime.now(datetime.timezone.utc),
            "last_updated": datetime.datetime.now(datetime.timezone.utc)
        }, merge=True)
        
        db_done = datetime.datetime.now()
        print(f"✅ Stored Ghost post ID: {ghost_post_id} for session: {session_id}")
        print(f"⏱️ DB WRITE: {(db_done - auth_done).total_seconds()}s")
        print(f"⏱️ TOTAL TIME: {(db_done - start_time).total_seconds()}s")
        
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
