# --- /worker-feedback/main.py ---
import functions_framework
from flask import jsonify
from vertexai.generative_models import HarmCategory, HarmBlockThreshold, FinishReason, SafetySetting
import os
import json

from shared.utils import (
    get_secret, 
    _firestore_call_with_timeout, 
    safe_n8n_delivery, 
    UnifiedModel,
    get_n8n_operation_type,
    get_output_target,
    extract_json,
    detect_audience_context,
    safe_generate_content,
    convert_html_to_markdown,
    get_system_instructions,
    extract_labeled_sources,
    get_stylistic_mentors
)
import re
import uuid
import datetime
import requests
from google.cloud import firestore, tasks_v2, logging as cloud_logging
import google.auth.transport.requests as google_auth_requests
import google.oauth2.id_token

safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

# --- SECRET MANAGER ---
secret_client = None
def log_safety_event(event_name, data):
    """Standardized safety event logger with payload clamping."""
    global logging_v_client
    try:
        if logging_v_client is None:
            logging_v_client = cloud_logging.Client()
        logger = logging_v_client.logger("safety_audit_feedback")
        
        # CLAMPING: Stay under 256KB Cloud Logging limit
        safe_data = {}
        for k, v in data.items():
            if isinstance(v, str) and len(v) > 50000:
                safe_data[k] = f"{v[:50000]}... [TRUNCATED]"
            else:
                safe_data[k] = v

        logger.log_struct({
            "event": event_name,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            **safe_data
        }, severity="WARNING")
    except Exception as e:
        print(f"Safety Event Logged (Local): {event_name} - {data} | Error: {e}")


# --- Configuration ---
PROJECT_ID = os.environ.get("PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT")
if not PROJECT_ID:
    try:
        import google.auth
        _, project_id_auth = google.auth.default()
        PROJECT_ID = project_id_auth
    except Exception:
        PROJECT_ID = None

LOCATION = os.environ.get("LOCATION", "us-central1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-2.0-flash") 
MODEL_PROVIDER = os.environ.get("MODEL_PROVIDER", "vertex_ai")
FLASH_MODEL_NAME = os.environ.get("FLASH_MODEL_NAME", "gemini-2.0-flash-lite")
ANTHROPIC_API_KEY = None
def get_anthropic_api_key():
    global ANTHROPIC_API_KEY
    if not ANTHROPIC_API_KEY:
        ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY") or get_secret("anthropic-api-key")
    return ANTHROPIC_API_KEY

OPENAI_API_KEY = None
def get_openai_api_key():
    global OPENAI_API_KEY
    if not OPENAI_API_KEY:
        OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or get_secret("openai-api-key")
    return OPENAI_API_KEY
N8N_PROPOSAL_WEBHOOK_URL = os.environ.get("N8N_PROPOSAL_WEBHOOK_URL") 
INGEST_KNOWLEDGE_URL = os.environ.get("INGEST_KNOWLEDGE_URL")
STORY_WORKER_URL = os.environ.get("STORY_WORKER_URL") 
QUEUE_NAME = "story-worker-queue"
FUNCTION_IDENTITY_EMAIL = os.environ.get("FUNCTION_IDENTITY_EMAIL")



unimodel = None
specialist_model = None
db = None
tasks_client = None
secret_client = None
logging_v_client = None

# (detect_audience_context removed, now imported from shared.utils)



# --- STYLE & SANITIZATION PROTOCOL (Consolidated in shared.utils) ---
# (PROTOCOL_* constants and get_system_instructions removed, now imported from shared.utils)

# --- HELPER: Dynamic Linguistic Palette (Consolidated in shared.utils) ---
# (get_stylistic_mentors removed, now imported from shared.utils)

# --- HELPER: Citation Engine (Consolidated in shared.utils) ---
# (extract_labeled_sources removed, now imported from shared.utils)

# (convert_html_to_markdown removed, now imported from shared.utils)

# --- Utils ---
# (extract_json removed, now imported from shared.utils)

def extract_url_from_text(text):
    url_pattern = r'(https?://[^\s<>|]+)'
    match = re.search(url_pattern, text)
    if match: return match.group(1)
    return None

# --- Safety Utils ---
def run_global_safety_check(text):
    """
    Perform a pre-flight safety scan using the Vertex AI native model.
    Ensures feedback is checked before ingestion or routing.
    """
    global unimodel
    if not text or not isinstance(text, str): return False
    
    try:
        test_prompt = f"Analyze the following user feedback for potential safety policy violations (harassment, hate speech, or promotion of harm). Distinguish between harmful intent and the informational/social discussion of sensitive topics. Feedback: {text}"
        response = unimodel._native_model.generate_content(test_prompt, safety_settings=safety_settings)
        
        if not response.candidates or response.candidates[0].finish_reason == FinishReason.SAFETY:
            return True
        return False
    except Exception as e:
        print(f"⚠️ Feedback Safety Check Exception: {e}")
        if "safety" in str(e).lower(): return True
        return False


# (safe_generate_content removed, now imported from shared.utils)


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

def tell_then_and_now_story(interlinked_concepts, tool_confirmation=None, session_id=None, output_target="MODERATOR_VIEW"):
    global specialist_model
    if not tool_confirmation or not tool_confirmation.get("confirmed"): raise PermissionError("Wait for approval.")
    
    # UPGRADE: Force Specialist Model (Claude 4.5) for high-fidelity narrative
    if specialist_model is None:
        specialist_model = UnifiedModel(provider="anthropic", model_name="claude-sonnet-4-5")
        
    style_mentors = get_stylistic_mentors(session_id)
    audience_context = detect_audience_context(interlinked_concepts) # Check proposal context for tone
    
    # ARCHITECTURAL FIX: Modular Instruction Assembly
    sys_instruction = get_system_instructions("THEN_VS_NOW_PROPOSAL", output_target)
    sys_instruction += f"\n\n{style_mentors}"

    # --- STRATEGIC CONTEXT SANITIZATION ---
    if output_target == "CMS_DRAFT":
        internal_keywords = [
            "competitor gap", "audit scores", "ranking analysis", "aeo strategy", 
            "moat factor", "technical density score", "turn 1", "turn 2", "turn 3", 
            "turn 4", "internal blueprint", "vetted prompt", "technical vacuum"
        ]
        for kw in internal_keywords:
            interlinked_concepts = re.sub(rf"(?i){kw}.*?\n?", "[STRATEGIC_CONTEXT_OMITTED] ", str(interlinked_concepts))
    
    prompt = f"""
    TASK: Tell a 'Then and Now' story using these concepts: {interlinked_concepts}
    
    AUDIENCE: {audience_context}
    
    {extract_labeled_sources(interlinked_concepts)}
    
    ### CRITICAL CITATION RULE:
    - **Inline Anchored Links**: When referencing facts supported by the GROUNDING SOURCES, you MUST use semantic HTML anchored links: `<a href="URL">Anchor Text</a>`.
    - **No Link Dumps**: Do NOT append a "Sources" list at the end.
    """
    
    print(f"DEBUG: tell_then_and_now_story: Using Specialist Model for high-fidelity synthesis. [Target: {output_target}]")
    return safe_generate_content(specialist_model, prompt, system_instruction=sys_instruction)

def refine_proposal(topic, current_proposal, critique, session_id=None):
    global specialist_model
    
    # UPGRADE: Force Specialist Model (Claude 4.5) for high-fidelity refinement
    if specialist_model is None:
        specialist_model = UnifiedModel(provider="anthropic", model_name="claude-sonnet-4-5")
        
    style_mentors = get_stylistic_mentors(session_id)
    
    # ARCHITECTURAL FIX: Modular Instruction Assembly
    sys_instruction = get_system_instructions("REFINE", "MODERATOR_VIEW")
    sys_instruction += f"\n\n{style_mentors}"

    prompt = f"""
    REWRITE the following content proposal blueprint based on the user's critique.
    CRITIQUE: {critique}
    
    CURRENT PROPOSAL: {json.dumps(current_proposal)}
    
    Return ONLY a valid JSON object matching the input structure.
    """
    print(f"DEBUG: refine_proposal: Using Specialist Model for high-fidelity instruction adherence.")
    raw_text = safe_generate_content(specialist_model, prompt, system_instruction=sys_instruction)
    return extract_json(raw_text)

# (get_output_target removed, now imported from shared.utils)

# --- THE STATEFUL AND HARDENED FEEDBACK WORKER ---
@functions_framework.http
def process_feedback_logic(request):
    global unimodel, specialist_model, db
    if any(m is None for m in [unimodel, specialist_model]):
        if unimodel is None:
            unimodel = UnifiedModel("vertex_ai", MODEL_NAME)
        if specialist_model is None:
            specialist_model = UnifiedModel(provider="anthropic", model_name="claude-sonnet-4-5")
        db = firestore.Client()

    req = request.get_json(silent=True)
    if not req:
        print("🛑 Feedback Worker: Aborting. No payload provided.")
        return jsonify({"error": "Missing payload"}), 400
        
    if isinstance(req, list): req = req[0] # Add safety for n8n list payloads
    print(f"DEBUG: Feedback Worker received payload keys: {list(req.keys()) if req else 'None'}")
    session_id = req.get('session_id')
    user_feedback = req.get('feedback', '')
    images = req.get('images', []) # New sensory input array
    print(f"DEBUG: Feedback Worker images count: {len(images)}")

    # --- ANCHOR SESSION (Early Initialization) ---
    doc_ref = db.collection('agent_sessions').document(session_id)
    
    # NATIVE RESCUE: Try/Except wrap the synchronous read to catch TSI_DATA_CORRUPTED
    try:
        session_doc = _firestore_call_with_timeout(lambda: doc_ref.get())
    except Exception as e:
        print(f"⚠️ FIRESTORE READ ERROR ({e}). Attempting channel resuscitation...")
        # Forcing a fresh client instantiation specifically for this container runtime
        db = firestore.Client(project=PROJECT_ID)
        doc_ref = db.collection('agent_sessions').document(session_id)
        session_doc = _firestore_call_with_timeout(lambda: doc_ref.get())
        print("✅ Channel resuscitation successful!")
        
    session_data = {}
    if not session_doc.exists:
        return jsonify({"error": "Session not found"}), 404
    else:
        session_data = session_doc.to_dict()

    # NORMALIZE: Basic context mapping
    req_slack = req.get('slack_context', {}).copy()
    if not req_slack.get('channel'):
        req_slack['channel'] = req.get('slack_channel') or req.get('channel')
    if not req_slack.get('ts'):
        req_slack['ts'] = req.get('ts') or req.get('slack_ts')

    # LIGHT ARCHITECTURE: Context Inheritance (Dispatcher-Anchored)
    session_slack = session_data.get('slack_context', {})
    slack_context = session_slack.copy()
    for k, v in req_slack.items():
        if v is not None:
            slack_context[k] = v
            
    if not slack_context.get('channel'):
        slack_context['channel'] = req.get('slack_channel') or req.get('channel') or req.get('event', {}).get('channel')
    if not slack_context.get('ts'):
        slack_context['ts'] = req.get('slack_ts') or req.get('ts') or req.get('event', {}).get('ts')

    # 0. Global Safety Shield (Safety Pre-flight)
    if run_global_safety_check(user_feedback):
        print(f"🛑 GLOBAL SAFETY SHIELD: Block triggered for feedback: '{user_feedback[:50]}...'")
        refusal_text = "I'm sorry, I cannot process this feedback as it violates my safety guidelines. If you are in Nigeria and need support, please contact the Nigerian Mental Health helplines: https://www.nigerianmentalhealth.org/helplines"
        
        # Log to Firestore
        _firestore_call_with_timeout(lambda: db.collection('agent_sessions').document(session_id).collection('events').add({
            "event_type": "safety_block",
            "text": refusal_text,
            "reason": "RAI_FILTER",
            "timestamp": datetime.datetime.now(datetime.timezone.utc)
        }))
        
        # Determine global operation type for all N8N payloads
        ghost_post_id = session_data.get('ghost_post_id') if isinstance(session_data, dict) else None
        global_operation_type = get_n8n_operation_type("SAFETY_BLOCK", session_data.get('topic', ''), user_feedback, ghost_post_id)
        
        # Send refusal back to Slack/n8n
        target = get_output_target(global_operation_type)
        safe_n8n_delivery({
            "session_id": session_id,
            "type": global_operation_type,
            "message": refusal_text,
            "query": user_feedback,
            "output_target": target,
            "channel_id": slack_context.get('channel'),
            "thread_ts": slack_context.get('ts') or slack_context.get('thread_ts')
        })
        return jsonify({"msg": "Safety refusal sent"}), 200
    
    # --- SYSTEM FILTER: Prevent Infinite Loops & Redundant Ingestion ---
    # Case: The message is a confirmation notification (e.g., from N8N/Ghost)
    is_system_confirmation = "created in ghost" in user_feedback.lower() or "ready for ghost" in user_feedback.lower()
    if is_system_confirmation:
        print(f"🛑 SYSTEM FILTER: Ignoring Ghost/N8N status notification: {user_feedback[:50]}...")
        return jsonify({"msg": "Status notification ignored"}), 200

    # --- CALCULATE THE EXPIRATION TIMESTAMP ---
    expire_time = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=30)
    
    # --- ADK FIX 1: RESTORED DETERMINISTIC GUARDRAIL ---
    # Always check for a URL first. It's the fastest and most reliable signal.
    if extract_url_from_text(user_feedback):
        print("URL detected in feedback. Delegating to research worker immediately.")
        _firestore_call_with_timeout(lambda: doc_ref.update({
            "status": "delegating_research",
            "last_updated": expire_time
            }))
        # FIX: Persist the feedback into the events subcollection
        doc_ref.collection('events').add({
            "event_type": "user_feedback",
            "text": user_feedback,
            "images": images,
            "code_files": req.get('code_files', []), # FIX: Include code files
            "timestamp": datetime.datetime.now(datetime.timezone.utc)
        })
        
        dispatch_task({
            "session_id": session_id, 
            "topic": session_data.get('topic'), 
            "feedback_text": user_feedback, 
            "slack_context": slack_context, 
            "images": images, 
            "code_files": req.get('code_files', [])
        }, STORY_WORKER_URL)
        return jsonify({"msg": "Delegated (URL detected)"}), 200

    # --- ADK FIX 2: CONTEXTUAL LLM TRIAGE ---
    # FIX: Read from 'events' subcollection instead of 'event_log' array
    events_ref = doc_ref.collection('events')
    
    # MEMORY EXPANSION: Removed .limit(5) to ensure context isn't lost during feedback loops
    recent_events_query = events_ref.order_by('timestamp', direction=firestore.Query.DESCENDING)


    # 2. Stream and then REVERSE the list to get chronological order (Oldest -> Newest)
    all_raw_events = _firestore_call_with_timeout(lambda: [doc.to_dict() for doc in recent_events_query.stream()], timeout_secs=20)
    # CLAMPING: Only take the last 10 events (5 turns) for feedback triage
    recent_events = all_raw_events[:10][::-1] if len(all_raw_events) > 10 else all_raw_events[::-1]
    
    # MEMORY EXPANSION: Removed char limits (was 500) to ensure full context for triage
    formatted_history = []
    for e in recent_events:
        etype = e.get('event_type', 'unknown')
        # Try all possible content fields
        content = e.get('text') or e.get('content') or e.get('payload') or str(e.get('data', ''))
        formatted_history.append(f"Turn ({etype}): {str(content)}")
    
    history_text = "\n".join(formatted_history)

    # --- ADK FIX 3: ROBUST APPROVAL DETECTION ---
    # Fast-path for single-word approvals
    tokens = re.sub(r'[^\w\s]', '', user_feedback).strip().lower().split()
    is_quick_approval = len(tokens) <= 3 and any(t in ["approved", "approve"] for t in tokens)

    # ADK FIX: SCHEMA fast-exit override (highest specificity — checked first)
    if any(kw in user_feedback.lower() for kw in ["inject schema", "fix schema", "remediate schema", "add json-ld", "push schema"]):
        intent = "SCHEMA_INJECT"
        print(f"Schema Inject intent detected in feedback. Routing to worker-schema.")
    elif any(kw in user_feedback.lower() for kw in ["validate schema", "check schema", "audit schema", "eds score", "schema score", "json-ld"]):
        intent = "SCHEMA_VALIDATE"
        print(f"Schema Validate intent detected in feedback. Routing to worker-schema.")
    # ADK FIX: Explicitly DELEGATE pSEO/Ghost requests to the Story Worker engine.
    # The pSEO pipeline is NOT an approval flow; it's a closed-ended provisioning directive.
    elif any(kw in user_feedback.lower() for kw in ["pseo", "ghost", "cms", "publish as", "lp"]):
        intent = "DELEGATE"
        print(f"pSEO/Ghost intent detected. Forcing DELEGATE to Story Worker.")
    elif is_quick_approval:
        intent = "APPROVE"
        print(f"Quick Approval detected. Forcing APPROVE intent.")
    else:
        # ARCHITECTURAL FIX: Modular Instruction Assembly
        # Triage uses baseline guardrails only — no high-fidelity persona needed.
        triage_sys = get_system_instructions("SIMPLE_QUESTION", "MODERATOR_VIEW")

        feedback_triage_prompt = f"""
        Analyze the user's latest message in the context of the conversation history. Classify the user's INTENT into one of three categories:

        1.  **APPROVE**: The user is explicitly confirming, finalizing, or "approving" the current result for the record or knowledge base.
            *   *Examples:* "I approve on the ideas in this conversation.", "Approved!", "This is perfect, save it.", "I like this, we are done."

        2.  **REFINE**: The user is asking for a change to the agent's current strategy or last structured proposal (e.g., a 'Then vs Now' draft).
            *   *Examples:* "Make it shorter", "Can you change the tone?", "Add a point about X"

        3.  **DELEGATE**: The user is asking a new factual question, confirming a proposed research direction, or asking a "meta" question. 
            *   **CRITICAL:** If the user is saying "Yes", "Proceed", "Looks good", or confirming a suggestion for *further research*, select **DELEGATE**.

        CONVERSATION HISTORY:
        {history_text}

        USER'S LATEST MESSAGE: "{user_feedback}"

        Respond with ONLY the category name (APPROVE, REFINE, or DELEGATE).
        """
        intent = safe_generate_content(unimodel, feedback_triage_prompt, system_instruction=triage_sys).upper()
    
    print(f"Feedback Triage classified intent as: {intent}")
    
    # Update global operation type based on triage result
    ghost_post_id = session_data.get('ghost_post_id')
    global_operation_type = get_n8n_operation_type(intent, session_data.get('topic', ''), user_feedback, ghost_post_id)
    
    # --- 3. STATE-AWARE ROUTING LOGIC ---

    # SCHEMA COUPLED-TASK DETECTION: Use Flash LLM to determine if schema request
    # needs additional research/enrichment beyond pure validation/injection.
    # Falls back to keywords if LLM call fails.
    has_coupled_task = False
    if intent in ["SCHEMA_VALIDATE", "SCHEMA_INJECT"]:
        _SCHEMA_FALLBACK_SIGNALS = ["research", "enrich", "generate description", "paa", "summarize"]
        try:
            coupling_prompt = (
                "You are a schema intent classifier. The user sent this message:\n"
                f'"{user_feedback}"\n\n'
                f"The detected intent is: {intent}\n\n"
                "Does this request require ANYTHING beyond pure schema validation or injection?\n"
                "This includes: researching content for schema fields, generating descriptions, "
                "PAA/FAQ enrichment, or other additional tasks.\n\n"
                "Answer with ONLY one word: YES or NO."
            )
            coupling_answer = str(safe_generate_content(unimodel, coupling_prompt, generation_config={"temperature": 0.0})).strip().upper()
            has_coupled_task = coupling_answer.startswith("YES")
            print(f"TELEMETRY: Schema coupling LLM check → [{coupling_answer}] for feedback: '{user_feedback[:40]}'")
        except Exception as _e:
            print(f"⚠️ Schema coupling LLM failed, falling back to keywords: {_e}")
            has_coupled_task = any(sig in user_feedback.lower() for sig in _SCHEMA_FALLBACK_SIGNALS)

    is_pure_schema = intent in ["SCHEMA_VALIDATE", "SCHEMA_INJECT"] and not images and not has_coupled_task


    if is_pure_schema:
        print(f"Executing Schema Fast-Exit from feedback loop: [{intent}]")
        schema_worker_url = os.environ.get("WORKER_SCHEMA_URL")
        target_url = req.get("site_url") or req.get("url")
        schema_ghost_post_id = req.get("ghost_post_id") or session_data.get("ghost_post_id")
        schema_mode = "inject" if intent == "SCHEMA_INJECT" else "validate"

        schema_result = None
        schema_reply = "Schema audit complete."
        try:
            auth_req = google_auth_requests.Request()
            id_token = google.oauth2.id_token.fetch_id_token(auth_req, schema_worker_url)
            schema_resp = requests.post(
                schema_worker_url,
                headers={"Authorization": f"Bearer {id_token}", "Content-Type": "application/json"},
                json={"site_url": target_url, "session_id": session_id, "mode": schema_mode, "ghost_post_id": schema_ghost_post_id},
                timeout=30
            )
            schema_resp.raise_for_status()
            schema_result = schema_resp.json().get("audit", {})
            eds = schema_result.get("eds_score", "N/A")
            findings = schema_result.get("findings", [])
            remediations = schema_result.get("remediations", [])
            priority = schema_result.get("priority", "NORMAL")
            injection = schema_result.get("injection_status", "N/A")
            schema_reply = (
                f"*🔍 Schema Audit: {target_url}*\n"
                f"*EDS Score:* `{eds}/100` | *Priority:* `{priority}`\n"
                f"*Findings:*\n" + "\n".join(f"  • {f}" for f in findings[:5]) +
                (f"\n*Remediations:* {len(remediations)} items flagged." if remediations else "") +
                (f"\n*Injection:* `{injection}`" if schema_mode == "inject" else "")
            )
        except Exception as e:
            print(f"⚠️ Schema Fast-Exit Error (feedback): {e}")
            schema_reply = f"Schema audit error: {str(e)[:100]}"

        _firestore_call_with_timeout(lambda: doc_ref.collection('events').add({
            "event_type": "schema_audit",
            "text": schema_reply,
            "audit": schema_result,
            "timestamp": datetime.datetime.now(datetime.timezone.utc)
        }))
        _firestore_call_with_timeout(lambda: doc_ref.update({"status": "completed", "last_updated": expire_time}))
        safe_n8n_delivery({
            "session_id": session_id,
            "type": global_operation_type,
            "message": schema_reply,
            "query": user_feedback,
            "output_target": get_output_target(intent),
            "channel_id": slack_context.get('channel'),
            "thread_ts": slack_context.get('ts'),
            "is_initial_post": False
        })
        return jsonify({"msg": "Schema audit complete", "eds_score": schema_result.get("eds_score") if schema_result else None}), 200

    elif intent == "DELEGATE":
        # This now handles all factual questions, meta-questions, and simple chat.
        print("Intent requires new research or is conversational. Delegating to story worker...")
        _firestore_call_with_timeout(lambda: doc_ref.update({
            "status": "delegating_research",
            "last_updated": expire_time
            }))
        # FIX: Persist the feedback into the events subcollection
        doc_ref.collection('events').add({
            "event_type": "user_feedback",
            "text": user_feedback,
            "images": req.get('images', []),
            "code_files": req.get('code_files', []), # FIX: Include code files
            "timestamp": datetime.datetime.now(datetime.timezone.utc)
        })
        # No-Strip: Pass full payload forward, but preserve mission topic
        payload = req.copy()
        # FIX: Ensure we pass the CURRENT turn's context, not the stale session context
        if req.get('slack_ts'): slack_context['ts'] = req.get('slack_ts')
        if req.get('slack_thread_ts'): slack_context['thread_ts'] = req.get('slack_thread_ts')
        payload.update({"topic": session_data.get('topic'), "feedback_text": user_feedback, "slack_context": slack_context, "code_files": req.get('code_files', [])})
        dispatch_task(payload, STORY_WORKER_URL)
        return jsonify({"msg": "Delegated to Research Worker"}), 200
        
    elif intent == "APPROVE":
        # Final safety check: We proceed if either quick-approval was hit OR LLM classified as APPROVE
        print(f"Executing APPROVE path for: {user_feedback}")

        ts = datetime.datetime.now(datetime.timezone.utc)
        user_event = {"event_type": "user_feedback", "text": user_feedback, "intent": "APPROVE", "timestamp": ts}

        # FIX: Fetch FULL history from subcollection to find the proposal
        full_history = _firestore_call_with_timeout(lambda: [doc.to_dict() for doc in events_ref.order_by('timestamp').stream()], timeout_secs=20)
        # --- ROBUST ARTIFACT EXTRACTION (Avoids Old "Then-and-Now" Rants) ---
        target_content = None
        
        # We go backwards. The FIRST high-fidelity thing we hit is our target.
        for event in reversed(full_history):
            etype = event.get('event_type')
            
            # CATEGORY A: Direct Artifacts (Text already exists)
            # This includes blog posts, topic clusters, and direct answers
            if etype in ['agent_answer', 'agent_proposal', 'loop_draft']:
                # Prefer text, then data/payload
                raw_data = event.get('proposal_data') or event.get('payload') or event.get('data')
                target_content = event.get('text') or event.get('content')
                
                if not target_content and raw_data:
                    target_content = json.dumps(raw_data, indent=2) if isinstance(raw_data, (dict, list)) else str(raw_data)
                
                if target_content:
                    print(f"Found Recent Artifact: {etype}")
                    break

            # CATEGORY B: Deferred Synthesis (Requires calling a tool)
            # Only synthesize if this was the most recent professional event
            elif etype == 'adk_request_confirmation' and event.get('payload'):
                print("Found Recent Request for Synthesis (Then-and-Now). Synthesizing...")
                # Determine target based on session intent
                target = get_output_target(session_data.get('intent', 'THEN_VS_NOW_PROPOSAL'))
                target_content = tell_then_and_now_story(event['payload'], tool_confirmation={"confirmed": True}, session_id=session_id, output_target=target)
                break

        # Pass 2: Social Fallback
        if not target_content:
            for event in reversed(full_history):
                if event.get('event_type') == 'agent_reply':
                    target_content = event.get('text') or event.get('content')
                    if target_content: break
        
        if not target_content:
            return jsonify({"msg": "Nothing to approve found."}), 404

        # --- RAG INGESTION (Choice-Based Promotion) ---
        if INGEST_KNOWLEDGE_URL:
            # Check if we were approving a solution brief (Structured)
            is_solution_brief = False
            objections_data = None
            
            # Find the event we categorized earlier
            for event in reversed(full_history):
                if event.get('event_type') == 'agent_proposal' and event.get('proposal_type') == 'solution_brief':
                    is_solution_brief = True
                    objections_data = event.get('proposal_data', {}).get('objections')
                    break
            
            if is_solution_brief:
                print("Promoting Solution Brief to permanent storage...")
                payload = {
                    "session_id": session_id,
                    "topic": session_data.get('topic'),
                    "type": "solution_brief",
                    "story": target_content, # The brief HTML/MD
                    "objections": objections_data
                }
                dispatch_task(payload, INGEST_KNOWLEDGE_URL)
            else:
                # Standard Knowledge Base Path
                print("Promoting General Knowledge to vector base...")
                rag_content = f"MISSION: {session_data.get('topic', 'General Inquiry')}\n\nCONTENT:\n{target_content}"
                dispatch_task({"session_id": session_id, "topic": session_data.get('topic'), "story": rag_content, "type": "knowledge"}, INGEST_KNOWLEDGE_URL)
        
        # FIX: Update parent status, but write events to subcollection
        _firestore_call_with_timeout(lambda: doc_ref.update({
            "status": "completed", 
            "final_story": target_content,
            "last_updated": expire_time
        }))
        
        _firestore_call_with_timeout(lambda: events_ref.add(user_event))
        _firestore_call_with_timeout(lambda: events_ref.add({"event_type": "final_output", "content": target_content, "timestamp": datetime.datetime.now(datetime.timezone.utc)}))
        
        target = get_output_target(global_operation_type)
        safe_n8n_delivery({
            "session_id": session_id, 
            "type": global_operation_type,
            "proposal": [{"link": convert_html_to_markdown(target_content)}], 
            "query": user_feedback,
            "output_target": target,
            "thread_ts": slack_context.get('ts') or slack_context.get('thread_ts'), 
            "channel_id": slack_context.get('channel') or req.get('channel') or req.get('slack_channel') or req.get('event', {}).get('channel'), 
            "is_final_story": True, 
            "is_initial_post": False 
        })
        return jsonify({"msg": "Approved and Ingested"}), 200

    elif intent == "REFINE":
        # FIX: Fetch history from subcollection
        full_history = _firestore_call_with_timeout(lambda: [doc.to_dict() for doc in events_ref.order_by('timestamp').stream()], timeout_secs=20)
        
        # --- ROBUST ARTIFACT EXTRACTION (Same as APPROVE path) ---
        last_prop_data = None
        for event in reversed(full_history):
            etype = event.get('event_type')
            # Look for structured data in common keys
            data = event.get('proposal_data') or event.get('payload') or event.get('data')
            if data and etype in ['agent_proposal', 'loop_draft', 'adk_request_confirmation', 'agent_answer']:
                last_prop_data = data
                print(f"Refine: Found target artifact in '{etype}' event.")
                break
        
        if not last_prop_data:
            print("Refine intent found, but no usable technical artifact exists in history. Delegating...")
            # No-Strip: Pass full payload forward
            payload = req.copy()
            payload.update({"topic": session_data.get('topic'), "feedback_text": user_feedback})
            dispatch_task(payload, STORY_WORKER_URL)
            return jsonify({"msg": "Delegated (Refine Fallback)"}), 200

        # Optimization: If it's a dict/list (JSON Proposal), we refine it locally.
        # If it's raw text/HTML (Article), we DELEGATE to Story Worker for "Repurpose Mode".
        if isinstance(last_prop_data, str) or (isinstance(last_prop_data, dict) and 'interlinked_concepts' not in last_prop_data):
            print("Refine target is Text/HTML Article. Delegating to Story Worker for Repurposing...")
            payload = req.copy()
            payload.update({"topic": session_data.get('topic'), "feedback_text": user_feedback, "slack_context": slack_context, "code_files": req.get('code_files', [])})
            dispatch_task(payload, STORY_WORKER_URL)
            return jsonify({"msg": "Delegated (Article Refinement)"}), 200

        # JSON Path (Then-vs-Now)
        new_prop = refine_proposal(session_data.get('topic'), last_prop_data, user_feedback, session_id=session_id)
        new_id = f"approval_{uuid.uuid4().hex[:8]}"
        
        # FIX: Write to subcollection
        ts = datetime.datetime.now(datetime.timezone.utc)
        _firestore_call_with_timeout(lambda: events_ref.add({"event_type": "user_feedback", "text": user_feedback, "intent": "REFINE", "timestamp": ts}))
        _firestore_call_with_timeout(lambda: events_ref.add({"event_type": "agent_proposal", "proposal_data": new_prop, "timestamp": ts}))
        _firestore_call_with_timeout(lambda: events_ref.add({"event_type": "adk_request_confirmation", "approval_id": new_id, "payload": new_prop['interlinked_concepts'], "timestamp": ts}))
        
        _firestore_call_with_timeout(lambda: doc_ref.update({"last_updated": expire_time}))
        
        target = get_output_target(global_operation_type)
        safe_n8n_delivery({
            "session_id": session_id, 
            "type": global_operation_type,
            "approval_id": new_id, 
            "proposal": [convert_html_to_markdown(c) if isinstance(c, str) else c for c in new_prop['interlinked_concepts']], 
            "query": user_feedback,
            "output_target": target,
            "thread_ts": slack_context.get('ts') or slack_context.get('thread_ts'), 
            "channel_id": slack_context.get('channel') or req.get('channel') or req.get('slack_channel') or req.get('event', {}).get('channel'),
            "is_final_story": False,
            "is_initial_post": False
        })

    return jsonify({"message": "Refinement processed."}), 200