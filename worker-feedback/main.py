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
from google.cloud import firestore, tasks_v2, secretmanager, logging as cloud_logging
import time
import random
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- UNIFIED MODEL ADAPTER (The Brain Switch) ---
class UnifiedModel:
    def __init__(self, provider, model_name):
        self.provider = provider
        self.model_name = model_name
        if provider == "vertex_ai":
            vertexai.init(project=PROJECT_ID, location=LOCATION)
            from vertexai.generative_models import GenerativeModel, SafetySetting
            # Standard safety settings
            safety = [
                SafetySetting(category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH),
                SafetySetting(category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH),
                SafetySetting(category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH),
                SafetySetting(category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH),
            ]
            self._native_model = GenerativeModel(model_name, safety_settings=safety)
            print(f"✅ Loaded Unified Vertex Model: {model_name}")

    def generate_content(self, prompt, generation_config=None, max_retries=3):
        import time
        import random
        
        if self.provider == "vertex_ai":
            retries = 0
            while retries <= max_retries:
                try:
                    return self._native_model.generate_content(prompt, generation_config=generation_config)
                except Exception as e:
                    error_msg = str(e).lower()
                    if "429" in error_msg or "resource exhausted" in error_msg:
                        if retries < max_retries:
                            wait_time = (2 ** retries) + random.uniform(0, 1)
                            print(f"⚠️ Feedback 429 (Quota): Retrying in {wait_time:.2f}s...")
                            time.sleep(wait_time)
                            retries += 1
                            continue
                    raise e
        elif self.provider == "anthropic":
            # LiteLLM failover style for Feedback Worker parity
            import litellm
            # Ensure litellm doesn't throw if billing is weird
            litellm.drop_params = True
            
            # Setup API Key (Prefer secret if available)
            api_key = get_secret("anthropic-api-key")
            if api_key:
                os.environ["ANTHROPIC_API_KEY"] = api_key
            
            response = litellm.completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=generation_config.get("temperature", 0.7) if generation_config else 0.7
            )
            return response.choices[0].message
        return None

# --- Config ---
PROJECT_ID = os.environ.get("PROJECT_ID")
LOCATION = os.environ.get("LOCATION")
MODEL_NAME = os.environ.get("MODEL_NAME")
N8N_PROPOSAL_WEBHOOK_URL = os.environ.get("N8N_PROPOSAL_WEBHOOK_URL") 

# --- HELPER: Safe n8n Webhook Delivery ---
def safe_n8n_delivery(payload, timeout=45):
    """
    Robust delivery to n8n with retries and SSL resilience.
    Uses 'requests' with an HTTPAdapter for exponential backoff on transient errors.
    """
    if not N8N_PROPOSAL_WEBHOOK_URL:
        print("⚠️ safe_n8n_delivery: No webhook URL configured.")
        return False

    session = requests.Session()
    # Retry on specific status codes and connection errors
    # backoff_factor 2 means waits 2, 4, 8 seconds
    retries = Retry(
        total=3, 
        backoff_factor=2, 
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["POST"]
    )
    session.mount('https://', HTTPAdapter(max_retries=retries))
    
    try:
        print(f"DEBUG: Attempting safe_n8n_delivery [Session: {payload.get('session_id')}]")
        response = session.post(
            N8N_PROPOSAL_WEBHOOK_URL, 
            json=payload, 
            verify=certifi.where(), 
            timeout=timeout
        )
        response.raise_for_status()
        print(f"✅ safe_n8n_delivery: Success (200 OK)")
        return True
    except requests.exceptions.SSLError as e:
        print(f"❌ safe_n8n_delivery: SSL Handshake Failed: {e}")
        # Final emergency attempt without strict session pooling
        try:
            time.sleep(2)
            requests.post(N8N_PROPOSAL_WEBHOOK_URL, json=payload, verify=certifi.where(), timeout=timeout)
            return True
        except Exception as inner_e:
            print(f"❌ safe_n8n_delivery: Emergency fallback also failed: {inner_e}")
            return False
    except Exception as e:
        print(f"⚠️ safe_n8n_delivery: Payload delivery failed: {e}")
        return False


STORY_WORKER_URL = os.environ.get("STORY_WORKER_URL") 
QUEUE_NAME = "story-worker-queue"
FUNCTION_IDENTITY_EMAIL = os.environ.get("FUNCTION_IDENTITY_EMAIL")
INGEST_KNOWLEDGE_URL = os.environ.get("INGEST_KNOWLEDGE_URL")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY") # Shared via secret manager if needed

unimodel = None
specialist_model = None
db = None
tasks_client = None
secret_client = None
logging_v_client = None

# --- STYLE & SANITIZATION PROTOCOL (Global Constant) ---
# This ensures 100% linguistic consistency across all generation paths
# to prevent "Context Pollution" in long-running threads.
STYLE_PROTOCOL = """
### STYLE & SANITIZATION PROTOCOL (ZERO-TOLERANCE):
- **MEAT-FIRST NARRATIVE**: BAN robotic framing like "Short answer:", "Bottom line:", or "The takeaway is:". Start with direct data.
- **HUMAN FINGERPRINT**: Vary sentence length. Mix punchy sentences (5-10 words) with fluid reflections (20-35 words).
- **EM-DASH RESTRAINT**: Limit em-dashes (—) to max ONE per paragraph. Use semicolons (;) or parentheses ( ).
- **NARRATIVE COLON BAN**: PROHIBIT colons in prose to connect claims to details. Use a period and a new sentence, or descriptive transitions. 
- **COLON PROTOCOL (LISTS ONLY)**: Colons are for vertical bulleted lists ONLY. 
- **MANDATORY CAPITALIZATION**: THE FIRST WORD AFTER ANY COLON MUST BE CAPITALIZED. Absolutely PROHIBIT "Label: lowercase" patterns.
- **LEXICAL VARIETY (ANTI-CLUMPING)**: MANDATE categorical rotation. PROHIBIT repeating the same key noun/verb in adjacent sentences.
- **ACTIVE VERB PRIORITY**: PRIORITIZE descriptive, context-aware actions. Use the provided Dynamic Palette as mentors.
- **ZERO-PASSIVE VOICE**: PRIORITIZE active voice to drive narrative momentum.
- **ANTI-WATERMARK**: BAN robotic buzzwords: 'delve', 'tapestry', 'landscape', 'unlock', 'embark', 'comprehensive', 'robust'. 
- **NO COLON CLUMPING**: Do not use "Label: Definition" structures. Use active, descriptive narrative flow.
- **TACTICAL TRANSITIONS**: BAN robotic connectors like "Furthermore" or "Moreover." Use the provided Dynamic Palette to maintain narrative "drag."
- **MERMAID MODULARITY**: For complex architectures, FAVOR multiple modular diagrams (e.g., Phase 1 vs Phase 2) over a single dense block. This ensures high-fidelity readability and prevents transport failures (Slack 3k URL limit).
"""

# --- HELPER: Dynamic Linguistic Palette ---
def get_stylistic_mentors(session_id=None):
    """
    Dynamically retrieves randomized stylistic mentors from the shared linguistic palette.
    """
    import hashlib
    import random
    
    paths_to_try = [
        os.path.join(os.path.dirname(__file__), "..", "shared", "linguistic_palette.json"),
        os.path.join(os.path.dirname(__file__), "shared", "linguistic_palette.json"),
        os.path.join(os.path.dirname(__file__), "linguistic_palette.json")
    ]
    
    palette_path = next((p for p in paths_to_try if os.path.exists(p)), None)
    
    try:
        if not palette_path: return ""
        with open(palette_path, 'r') as f:
            palette = json.load(f)
            
        if session_id:
            seed_val = int(hashlib.md5(session_id.encode()).hexdigest(), 16) % (10**8)
            rng = random.Random(seed_val)
        else:
            rng = random.Random()
            
        mentors = []
        categories = palette.get("transitions", {})
        key_categories = ["opposition_limitation_contradiction", "cause_condition_purpose", "examples_support_emphasis", "agreement_addition_similarity"]
        
        for cat in key_categories:
            words = categories.get(cat, [])
            if words:
                sample = rng.sample(words, min(3, len(words)))
                mentors.append(f"- {cat.replace('_', ' ').title()}: {', '.join(sample)}")
            
        verbs = palette.get("verbs", {}).get("active_mentors", [])
        if verbs:
            sample_verbs = rng.sample(verbs, min(6, len(verbs)))
            mentors.append(f"- Active Verb Mentors: {', '.join(sample_verbs)}")
            
        return "\n### DYNAMIC STYLE PALETTE (TURN MENTORS):\nREQUIRED: Use at least two (2) words from the mentors below to maintain linguistic texture.\n" + "\n".join(mentors)
    except Exception as e:
        print(f"⚠️ get_stylistic_mentors error: {e}")
        return ""

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

def get_secret(secret_id):
    """Unified secret retrieval with env fallback."""
    try:
        global secret_client
        if secret_client is None:
            secret_client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{PROJECT_ID}/secrets/{secret_id}/versions/latest"
        response = secret_client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8")
    except Exception as e:
        print(f"Secret {secret_id} failed: {e}. Falling back to ENV.")
        return os.environ.get(secret_id.upper().replace("-", "_"))

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

def safe_generate_content(model, prompt):
    """
    Robust wrapper for ALL LLM calls.
    Handles:
    1. Exceptions (Vertex/SDK Errors)
    2. Soft Refusals ("Error generating content" strings)
    3. Failover to Specialist Model (Claude)
    """
    global specialist_model
    try:
        response = model.generate_content(prompt)
        text_out = response.text.strip()
        
        # Detect soft refusal
        if not text_out or any(fail_str in text_out.lower() for fail_str in ["error generating content", "i cannot fulfill", "internal error"]):
            raise ValueError(f"Soft refusal detected: {text_out[:50]}...")
            
        return text_out
    except Exception as e:
        print(f"⚠️ Primary Model Failed: {e}. Attempting Specialist Failover...")
        log_safety_event("llm_failover", {"error": str(e), "prompt_snippet": prompt[:500]})
        
        if specialist_model is None:
            # Initialize Anthropic Specialist explicitly (User Config: 4.5)
            specialist_model = UnifiedModel(provider="anthropic", model_name="claude-sonnet-4-5")

        # ADK FIX: Add explicit delay for 429 mitigation (Parity with worker-story)
        print(f"🔄 SafeGen FAILOVER: Cooling down for 2s before Anthropic call...")
        time.sleep(2)
        
        # Retry with backup
        fallback_resp = specialist_model.generate_content(prompt, generation_config={"temperature": 0.4})
        return fallback_resp.content.strip() if hasattr(fallback_resp, 'content') else fallback_resp.text.strip()
    except Exception as e2:
        # Catch Anthropic 429 specifically
        if "rate_limit_error" in str(e2).lower():
            print("⚠️ Anthropic Rate Limit hit in Feedback. Ultimate fallback triggered.")
            return "Cognitive overload (429). Triage/Approval request received but requires higher quota."
            
        print(f"❌ SafeGen Feedback Fallback Failed: {e2}")
        return "The feedback system is currently maximizing its cognitive load. Please try again in 60 seconds."


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

def tell_then_and_now_story(interlinked_concepts, tool_confirmation=None, session_id=None):
    if not tool_confirmation or not tool_confirmation.get("confirmed"): raise PermissionError("Wait for approval.")
    style_mentors = get_stylistic_mentors(session_id)
    return safe_generate_content(unimodel, f"{STYLE_PROTOCOL}\n{style_mentors}\nTASK: Tell a 'Then and Now' story using these concepts: {interlinked_concepts}")

def refine_proposal(topic, current_proposal, critique, session_id=None):
    style_mentors = get_stylistic_mentors(session_id)
    prompt = f"""
    REWRITE the following content proposal blueprint based on the user's critique.
    CRITIQUE: {critique}
    
    CURRENT PROPOSAL: {json.dumps(current_proposal)}
    
    Return ONLY a valid JSON object matching the input structure.
    
    {STYLE_PROTOCOL}
    {style_mentors}
    """
    raw_text = safe_generate_content(unimodel, prompt)
    return extract_json(raw_text)

# --- THE STATEFUL AND HARDENED FEEDBACK WORKER ---
@functions_framework.http
def process_feedback_logic(request):
    global unimodel, db
    if unimodel is None: 
        unimodel = UnifiedModel("vertex_ai", MODEL_NAME)
        db = firestore.Client()

    req = request.get_json(silent=True)
    if isinstance(req, list): req = req[0] # Add safety for n8n list payloads
    print(f"DEBUG: Feedback Worker received payload keys: {list(req.keys()) if req else 'None'}")
    session_id = req.get('session_id')
    user_feedback = req.get('feedback', '')
    images = req.get('images', []) # New sensory input array
    print(f"DEBUG: Feedback Worker images count: {len(images)}")
    
    doc_ref = db.collection('agent_sessions').document(session_id)
    session_doc = doc_ref.get()
    if not session_doc.exists: return jsonify({"error": "Session not found"}), 404
    session_data = session_doc.to_dict()
    slack_context = session_data.get('slack_context', {})

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
        doc_ref.update({
            "status": "delegating_research",
            "last_updated": expire_time
            })
        # FIX: Persist the feedback into the events subcollection
        doc_ref.collection('events').add({
            "event_type": "user_feedback",
            "text": user_feedback,
            "images": images,
            "code_files": req.get('code_files', []), # FIX: Include code files
            "timestamp": datetime.datetime.now(datetime.timezone.utc)
        })
        
        # FIX: Inject precise Slack Timestamps to prevent Idempotency Clashing in Worker Story
        # Worker Story deduplicates based on slack_context['ts']. If we send the stale session ts, it skips.
        if req.get('slack_ts'):
            slack_context['ts'] = req.get('slack_ts')
        if req.get('slack_thread_ts'):
            slack_context['thread_ts'] = req.get('slack_thread_ts')
            
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
    all_raw_events = [doc.to_dict() for doc in recent_events_query.stream()]
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

    # ADK FIX: Explicitly DELEGATE pSEO/Ghost requests to the Story Worker engine.
    # The pSEO pipeline is NOT an approval flow; it's a closed-ended provisioning directive.
    if any(kw in user_feedback.lower() for kw in ["pseo", "ghost", "cms", "publish as"]):
        intent = "DELEGATE"
        print(f"pSEO/Ghost intent detected. Forcing DELEGATE to Story Worker.")
    elif is_quick_approval:
        intent = "APPROVE"
        print(f"Quick Approval detected. Forcing APPROVE intent.")
    else:
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
        intent = safe_generate_content(unimodel, feedback_triage_prompt).upper()
    
    print(f"Feedback Triage classified intent as: {intent}")
    
    # --- 3. STATE-AWARE ROUTING LOGIC ---
    if intent == "DELEGATE":
        # This now handles all factual questions, meta-questions, and simple chat.
        print("Intent requires new research or is conversational. Delegating to story worker...")
        doc_ref.update({
            "status": "delegating_research",
            "last_updated": expire_time
            })
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
        slack_context.update({
            "ts": req.get('slack_ts'),
            "thread_ts": req.get('slack_thread_ts')
        })
        payload.update({"topic": session_data.get('topic'), "feedback_text": user_feedback, "slack_context": slack_context, "code_files": req.get('code_files', [])})
        dispatch_task(payload, STORY_WORKER_URL)
        return jsonify({"msg": "Delegated to Research Worker"}), 200
        
    elif intent == "APPROVE":
        # Final safety check: We proceed if either quick-approval was hit OR LLM classified as APPROVE
        print(f"Executing APPROVE path for: {user_feedback}")

        ts = datetime.datetime.now(datetime.timezone.utc)
        user_event = {"event_type": "user_feedback", "text": user_feedback, "intent": "APPROVE", "timestamp": ts}

        # FIX: Fetch FULL history from subcollection to find the proposal
        full_history = [doc.to_dict() for doc in events_ref.order_by('timestamp').stream()]
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
                target_content = tell_then_and_now_story(event['payload'], tool_confirmation={"confirmed": True}, session_id=session_id)
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
        doc_ref.update({
            "status": "completed", 
            "final_story": target_content,
            "last_updated": expire_time
        })
        
        events_ref.add(user_event)
        events_ref.add({"event_type": "final_output", "content": target_content, "timestamp": datetime.datetime.now(datetime.timezone.utc)})
        
        safe_n8n_delivery({"session_id": session_id, "proposal": [{"link": target_content}], "thread_ts": slack_context.get('ts'), "channel_id": slack_context.get('channel'), "is_final_story": True, "is_initial_post": False })
        return jsonify({"msg": "Approved and Ingested"}), 200

    elif intent == "REFINE":
        # FIX: Fetch history from subcollection
        full_history = [doc.to_dict() for doc in events_ref.order_by('timestamp').stream()]
        
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
        events_ref.add({"event_type": "user_feedback", "text": user_feedback, "intent": "REFINE", "timestamp": ts})
        events_ref.add({"event_type": "agent_proposal", "proposal_data": new_prop, "timestamp": ts})
        events_ref.add({"event_type": "adk_request_confirmation", "approval_id": new_id, "payload": new_prop['interlinked_concepts'], "timestamp": ts})
        
        doc_ref.update({"last_updated": expire_time})
        
        safe_n8n_delivery({
            "session_id": session_id, 
            "approval_id": new_id, 
            "proposal": new_prop['interlinked_concepts'], 
            "thread_ts": slack_context.get('ts'), 
            "channel_id": slack_context.get('channel'),
            "is_final_story": False,
            "is_initial_post": False
        })

    return jsonify({"message": "Refinement processed."}), 200