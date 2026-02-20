# --- /worker-feedback/main.py ---
import functions_framework
from flask import jsonify
import vertexai
from vertexai.generative_models import GenerativeModel, HarmCategory, HarmBlockThreshold, FinishReason, SafetySetting
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

# --- Safety Configuration (ADK/RAI Compliant) ---
PROJECT_ID = os.environ.get("PROJECT_ID")
LOCATION = os.environ.get("LOCATION", "us-central1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-2.0-flash-exp")

safety_settings = [
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    ),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    ),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
    ),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    ),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
        threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    ),
]

# --- UNIFIED MODEL ADAPTER (The Brain Switch) ---
class UnifiedModel:
    def __init__(self, provider, model_name):
        self.provider = provider
        self.model_name = model_name
        if provider == "vertex_ai":
            vertexai.init(project=PROJECT_ID, location=LOCATION)
            self._native_model = GenerativeModel(model_name, safety_settings=safety_settings)
            print(f"✅ Loaded Unified Vertex Model: {model_name} (Safety Active)")

    def generate_content(self, prompt, generation_config=None, max_retries=3, system_instruction=None):
        import time
        import random
        
        if self.provider == "vertex_ai":
            retries = 0
            while retries <= max_retries:
                try:
                    # ADK FIX: Modular System Instruction Support
                    if system_instruction:
                        from vertexai.generative_models import GenerativeModel
                        # Re-init model with system instruction for this turn
                        # (Note: In production, we might want to cache models per instruction hash)
                        temp_model = GenerativeModel(
                            self.model_name, 
                            safety_settings=self._native_model._safety_settings if hasattr(self._native_model, '_safety_settings') else None,
                            system_instruction=system_instruction
                        )
                        return temp_model.generate_content(prompt, generation_config=generation_config)
                    
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
            
            messages = []
            if system_instruction:
                messages.append({"role": "system", "content": str(system_instruction)})
            messages.append({"role": "user", "content": prompt})
            
            response = litellm.completion(
                model=self.model_name,
                messages=messages,
                temperature=generation_config.get("temperature", 0.7) if generation_config else 0.7
            )
            
            content = response.choices[0].message.content
            
            class MockResponse:
                def __init__(self, c): self.text = c
            return MockResponse(content)
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

# --- STYLE & SANITIZATION PROTOCOL (Modularized) ---
PROTOCOL_GROUNDING_RAI = """
### CONTEXTUAL INTEGRITY (ZERO-INVENTION):
- **DATA FIDELITY**: You are FORBIDDEN from generating any specific statistic (%), project count (e.g., "120 projects"), or implementation claim that is not explicitly in the USER PROMPT, PROVIDED FILES, or VERIFIED SEARCH RESULTS. Use qualitative descriptions if data is absent.
- **LINK VERIFICATION**: Only generate URLs that have been explicitly provided or verified in the current turn's context. Never "guess" a logical URL path.
- **ARCHITECTURAL ANCHORING**: Claims must align with the provided context (e.g., architectural or contextual specs).
- **GROUNDING DATA SUPREMACY**: When GROUNDING DATA contains current info, you MUST prioritize that over your training data. Your training data has a knowledge cutoff; GROUNDING DATA is real-time.
- **EXPLICIT SOURCE ATTRIBUTION**: Cite source type (e.g., "According to recent search data...") for transparency.
"""

PROTOCOL_VISUAL_TABULAR = """
### VISUAL & LOGIC PROTOCOLS:
- **MERMAID MANDATE**: You are strictly PROHIBITED from generating ASCII-based diagrams (arrows, pipes). Use `mermaid` code blocks.
- **MERMAID MODULARITY**: FAVOR multiple modular diagrams over a single dense block.
- **TABLE COMPACTION**: PROHIBIT blank lines within Markdown tables. All rows MUST be contiguous.
"""

PROTOCOL_LITERARY_CORE = """
### LITERARY & NARRATIVE ARCHITECTURE:
- **MEAT-FIRST NARRATIVE**: BAN robotic framing. Start with direct data.
- **HUMAN FINGERPRINT**: Vary sentence length. Mix punchy (5-10 word) with fluid (20-35 word) sentences.
- **EM-DASH RESTRAINT**: Limit em-dashes to max ONE per paragraph. Use semicolons or periods.
- **NARRATIVE COLON BAN**: PROHIBIT colons in prose to connect claims to details.
- **COLON PROTOCOL**: Colons for vertical lists ONLY. THE FIRST WORD AFTER ANY COLON MUST BE CAPITALIZED.
- **LEXICAL VARIETY**: PROHIBIT repeating the same key noun/verb in adjacent sentences.
- **ACTIVE VERB PRIORITY**: Use descriptive, context-aware active verbs.
- **TONE REPLACEMENT**: Don't state quality; show it through technical precision.
"""

PROTOCOL_FORMAT_CMS = """
### TARGET: CMS_DRAFT (Ghost CMS)
- **HTML ONLY**: Use semantic HTML. PROHIBIT Markdown.
- **TABLES**: Use `<table>` tags.
- **HEADERS**: Use `<h2>` or `<h3>`. PROHIBIT `#`.
- **PARAGRAPHS**: Wrap all paragraphs in `<p>`.
"""

PROTOCOL_FORMAT_SLACK = """
### TARGET: MODERATOR_VIEW (Slack)
- **MARKDOWN ONLY**: Use Markdown exclusively. PROHIBIT HTML tags.
- **SPACING**: Use blank lines for paragraph separation.
"""

PROTOCOL_ANTI_SLOB = """
### ANTI-WATERMARK & NOISE REDUCTION:
- **BANNED BUZZWORDS**: 'delve', 'tapestry', 'landscape', 'unlock', 'embark', 'comprehensive', 'robust'.
- **NO COLON CLUMPING**: Do not use "Label: Definition" structures. Use active narrative flow.
- **STRATEGIC SANITIZATION**: PROHIBIT mentioning internal strategy (SEO metrics, turn-counts, internal benchmarks) in public drafts.
"""

# 6. PSEO_PAGE: Specialized Data Weaver (Non-Narrative)
PROTOCOL_PSEO_PAGE = """
- **ROLE: Specialized pSEO Data Weaver**: You create specific, data-rich pages for unique entity/location combinations.
- **CRITICAL "NO META-TALK" PROTOCOL**:
    1. **ABSOLUTELY NO "Guides"**: Do NOT write "Here is how you would write this page..." or "This page structure is designed to...".
    2. **DIRECT OUTPUT**: Start immediately with the Page Title (<h1> or #) or the first content section.
    3. **NO APOLOGIES**: If data is missing or masked, using the approved placeholder format without apologizing to the reader.
    4. **AUTHORITATIVE VOICE**: You are the definitive source. Do not use phrases like "based on the provided context."
- **DATA DENSITY**: Prioritize tables and lists over long paragraphs.
"""

# Backward Compatibility
STYLE_PROTOCOL = PROTOCOL_GROUNDING_RAI + PROTOCOL_VISUAL_TABULAR + PROTOCOL_LITERARY_CORE + PROTOCOL_ANTI_SLOB

def get_system_instructions(intent: str, output_target: str) -> str:
    """
    Architectural fix to assemble instructions modularly based on intent and target.
    Prevents token waste while ensuring 100% rule fidelity for relevant tasks.
    """
    intent = intent.upper().strip()
    instructions = "You are a highly capable AI assistant. Adhere strictly to these protocols:\n"
    instructions += PROTOCOL_GROUNDING_RAI
    instructions += PROTOCOL_VISUAL_TABULAR
    
    if output_target == "CMS_DRAFT":
        instructions += PROTOCOL_FORMAT_CMS
    else:
        instructions += PROTOCOL_FORMAT_SLACK
        
    # Condition: High-Fidelity intents get the full Literary and Anti-Slob treatment.
    high_fidelity_intents = ["DEEP_DIVE", "PSEO_ARTICLE", "PSEO_PAGE", "TECHNICAL_EXPLANATION", "REWRITE", "REFINE", "THEN_VS_NOW_PROPOSAL"]
    if intent in high_fidelity_intents:
        instructions += PROTOCOL_LITERARY_CORE
        
    if intent == "PSEO_PAGE":
        instructions += PROTOCOL_PSEO_PAGE
        
    if intent in ["PSEO_ARTICLE", "DEEP_DIVE", "PSEO_PAGE"]:
        instructions += PROTOCOL_ANTI_SLOB
        
    return instructions.strip()
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

# --- HELPER: Citation Engine (Inline Anchors) ---
def extract_labeled_sources(context_str):
    """
    Extracts URLs from context and returns a labeled list for the LLM.
    """
    if not context_str: return ""
    # Extract unique URLs
    found_urls = list(dict.fromkeys(re.findall(r'https?://[^\s<>"]+', str(context_str))))
    # Filter noisy dev/system URLs
    clean_urls = [u for u in found_urls if not any(x in u for x in ["ghost-api", "image", "google-adk", "localhost", "mcp-sensory-server", "favicon", "google.com/search"])]
    if not clean_urls: return ""
    
    labeled_list = "\n".join([f"SOURCE [{i+1}]: {url}" for i, url in enumerate(clean_urls[:8])])
    return f"\nADK_CITATION_GROUNDING_RESOURCES:\n{labeled_list}\n"

def convert_html_to_markdown(html_str):
    """
    Converts architectural HTML into Slack-friendly Markdown with a clear hierarchy.
    Hardened to protect code blocks from global tag stripping (fixes < > consumptions).
    """
    if not html_str: return ""
    import html
    import uuid

    # 0. Pre-Pass: Strip LLM Preamble/Banter
    text = re.sub(r'(?i)^(Part \d+:?|Here is.*?:\s*)', '', str(html_str)).strip()
    
    # 1. Strip raw Markdown headers
    text = re.sub(r'^\s*#{1,6}\s*(.*?)$', r'*\1*', text, flags=re.MULTILINE)
    
    # Placeholder system to protect code from final tag strip
    protected_blocks = {}

    def protect_block(md_content):
        placeholder = f"__PROTECTED_CODE_{uuid.uuid4().hex}__"
        protected_blocks[placeholder] = md_content
        return placeholder

    # 2. Mermaid Blocks (Specialist Visual Handling via MCP)
    def mermaid_markdown_handler(match):
        full_block = match.group(0)
        code_match = re.search(r'<code[^>]*class="[^"]*mermaid[^"]*"[^>]*>([\s\S]*?)</code>', full_block, re.IGNORECASE)
        if not code_match: return full_block
        mermaid_code = code_match.group(1).strip()
        
        caption = "Mermaid Diagram"
        caption_match = re.search(r'<figcaption>(.*?)</figcaption>', full_block, re.IGNORECASE)
        if caption_match:
            caption = re.sub(r'<[^>]+>', '', caption_match.group(1).strip())

        try:
            mcp = get_mcp_client()
            md = mcp.call_tool("render_mermaid", {
                "mermaid_code": mermaid_code, 
                "format": "markdown",
                "alt": caption,
                "caption": caption
            })
            return protect_block(md)
        except Exception:
            return protect_block(f"```mermaid\n{mermaid_code}\n```")

    fig_pattern = r'<figure[^>]*>(?:(?!</figure>)[\s\S])*?<code[^>]*class="[^"]*mermaid[^"]*"[^>]*>(?:(?!</figure>)[\s\S])*?</code>(?:(?!</figure>)[\s\S])*?</figure>'
    text = re.sub(fig_pattern, mermaid_markdown_handler, text, flags=re.IGNORECASE)

    # Orphan Mermaid Handler
    orphan_pattern = r'<pre\s*[^>]*><code\s+class="[^"]*mermaid[^"]*">([\s\S]*?)</code></pre>|<code\s+class="[^"]*mermaid[^"]*"[^>]*>([\s\S]*?)</code>'
    def orphan_handler(match):
        code = (match.group(1) or match.group(2) or "").strip()
        try:
            mcp = get_mcp_client()
            md = mcp.call_tool("render_mermaid", {"mermaid_code": code, "format": "markdown"})
            return protect_block(md)
        except Exception:
            return protect_block(f"```mermaid\n{code}\n```")
    text = re.sub(orphan_pattern, orphan_handler, text, flags=re.IGNORECASE)

    # 3. General Code Blocks
    def general_code_handler(match):
        attrs = match.group(1)
        code = match.group(2)
        # SANITATION: Unescape entities and strip internal HTML tags from code
        code = html.unescape(code)
        code = re.sub(r'<[^>]+>', '', code)
        
        lang_match = re.search(r'language-(\w+)', attrs, re.IGNORECASE)
        if lang_match or '\n' in code.strip():
            lang = lang_match.group(1) if lang_match else ""
            return protect_block(f"```{lang}\n{code.strip()}\n```")
        return protect_block(f"`{code.strip()}`")

    text = re.sub(r'<pre><code([^>]*)>([\s\S]*?)</code></pre>', general_code_handler, text, flags=re.IGNORECASE)
    text = re.sub(r'<code([^>]*)>([\s\S]*?)</code>', general_code_handler, text, flags=re.IGNORECASE)

    # 5. Tables (Simple to Pipe conversion for Slack)
    def table_handler(match):
        table_html = match.group(0)
        rows = re.findall(r'<tr[^>]*>(.*?)</tr>', table_html, re.DOTALL | re.IGNORECASE)
        if not rows: return ""
        md_table = []
        for i, row in enumerate(rows):
            cols = re.findall(r'<(?:td|th)[^>]*>(.*?)</(?:td|th)>', row, re.DOTALL | re.IGNORECASE)
            # Strip internal tags from columns
            cols = [re.sub(r'<[^>]+>', '', c).strip() for c in cols]
            if not cols: continue
            md_table.append("| " + " | ".join(cols) + " |")
            if i == 0 and len(rows) > 1: # Add separator after header
                md_table.append("| " + " | ".join(["---"] * len(cols)) + " |")
        return "\n" + "\n".join(md_table) + "\n"

    text = re.sub(r'<table[^>]*>(.*?)</table>', table_handler, text, flags=re.DOTALL | re.IGNORECASE)

    # 6. Hierarchical Elements (H1-H3, Sections)
    text = re.sub(r'<h1>(.*?)</h1>', r'*\1*\n\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<h2>(.*?)</h2>', r'\n*\1*\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<h3>(.*?)</h3>', r'_ \1 _\n', text, flags=re.IGNORECASE)
    text = text.replace('</section>', '\n\n---\n\n')
    text = re.sub(r'<section[^>]*>', '', text, flags=re.IGNORECASE)

    # 7. Lists
    def list_item_handler(match):
        content = match.group(1).strip()
        if re.match(r'^(\d+\.|•|-|\*)', content):
            return f"  {content}\n"
        return f"  • {content}\n"

    text = re.sub(r'<li>(.*?)</li>', list_item_handler, text, flags=re.DOTALL | re.IGNORECASE)
    text = text.replace('<ul>', '\n').replace('</ul>', '\n').replace('<ol>', '\n').replace('</ol>', '\n')
    
    # 6. Paragraphs & Line Breaks
    text = text.replace('<p>', '').replace('</p>', '\n\n')
    text = text.replace('<br>', '\n').replace('<br/>', '\n')
    
    # 7. Final Strip of remaining tags
    text = re.sub(r'<(?!https?://|!)[^>]+>', '', text, flags=re.IGNORECASE)
    
    # 8. Restore Protected Blocks
    for placeholder, original in protected_blocks.items():
        text = text.replace(placeholder, original)

    # 9. Clean up excess newlines
    text = re.sub(r'\n{4,}', '\n\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

# --- Utils ---
def extract_json(text):
    """
    Super-Listener: Hardened extraction of JSON from LLM responses.
    Handles markdown fences (```json) and raw braced strings/lists.
    """
    if not text: return None
    
    # 1. Look for Markdown-wrapped blocks first
    markdown_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if markdown_match:
        content = markdown_match.group(1).strip()
    else:
        # 2. Hardened Bracket Discovery (Objects or Lists)
        # Finds the outermost [...] or {...}
        match = re.search(r'([\[\{][\s\S]*[\]\}])', text)
        if not match: return None
        content = match.group(1).strip()
    
    try:
        return json.loads(content)
    except Exception as e:
        print(f"FAILED FEEDBACK JSON PARSE: {e}")
        raise ValueError(f"No valid JSON found in response.")

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

def safe_generate_content(model, prompt, system_instruction=None, generation_config=None):
    """
    Robust wrapper for ALL LLM calls.
    Handles:
    1. Exceptions (Vertex/SDK Errors)
    2. Soft Refusals ("Error generating content" strings)
    3. Failover to Specialist Model (Claude)
    """
    global specialist_model
    try:
        response = model.generate_content(prompt, system_instruction=system_instruction, generation_config=generation_config)
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
        fallback_resp = specialist_model.generate_content(prompt, system_instruction=system_instruction, generation_config=generation_config or {"temperature": 0.4})
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

def tell_then_and_now_story(interlinked_concepts, tool_confirmation=None, session_id=None, output_target="MODERATOR_VIEW"):
    global specialist_model
    if not tool_confirmation or not tool_confirmation.get("confirmed"): raise PermissionError("Wait for approval.")
    
    # UPGRADE: Force Specialist Model (Claude 4.5) for high-fidelity narrative
    if specialist_model is None:
        specialist_model = UnifiedModel(provider="anthropic", model_name="claude-sonnet-4-5")
        
    style_mentors = get_stylistic_mentors(session_id)
    
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

def get_output_target(intent: str) -> str:
    """
    Centralized mapping logic for target-aware formatting.
    """
    intent = intent.upper().strip()
    if intent == "PSEO_ARTICLE":
        return "CMS_DRAFT"
    return "MODERATOR_VIEW"

# --- THE STATEFUL AND HARDENED FEEDBACK WORKER ---
@functions_framework.http
def process_feedback_logic(request):
    global unimodel, db
    if unimodel is None: 
        unimodel = UnifiedModel("vertex_ai", MODEL_NAME)
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

    # 0. Global Safety Shield (Safety Pre-flight)
    if run_global_safety_check(user_feedback):
        print(f"🛑 GLOBAL SAFETY SHIELD: Block triggered for feedback: '{user_feedback[:50]}...'")
        refusal_text = "I'm sorry, I cannot process this feedback as it violates my safety guidelines. If you are in Nigeria and need support, please contact the Nigerian Mental Health helplines: https://www.nigerianmentalhealth.org/helplines"
        
        # Log to Firestore
        db.collection('agent_sessions').document(session_id).collection('events').add({
            "event_type": "safety_block",
            "text": refusal_text,
            "reason": "RAI_FILTER",
            "timestamp": datetime.datetime.now(datetime.timezone.utc)
        })
        
        # Send refusal back to Slack/n8n
        safe_n8n_delivery({
            "session_id": session_id,
            "type": "social",
            "message": refusal_text
        })
        return jsonify({"msg": "Safety refusal sent"}), 200
    
    doc_ref = db.collection('agent_sessions').document(session_id)
    session_doc = doc_ref.get()
    session_data = {}
    if not session_doc.exists:
        # PSEO RESILIENCE: Check if the 'events' subcollection has history (Shadow Session)
        events_ref = doc_ref.collection('events')
        # Get recent events to find metadata (Topic, Type, Slack Context)
        history_stream = events_ref.order_by('timestamp', direction=firestore.Query.DESCENDING).limit(10).stream()
        history_events = [d.to_dict() for d in history_stream]
        
        if history_events:
            print(f"⚠️ SHADOW SESSION DETECTED: {session_id}. Attempting auto-heal...")
            # Try to reconstruct from history
            recovered_topic = "Recovered Session"
            recovered_type = "general"
            recovered_slack = {}
            recovered_ghost_id = None
            
            for event in history_events:
                # Look for topic, type, or slack context in both event root and nested payloads
                event_topic = event.get('topic') or event.get('proposal_data', {}).get('topic')
                if event_topic and recovered_topic == "Recovered Session":
                    recovered_topic = event_topic
                    
                event_type = event.get('type') or event.get('proposal_type')
                if event_type and recovered_type == "general":
                    recovered_type = event_type
                    
                event_slack = event.get('slack_context')
                if event_slack and not recovered_slack:
                    recovered_slack = event_slack
                
                # ADK FIX: Recover Ghost Post ID if it exists in any proposal event
                event_ghost_id = event.get('ghost_post_id') or event.get('proposal_data', {}).get('ghost_post_id')
                if event_ghost_id and not recovered_ghost_id:
                    recovered_ghost_id = event_ghost_id
            
            session_data = {
                "status": "self_healing",
                "topic": recovered_topic,
                "type": recovered_type,
                "slack_context": recovered_slack,
                "ghost_post_id": recovered_ghost_id,
                "last_updated": datetime.datetime.now(datetime.timezone.utc)
            }
            # Commit the healed root document to prevent repetitive recovery overhead
            doc_ref.set(session_data, merge=True)
        else:
            return jsonify({"error": "Session not found"}), 404
    else:
        session_data = session_doc.to_dict()
    
    # FIX: Robustly merge Slack Context from request and (potentially healed) session data
    req_slack = req.get('slack_context', {})
    # Use request context as baseline for identity/channel, then augment from database for thread history
    slack_context = {**session_data.get('slack_context', {}), **req_slack}
    
    # Ensure turn-specific signals are correctly mapped (Fixes KeyError: 'channel')
    if req.get('slack_ts'): slack_context['ts'] = req.get('slack_ts')
    if req.get('slack_thread_ts'): slack_context['thread_ts'] = req.get('slack_thread_ts')
    if req.get('channel_id') and not slack_context.get('channel'): slack_context['channel'] = req.get('channel_id')
    
    # Fallback for channel if still missing (Fixes 500 error in worker-story)
    if not slack_context.get('channel'):
        slack_context['channel'] = req.get('channel') or "UNKNOWN_CHANNEL"

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
        doc_ref.set({
            "status": "delegating_research",
            "last_updated": expire_time
            }, merge=True)
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
        # ARCHITECTURAL FIX: Modular Instruction Assembly
        # Triage doesn't need the full STYLE_PROTOCOL, but benefits from baseline guardrails.
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
    
    # --- 3. STATE-AWARE ROUTING LOGIC ---
    if intent == "DELEGATE":
        # This now handles all factual questions, meta-questions, and simple chat.
        print("Intent requires new research or is conversational. Delegating to story worker...")
        doc_ref.set({
            "status": "delegating_research",
            "last_updated": expire_time
            }, merge=True)
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
        doc_ref.set({
            "status": "completed", 
            "final_story": target_content,
            "last_updated": expire_time
        }, merge=True)
        
        events_ref.add(user_event)
        events_ref.add({"event_type": "final_output", "content": target_content, "timestamp": datetime.datetime.now(datetime.timezone.utc)})
        
        safe_n8n_delivery({
            "session_id": session_id, 
            "proposal": [{"link": convert_html_to_markdown(target_content)}], 
            "thread_ts": slack_context.get('ts'), 
            "channel_id": slack_context.get('channel'), 
            "is_final_story": True, 
            "is_initial_post": False 
        })
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
        
        doc_ref.set({"last_updated": expire_time}, merge=True)
        
        safe_n8n_delivery({
            "session_id": session_id, 
            "approval_id": new_id, 
            "proposal": [convert_html_to_markdown(c) if isinstance(c, str) else c for c in new_prop['interlinked_concepts']], 
            "thread_ts": slack_context.get('ts'), 
            "channel_id": slack_context.get('channel'),
            "is_final_story": False,
            "is_initial_post": False
        })

    return jsonify({"message": "Refinement processed."}), 200