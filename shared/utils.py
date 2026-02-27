import os
import random
import hashlib
import certifi
import requests
import concurrent.futures
import re
import json
import uuid
import datetime
import html
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from google.cloud import secretmanager
import google.auth
import google.auth.transport.requests
from google.oauth2 import id_token
import litellm
from litellm import completion
import vertexai
from vertexai.generative_models import GenerativeModel, HarmCategory, HarmBlockThreshold

litellm.drop_params = True

# --- GLOBAL CONFIG & SAFETY ---
PROJECT_ID = os.environ.get("PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT")
if not PROJECT_ID:
    try:
        _, project_id_auth = google.auth.default()
        PROJECT_ID = project_id_auth
    except:
        PROJECT_ID = None

LOCATION = os.environ.get("LOCATION", "us-central1")
N8N_PROPOSAL_WEBHOOK_URL = os.environ.get("N8N_PROPOSAL_WEBHOOK_URL") 
MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://localhost:8080")
mcp_client = None

# Keys will be fetched dynamically below to avoid gRPC deadlocks

safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

# --- N8N Payload Helpers ---
def get_n8n_operation_type(intent, original_topic="", sanitized_topic="", ghost_post_id=None):
    """
    Standardized mapper to ensure high-fidelity N8N operation types.
    Prevents hardcoding strings like 'social' when a specific triage intent is present.
    """
    if not intent: return "social"
    intent_str = str(intent).upper()
    
    # PSEO Logic: Differentiate between Create and Update for Posts vs Pages
    if intent_str in ["PSEO_ARTICLE", "PSEO_PAGE", "PSEO_LP"]:
        if intent_str == "PSEO_LP":
            return "pseo_lp_update" if ghost_post_id else "pseo_lp_create"
            
        is_page_target = (intent_str == "PSEO_PAGE") or any(kw in original_topic.lower() or kw in sanitized_topic.lower() for kw in ["pseo page", "collection page", "page template", "ghost page", "page slug"])
        if is_page_target: 
            return "pseo_page_update" if ghost_post_id else "pseo_page_create"
        else: 
            return "pseo_update" if ghost_post_id else "pseo_draft"
            
    # Standard mapping for other intents
    return intent_str.lower()

def get_output_target(intent: str) -> str:
    """
    Centralized mapping logic for target-aware formatting.
    """
    if not intent: return "MODERATOR_VIEW"
    intent_str = str(intent).upper().strip()
    if intent_str in ["PSEO_ARTICLE", "PSEO_PAGE", "PSEO_LP"]:
        return "CMS_DRAFT"
    return "MODERATOR_VIEW"

secret_client = None
def get_secret(secret_id):
    """Retrieves a secret from Cloud Secret Manager with env var fallback."""
    global secret_client
    
    env_key = secret_id.upper().replace("-", "_")
    env_val = os.environ.get(env_key)
    if env_val:
        return env_val.strip()
        
    try:
        if secret_client is None:
            secret_client = secretmanager.SecretManagerServiceClient()
        
        name = f"projects/{PROJECT_ID}/secrets/{secret_id}/versions/latest"
        response = secret_client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8").strip()
    except Exception as e:
        print(f"⚠️ Secret Manager error for {secret_id}: {e}")
        return None

# --- gRPC CIRCUIT BREAKER ---
def _firestore_call_with_timeout(callable_fn, timeout_secs=20):
    """
    Executes a callable in a separate thread.
    If it hangs longer than standard timeout, it immediately abandons the thread 
    and raises TimeoutError, bypassing TSI_DATA_CORRUPTED lockups natively.
    """
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(callable_fn)
    try:
        return future.result(timeout=timeout_secs)
    except concurrent.futures.TimeoutError as exc:
        print(f"⚠️ gRPC CRITICAL TIMEOUT: Operation hung beyond {timeout_secs}s.")
        # Abandon the thread. NEVER use 'with' or wait=True, or it will deadlock the instance.
        executor.shutdown(wait=False, cancel_futures=True)
        raise TimeoutError(f"gRPC Thread Pool Timeout after {timeout_secs}s") from exc
    finally:
        # On success, clean up immediately without blocking
        executor.shutdown(wait=False)

# --- RESILIENT HTTP DELIVERY ---
def safe_n8n_delivery(payload, timeout=45):
    """Robust pure HTTP delivery to bypass gRPC failures."""
    webhook_url = N8N_PROPOSAL_WEBHOOK_URL
    if not webhook_url:
        print("⚠️ safe_n8n_delivery: No webhook URL configured.")
        return False

    session = requests.Session()
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
            webhook_url, 
            json=payload, 
            verify=certifi.where(), 
            timeout=timeout
        )
        response.raise_for_status()
        print(f"✅ safe_n8n_delivery: Success (200 OK)")
        return True
    except requests.exceptions.SSLError as e:
        print(f"❌ safe_n8n_delivery: SSL Handshake Failed: {e}")
        import time
        try:
            time.sleep(2)
            requests.post(webhook_url, json=payload, verify=certifi.where(), timeout=timeout)
            return True
        except Exception as inner_e:
            print(f"❌ safe_n8n_delivery: Emergency fallback also failed: {inner_e}")
            return False
    except Exception as e:
        print(f"⚠️ safe_n8n_delivery: Payload delivery failed: {e}")
        return False

# --- UNIFIED MODEL ADAPTER (The Brain Switch) ---
class UnifiedModel:
    """Routes requests to Vertex AI or fallback LLM providers and mitigates gRPC hangs."""
    def __init__(self, provider, model_name):
        self.provider = provider
        self.model_name = model_name
        
        if provider == "vertex_ai":
            # Per-request gRPC initialization
            if PROJECT_ID:
                vertexai.init(project=PROJECT_ID, location=LOCATION)
            else:
                vertexai.init()
            self._native_model = GenerativeModel(model_name, safety_settings=safety_settings)
            print(f"✅ Executed clean vertexai.init for {model_name}.")

    def generate_content(self, prompt, generation_config=None, max_retries=3, system_instruction=None):
        import time
        import random
        
        if self.provider == "vertex_ai":
            retries = 0
            while retries <= max_retries:
                try:
                    if generation_config is None: generation_config = {}
                    if "max_output_tokens" not in generation_config: generation_config["max_output_tokens"] = 8192
                    
                    model = GenerativeModel(
                        self.model_name, 
                        safety_settings=safety_settings,
                        system_instruction=system_instruction
                    )
                    
                    def _call_vertex():
                        return model.generate_content(prompt, generation_config=generation_config)
                    
                    # Wrap the API call in our thread circuit-breaker to prevent hanging
                    response = _firestore_call_with_timeout(_call_vertex, timeout_secs=45)
                    
                    if not response.candidates or response.candidates[0].finish_reason == 3: 
                         raise ValueError("Safety Block via FinishReason")
                    
                    return response
                except Exception as e:
                    error_msg = str(e).lower()
                    if "429" in error_msg or "resource exhausted" in error_msg:
                        if retries < max_retries:
                            wait_time = (2 ** retries) + random.uniform(0, 1)
                            print(f"⚠️ Vertex 429: Retrying in {wait_time:.2f}s... (Attempt {retries+1}/{max_retries})")
                            time.sleep(wait_time)
                            retries += 1
                            continue
                            
                    print(f"⚠️ Vertex AI Safety/gRPC Error: {e}. Attempting Specialist Failover to Claude 4.5...")
                    failover_model = UnifiedModel(provider="anthropic", model_name="claude-sonnet-4-5")
                    return failover_model.generate_content(prompt, system_instruction=system_instruction, generation_config=generation_config)

        # Universal Route (Anthropic via HTTP/LiteLLM)
        else:
            print(f"🔄 Fallback to HTTP via LiteLLM ({self.provider})...", flush=True)
            if self.provider == "anthropic":
                key = os.environ.get("ANTHROPIC_API_KEY") or get_secret("anthropic-api-key")
                if key: os.environ["ANTHROPIC_API_KEY"] = key
            elif self.provider == "openai":
                key = os.environ.get("OPENAI_API_KEY") or get_secret("openai-api-key")
                if key: os.environ["OPENAI_API_KEY"] = key
            
            temp = generation_config.get('temperature', 0.7) if generation_config else 0.7
            
            try:
                extra_headers = {}
                if "claude-sonnet-4-5" in self.model_name or "claude-3-5-sonnet" in self.model_name:
                    extra_headers = {"anthropic-beta": "max-tokens-3-5-sonnet-2024-07-15"}

                messages = []
                if system_instruction:
                    messages.append({"role": "system", "content": str(system_instruction)})
                messages.append({"role": "user", "content": prompt})

                # LiteLLM requires provider/model format for unknown/new models
                litellm_model = self.model_name
                if self.provider and f"{self.provider}/" not in litellm_model:
                    litellm_model = f"{self.provider}/{self.model_name}"

                response = completion(
                    model=litellm_model, 
                    messages=messages,
                    temperature=temp,
                    max_tokens=8192,
                    extra_headers=extra_headers
                )
                
                class MockResponse:
                    def __init__(self, content): self.text = content
                
                content = response.choices[0].message.content
                if not content:
                    content = "The model was unable to generate a response."
                return MockResponse(content)
                
            except Exception as e:
                print(f"❌ HTTP LiteLLM Error: {e}")
                class MockResponse:
                    def __init__(self, content): self.text = content
                return MockResponse("Error generating content.")

# --- MCP CLIENT ---

class RemoteTools:
    """Acts as a bridge to the MCP Sensory Tools Server."""
    def __init__(self, server_url="http://localhost:8080"):
        self.server_url = server_url.rstrip("/")
        self.is_gcp = "run.app" in server_url or "cloudfunctions.net" in server_url

    def _get_id_token(self):
        """Fetches an OIDC ID token from the metadata server (only works on GCP)."""
        try:
            auth_req = google.auth.transport.requests.Request()
            return id_token.fetch_id_token(auth_req, self.server_url)
        except Exception as e:
            print(f"Auth Hint: If local, ensure you are authenticated. Error: {e}")
            return None

    def call_tool(self, tool_name, arguments):
        """Standard MCP call interface."""
        print(f"MCP: Calling remote tool '{tool_name}'")
        headers = {}
        if self.is_gcp:
            token = self._get_id_token()
            if token: headers["Authorization"] = f"Bearer {token}"
        try:
            response = requests.post(
                f"{self.server_url}/messages", 
                json={"method": "tools/call", "params": {"name": tool_name, "arguments": arguments}},
                headers=headers, timeout=300, verify=certifi.where()
            )
            content_list = response.json().get("result", {}).get("content", [])
            
            text_outputs = []
            for item in content_list:
                if isinstance(item, dict) and "text" in item:
                    text_outputs.append(item["text"])
                else:
                    text_outputs.append(str(item))
            
            final_text = "\n".join(text_outputs)
            print(f"TELEMETRY: MCP Hub: [{tool_name}] execution complete. Size: {len(final_text)} chars.")
            return final_text
        except Exception as e:
            print(f"MCP Server Error: {e}")
            return f"MCP Server Error: {e}"

    def call(self, tool_name, arguments):
        return self.call_tool(tool_name, arguments)

def get_mcp_client():
    global mcp_client
    if mcp_client is None:
        mcp_client = RemoteTools(MCP_SERVER_URL)
    return mcp_client

# --- SHARED UTILITIES ---

def extract_json(text):
    """Hardened extraction of JSON from LLM responses."""
    if not text: return None
    markdown_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if markdown_match:
        content = markdown_match.group(1).strip()
    else:
        match = re.search(r'([\[\{][\s\S]*[\]\}])', text)
        if not match: return None
        content = match.group(1).strip()
    
    try:
        return json.loads(content)
    except Exception as e:
        print(f"FAILED INITIAL JSON PARSE: {e}. Attempting repair...")
        try:
            repaired = re.sub(r'(?<![:])\s*"\s*(?![,}])', r'\"', content)
            return json.loads(repaired)
        except:
            return None

def detect_audience_context(history):
    """Detects target audience from conversation history."""
    default = "Likely Senior Marketers, CTOs, and Founders, but you must value clarity and simplicity above all."
    if not history: return default
    hist_lower = history.lower()
    if any(kw in hist_lower for kw in ["8-grader", "8th grade", "explain like i'm 5", "eli5"]):
         return "An 8th-grader. Use extremely simple words, short sentences, and clear analogies."
    elif "non-technical" in hist_lower:
         return "Non-technical business owners. Focus on value and 'what it does'."
    return default

def safe_generate_content(model, prompt, system_instruction=None, generation_config=None):
    """Universal wrapper for model generation with safety refusal traps."""
    if generation_config is None: generation_config = {"temperature": 0.4}
    try:
        # Use UnifiedModel's native failover if applicable
        if hasattr(model, 'generate_content'):
            response = model.generate_content(prompt, generation_config=generation_config, system_instruction=system_instruction)
            text = response.text if hasattr(response, 'text') else str(response)
        else:
            # Fallback for raw model objects
            response = model.generate_content(prompt, generation_config=generation_config)
            text = response.text
            
        refusal_triggers = ["I encountered a safety limit", "unable to generate", "Error generating content"]
        if any(err in text for err in refusal_triggers):
             raise ValueError(f"Model returned Soft Refusal")
        return text.strip()
    except Exception as e:
        print(f"⚠️ safe_generate_content failure: {e}")
        return "The system is currently maximizing its cognitive load. Please try again in 60 seconds."


# --- MODULAR STYLE & SANITIZATION PROTOCOL (Zero-Loss Fidelity) ---

PROTOCOL_GROUNDING_RAI = """
- **CONTEXTUAL INTEGRITY (ZERO-INVENTION)**:
    1. **DATA FIDELITY**: You are FORBIDDEN from generating any specific statistic (%), project count (e.g., "120 projects"), or implementation claim that is not explicitly in the USER PROMPT, PROVIDED FILES, or VERIFIED SEARCH RESULTS. Use qualitative descriptions if data is absent.
    2. **LINK VERIFICATION**: Only generate URLs that have been explicitly provided or verified in the current turn's context. Never "guess" a logical URL path.
    3. **ARCHITECTURAL ANCHORING**: Claims must align with the provided context (e.g., architectural or contextual specs). If a claim contradicts the Primary Anchor (Prompt/Files), the anchor takes precedence.
    4. **GROUNDING DATA SUPREMACY (Universal)**: When GROUNDING DATA contains current information on ANY topic (APIs, news events, trend data, regulatory changes, market statistics, etc.), you MUST prioritize that information over your training data. Your training data has a knowledge cutoff; GROUNDING DATA represents real-time, verified information. Specific applications:
       - **Technical**: API versions, library deprecations, framework updates
       - **News/Events**: Current events, breaking news, recent announcements
       - **Trends**: Search volume data, market shifts, emerging patterns
       - **Regulatory**: New laws, policy changes, compliance updates
       - **Statistics**: Current metrics, recent studies, updated benchmarks
    5. **TEMPORAL VERIFICATION**: When GROUNDING DATA contains timestamps, publication dates, or version numbers, treat those as authoritative. If GROUNDING DATA shows information is outdated or superseded, you MUST use the current replacement mentioned in GROUNDING DATA.
    6. **SYNTHETIC PROVENANCE**: While transparency is required, PROHIBIT repetitive name-dropping of search platforms (e.g., "According to TripAdvisor", "Bing results show") in the body prose. Instead, weave findings into a cohesive narrative and rely on **INLINE ANCHORED LINKS** for primary attribution. Use platform names ONLY for high-level context or when the platform itself is the subject of the claim.
"""

PROTOCOL_VISUAL_TABULAR = """
- **MERMAID MANDATE**: You are strictly PROHIBITED from generating ASCII-based diagrams or logic maps (e.g., using arrows like `-->` or pipes `|` for flow). For any architectural maps, sequence flows, or visual logic, you MUST use `mermaid` code blocks. This ensures high-fidelity rendering via the MCP gateway.
- **MERMAID MODULARITY**: Only generate a diagram if the content genuinely involves a visual structure (e.g., a multi-step flow or system architecture). When a diagram IS warranted and the architecture is complex, prefer multiple modular diagrams (e.g., Phase 1 vs Phase 2) over one dense block to prevent transport failures (Slack 3k URL limit).
- **CODE BLOCKS — CONDITIONAL USE**: Only include fenced code blocks when the article is explicitly about a programming task, CLI command, or configuration syntax. Do NOT add code blocks as decoration or padding.
- **TABLE COMPACTION**: PROHIBIT blank lines within Markdown tables. All rows (header, separator, and data) MUST be contiguous for parser integrity.
"""

PROTOCOL_LITERARY_CORE = """
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
- **DESCRIPTIVE ANCHOR TEXT (SEO MANDATE)**: PROHIBIT non-descriptive hyperlink anchors: 'here', 'click here', 'this link', 'read more', 'learn more', 'source'. Every hyperlink MUST use contextually descriptive anchor text that conveys the topic or claim being cited (e.g., `context rot in long-context benchmarks` not `here`).
"""

PROTOCOL_FORMAT_CMS = """
- **OUTPUT TARGET: CMS_DRAFT (STRATEGIC PROTOCOL)**:
    1. **STRICT SEMANTIC HTML**: You MUST use semantic HTML tags ONLY. PROHIBIT all Markdown.
    2. **HEADER PROTOCOL**: Use `<h2>` and `<h3>` for all section headers. **ABSOLUTELY PROHIBIT** Markdown hashtags (`#`, `##`, `###`). The presence of a single hashtag is a protocol failure.
    3. **TABLE PROTOCOL**: Use strictly semantic `<table>`, `<tr>`, `<th>`, `<td>` tags. PROHIBIT Markdown pipe tables.
    4. **PROSE PROTOCOL**: Wrap every paragraph in `<p>` tags.
    5. **LISTS**: Use `<ul>`/`<li>` or `<ol>`/`<li>`.
    6. **CODE**: Use `<pre><code class="language-...">...</code></pre>`.
    7. **NO MARKDOWN ESCAPES**: Do not attempt to use `_` for italics or `**` for bold. Use `<em>` and `<strong>`.
"""

PROTOCOL_FORMAT_SLACK = """
- **OUTPUT TARGET: MODERATOR_VIEW**: If the target is a Slack brief or research discovery, use Markdown exclusively for tables (pipes: `|`), headers (`#`, `##`), and code blocks. PROHIBIT all HTML tags (no `<p>`, `<h2>`, etc.). Use blank lines for paragraph separation.
"""

PROTOCOL_ANTI_SLOB = """
- **STRATEGIC CONTEXT SANITIZATION (ANTI-META-TALK)**:
    - **UNIVERSAL PROHIBITION**: You are strictly PROHIBITED from mentioning internal strategic decision-making, competitive audits, or benchmarking scores when the user's latest message PRIMARY GOAL (intent) is classified as DEEP_DIVE or PSEO_ARTICLE.
    - **BANNED CATEGORIES**:
        1. **SEO/Metrics**: "competitor gap," "audit scores," "ranking analysis," "search volume," "AEO strategy," "moat factor," "technical density score."
        2. **Process/Turns**: "turn-based analysis," "Turn 1/2/3/4," "internal blueprint," "iterative refinement," "previous response," "vetted prompt."
        3. **Implementation Meta**: "strategic decision primitives," "policy enforcement point (PEP) architecture," "architectural logic implementation."
        4. **Comparative Bias**: "technical vacuum," "marketing hype," "marketing fluff," "displace leaders."
    - **DEFINITION SHIELD**: PROHIBIT "Define [Abbreviation] as [Full Name]" sentence structures. Integrate definitions naturally (e.g., "The Policy Enforcement Point (PEP)...") or assume professional context.
    - **TONE REPLACEMENT**: Instead of saying "Other guides score 2/10," simply present the authoritative technical finding with 10/10 technical depth. The "moat" is felt through your technical precision, not stated in prose.
"""

PROTOCOL_AUTHORITATIVE_GHOST = """
- **CRITICAL "NO META-TALK" PROTOCOL**:
    1. **ABSOLUTELY NO "Guides"**: Do NOT write "Here is how you would write this page..." or "This page structure is designed to...".
    2. **DIRECT OUTPUT**: Start immediately with the Page/Article Title (<h1> or #) or the first structural content section.
    3. **NO APOLOGIES**: If data is missing or unavailable, use a descriptive qualitative placeholder without apologizing to the reader.
    4. **AUTHORITATIVE VOICE**: You are the definitive source. Do not use phrases like "based on the provided context."
- **NARRATIVE FOCUS**: Weave facts into a cohesive, persuasive story. Avoid disjointed bullet points unless they support a specific technical breakdown.
- **ANCHOR TEXT PROTOCOL (HTML)**:
    - **PROHIBIT**: `<a href="...">here</a>`, `<a href="...">click here</a>`, `<a href="...">this link</a>`, `<a href="...">read more</a>`, `<a href="...">source</a>`.
    - **MANDATE**: Anchor text MUST describe the destination topic or claim. Example — CORRECT: `<a href="https://example.com">context rot in multi-agent retrieval pipelines</a>`. INCORRECT: `<a href="https://example.com">here</a>`.
    - **PLACEMENT**: Embed the link inline within the sentence at the exact phrase it attributes, not appended after the sentence as a trailing 'here'.
"""

PROTOCOL_PSEO_PAGE = """
- **ROLE: Specialized pSEO Data Weaver**: You create specific, data-rich pages for unique entity/location combinations.
- **DATA DENSITY**: Prioritize tables and lists over long paragraphs.
"""

PROTOCOL_LIFESTYLE_PERSONA = """
- **LIFESTYLE PERSONA**: You are a versatile Lifestyle Writer and Independent Curator. Your
  scope spans all Lifestyle sub-categories: Health & Wellness, Food & Drink, Travel &
  Adventure, Fashion & Beauty, Home & Decor, and Personal Development & Relationships.
- **TONE**: Conversational, relatable, and authentic. Write like a trusted friend with
  expertise — not a brand. Prioritise warmth and immediacy over formality.
- **PRACTICAL VALUE FIRST**: Every piece must deliver immediate value — a solution to a daily
  frustration, inspiration for a lifestyle change, or a pleasant mental escape. Lead with this.
- **FORMAT INTELLIGENCE**: Detect the article type from context and shape structure accordingly:
    - **How-To / Tutorial**: Open with the outcome, then numbered actionable steps.
      Each step = one concrete action. No filler steps.
    - **Listicle**: Scannable bold lead-ins per item. Items are self-contained and ordered
      by relevance — not alphabetically or arbitrarily.
    - **Personal Essay**: Open with a specific, vivid scene. Draw the universal lesson out
      through story — never state the moral directly.
    - **Product Review / Recommendation**: Lead with the verdict. Support with specific,
      honest observations. PROHIBIT vague praise ("great quality", "highly recommend").
- **SYNTHETIC NARRATIVE**: Avoid name-dropping review platforms (e.g., TripAdvisor, Yelp).
  Weave findings into your own voice and use inline links for attribution only.
- **NO VISUALS DIRECTIVE**: Do not reference or describe images. The pipeline does not process
  visuals for Lifestyle articles at this time.
"""

PROTOCOL_HUMANITIES_PERSONA = """
- **HUMANITIES PERSONA**: You are a Sociopolitical Analyst, Researcher, and Public Intellectual.
  Your scope spans all Humanities sub-categories: Politics & Governance, Law & Policy, History,
  Sociology & Culture, Security & Geopolitics, Ethics & Philosophy, and International Relations.
- **TONE**: Objective, authoritative, and nuanced. Present complexity without sensationalism.
  Avoid tribal framing or loaded emotional language. Let the evidence carry the weight.
- **EVIDENCE ANCHORING**: Every significant claim must be grounded in a real event, established
  framework, verified statistic, or named source. PROHIBIT vague generalisations
  ("many experts say," "it is widely believed").
- **MULTI-PERSPECTIVE INTEGRITY**: Where a topic is genuinely contested, represent the strongest
  version of competing positions before resolving. Do not strawman opposing views.
- **FORMAT INTELLIGENCE**: Detect the article type from context and shape structure accordingly:
    - **Analysis**: Open with the core finding or insight, then build the systemic argument.
      Use sub-sections for drivers, implications, and outlook.
    - **Explainer**: Lead with the plain-language definition, then layer in complexity.
      Use analogies to ground abstract concepts before introducing formal terms.
    - **Opinion / Commentary**: State the argument in the first paragraph — don't bury the
      thesis. Support with evidence, not assertion. Acknowledge the strongest counterpoint.
    - **Profile (Person / Institution / Movement)**: Open with a defining action or moment,
      then contextualise through their broader significance, not biography alone.
- **SYSTEMIC FOCUS**: Prioritise structural drivers and institutional forces over individual
  personalities. Name people where necessary, but keep the lens on systems.
- **NO TECHNICAL ARTEFACTS**: PROHIBIT code blocks, mermaid diagrams, and JSON-LD schemas.
  Use prose, simple Markdown tables (for comparative data), and inline citations.
"""

def get_system_instructions(intent: str, output_target: str, topic_sector: str = "TECHNICAL") -> str:
    """
    Architectural fix to assemble instructions modularly based on intent, target, and topic sector.
    Prevents token waste while ensuring 100% rule fidelity for relevant tasks.
    """
    intent = intent.upper().strip()
    instructions = "You are a professional AI Assistant. Protocols:\n"
    instructions += PROTOCOL_GROUNDING_RAI

    # Sector-Aware Visual Protocol
    if topic_sector == "TECHNICAL":
        instructions += PROTOCOL_VISUAL_TABULAR
    else:
        instructions += "\n- **VISUAL RESTRAINT**: For non-technical LIFESTYLE or HUMANITIES content, PROHIBIT Mermaid flowcharts and technical JSON-LD schema blocks unless explicitly requested. Use prose or simple Markdown tables instead.\n"
    
    if output_target == "CMS_DRAFT":
        instructions += PROTOCOL_FORMAT_CMS
        instructions += "\n- **CRITICAL**: CMS_DRAFT MODE ACTIVE. YOU ARE FORBIDDEN FROM OUTPUTTING MARKDOWN HASHTAGS (#). USE HTML TAGS ONLY."
    else:
        instructions += PROTOCOL_FORMAT_SLACK
        
    # Condition: High-Fidelity intents get the full Literary and Anti-Slob treatment.
    high_fidelity_intents = [
        "DEEP_DIVE", "PSEO_ARTICLE", "PSEO_PAGE", "PSEO_LP", 
        "TECHNICAL_EXPLANATION", "BLOG_OUTLINE", "AUTHOR", 
        "REWRITE", "REFINE", "THEN_VS_NOW_PROPOSAL",
        "DIRECT_ANSWER", "SIMPLE_QUESTION", "FORMAT_GENERAL", "OPERATIONAL_REFORMAT"
    ]
    if intent in high_fidelity_intents:
        instructions += PROTOCOL_LITERARY_CORE

        # Sector-Aware Persona
        if topic_sector == "LIFESTYLE":
            instructions += PROTOCOL_LIFESTYLE_PERSONA
        elif topic_sector == "HUMANITIES":
            instructions += PROTOCOL_HUMANITIES_PERSONA
        else:
            instructions += "\n- **TECHNICAL PERSONA**: You are a Specialized Technical Writer. Use precise, efficient, and concept-driven prose. Prioritise clarity and conceptual depth. Include code examples or diagrams ONLY when they directly illustrate a point that prose alone cannot convey — never as padding."
        
    if intent == "PSEO_PAGE":
        instructions += PROTOCOL_PSEO_PAGE
        
    if intent in ["PSEO_ARTICLE", "PSEO_LP", "PSEO_PAGE", "DEEP_DIVE"]:
        instructions += PROTOCOL_ANTI_SLOB
        if output_target == "CMS_DRAFT":
            instructions += PROTOCOL_AUTHORITATIVE_GHOST
        
    return instructions.strip()

def extract_labeled_sources(context_str):
    """
    Extracts URLs from context and returns a labeled list for the LLM.
    """
    if not context_str: return ""
    urls = list(set(re.findall(r'https?://[^\s<>"]+|www\.[^\s<>"]+', str(context_str))))
    if not urls: return ""
    
    labeled = "\n### VERIFIED SOURCES (GROUNDING ANCHORS):\n"
    for i, url in enumerate(urls[:15], 1):
        labeled += f"SOURCE_{i}: {url}\n"
    labeled += "\nREQUIRED: Use inline links to these sources for any factual claim derived from them.\n"
    return labeled

def get_stylistic_mentors(session_id=None):
    """
    Dynamically retrieves randomized stylistic mentors from the shared linguistic palette.
    Seeded by session_id to ensure consistency per-thread while maintaining cross-thread variety.
    """
    # Try multiple common relative paths for the shared JSON
    current_dir = os.path.dirname(__file__)
    paths_to_try = [
        os.path.join(current_dir, "linguistic_palette.json"),
        os.path.join(current_dir, "..", "shared", "linguistic_palette.json"),
        os.path.join(current_dir, "shared", "linguistic_palette.json"),
    ]
    
    palette_path = next((p for p in paths_to_try if os.path.exists(p)), None)
    
    try:
        if not palette_path:
            return ""
            
        with open(palette_path, 'r') as f:
            palette = json.load(f)
            
        # Seed for stable variety within a session
        if session_id:
            seed_val = int(hashlib.md5(session_id.encode()).hexdigest(), 16) % (10**8)
            rng = random.Random(seed_val)
        else:
            rng = random.Random()
            
        mentors = []
        
        # Pick 3 transitions from each core category
        categories = palette.get("transitions", {})
        key_categories = [
            "opposition_limitation_contradiction", 
            "cause_condition_purpose", 
            "examples_support_emphasis",
            "agreement_addition_similarity"
        ]
        
        for cat in key_categories:
            words = categories.get(cat, [])
            if words:
                sample = rng.sample(words, min(3, len(words)))
                mentors.append(f"- {cat.replace('_', ' ').title()}: {', '.join(sample)}")
            
        # Pick 6 active verbs
        verbs = palette.get("verbs", {}).get("active_mentors", [])
        if verbs:
            sample_verbs = rng.sample(verbs, min(6, len(verbs)))
            mentors.append(f"- Active Verb Mentors: {', '.join(sample_verbs)}")
            
        return "\n### DYNAMIC STYLE PALETTE (TURN MENTORS):\nREQUIRED: Use at least two (2) words from the mentors below to maintain linguistic texture.\n" + "\n".join(mentors)
    except Exception as e:
        print(f"⚠️ get_stylistic_mentors error: {e}")
        return ""

def classify_topic_sector(topic: str, flash_model=None) -> str:
    """
    Semantic Sector Classification (ADK-FLASH):
    Classifies a topic into LIFESTYLE, HUMANITIES, or TECHNICAL sectors.
    """
    topic_str = str(topic).strip()
    
    # 1. Semantic Match (Specialized Flash Call)
    if flash_model:
        classification_prompt = f"""
        Strictly classify the following TOPIC into one of three SECTORS based on its semantic nature. 
        The themes listed below are non-exhaustive EXAMPLES:
        
        1. LIFESTYLE (Example Themes: Tourism, Food, Travel, Lifestyle, Hangouts, Entertainment, Recreation)
        2. HUMANITIES (Example Themes: Politics, Sociology, Governance, Security, Law, Policy, History, Ethics)
        3. TECHNICAL (Example Themes: Software, AI, Engineering, Infrastructure, Data Systems, Scientific Method)
        
        TOPIC: "{topic_str}"
        
        Return ONLY the sector name (e.g., "LIFESTYLE").
        """
        
        try:
            # Use safe_generate_content if available in scope or just call model directly
            raw_sector = safe_generate_content(flash_model, classification_prompt, generation_config={"temperature": 0.0, "max_output_tokens": 10})
            sector = str(raw_sector).upper().strip()
            
            if any(s in sector for s in ["LIFESTYLE", "HUMANITIES", "TECHNICAL"]):
                for s in ["LIFESTYLE", "HUMANITIES", "TECHNICAL"]:
                    if s in sector:
                        print(f"  + Semantic Sector Classification: {s}")
                        return s
        except Exception as e:
            print(f"⚠️ Semantic Triage Failed: {e}. Falling back to keywords.")
    
    # 2. Keyword Fallback
    t = topic_str.lower()
    lifestyle_keywords = [
        "hangout", "places to", "travel", "food", "lifestyle", "tourism", "restaurant", 
        "club", "bar", "visit", "shopping", "mall", "lake", "hotel", "cafe", "nightlife"
    ]
    if any(kw in t for kw in lifestyle_keywords):
        return "LIFESTYLE"
        
    humanities_keywords = [
        "politics", "terrorism", "governance", "nigeria", "policy", "societal", 
        "public", "international", "un ", "security", "human rights", "law", "ethics"
    ]
    if any(kw in t for kw in humanities_keywords):
        return "HUMANITIES"
        
    return "TECHNICAL"

def convert_html_to_markdown(html_str):
    """Converts architectural HTML into Slack-friendly Markdown with a clear hierarchy."""
    if not html_str: return ""
    text = re.sub(r'(?i)^(Part \d+:?|Here is.*?:\s*)', '', str(html_str)).strip()
    
    # Leaky Markdown Fix
    if "<code>" in text or "```" in text:
        text = re.sub(r'(?<!<code>)```[a-z]*\n?([\s\S]*?)\n?```(?!</code>)', r'<code>\1</code>', text, flags=re.IGNORECASE)
        text = re.sub(r'(?<!<code>)`([^`\n]+)`(?!</code>)', r'<code>\1</code>', text, flags=re.IGNORECASE)

    text = re.sub(r'^\s*#{1,6}\s*(.*?)$', r'*\1*', text, flags=re.MULTILINE)
    protected_blocks = {}
    def protect_block(md_content):
        placeholder = f"__PROTECTED_CODE_{uuid.uuid4().hex}__"
        protected_blocks[placeholder] = md_content
        return placeholder

    # Code Blocks
    def code_handler(match):
        attrs = match.group(1) if match.lastindex >= 1 else ""
        code = match.group(2) if match.lastindex >= 2 else match.group(0)
        code = html.unescape(re.sub(r'<[a-z/][^>]*>', '', code, flags=re.IGNORECASE))
        if '\n' in code.strip():
            return protect_block(f"```\n{code.strip()}\n```")
        return protect_block(f"`{code.strip()}`")

    text = re.sub(r'<pre><code([^>]*)>([\s\S]*?)</code></pre>', code_handler, text, flags=re.IGNORECASE)
    text = re.sub(r'<code([^>]*)>([\s\S]*?)</code>', code_handler, text, flags=re.IGNORECASE)

    # Tables
    def table_handler(match):
        rows = re.findall(r'<tr[^>]*>(.*?)</tr>', match.group(0), re.DOTALL | re.IGNORECASE)
        if not rows: return ""
        md_table = []
        for i, row in enumerate(rows):
            cols = [re.sub(r'<[^>]+>', '', c).strip() for c in re.findall(r'<(?:td|th)[^>]*>(.*?)</(?:td|th)>', row, re.DOTALL | re.IGNORECASE)]
            if not cols: continue
            md_table.append("| " + " | ".join(cols) + " |")
            if i == 0 and len(rows) > 1: md_table.append("| " + " | ".join(["---"] * len(cols)) + " |")
        return "\n" + "\n".join(md_table) + "\n"

    text = re.sub(r'<table[^>]*>(.*?)</table>', table_handler, text, flags=re.DOTALL | re.IGNORECASE)

    # Headers & Lists
    text = re.sub(r'<h[1-4][^>]*>(.*?)</h[1-4]>', r'\n*\1*\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<li>(.*?)</li>', r'  • \1\n', text, flags=re.IGNORECASE)
    text = text.replace('<ul>', '\n').replace('</ul>', '\n').replace('<ol>', '\n').replace('</ol>', '\n')
    text = text.replace('<p>', '').replace('</p>', '\n\n').replace('<br>', '\n').replace('<br/>', '\n')
    
    text = re.sub(r'<[a-z/][^>]*>', '', text, flags=re.IGNORECASE)
    
    # Mermaid Diagram Restoration (Specialist)
    def restore_mermaid_blocks(match):
        try:
            return get_mcp_client().call_tool("render_mermaid", {"mermaid_code": match.group(1), "format": "markdown"})
        except Exception as e:
            print(f"Mermaid Render Error: {e}")
            return f"```mermaid\n{match.group(1)}\n```"

    text = re.sub(r'<pre><code class="language-mermaid">([\s\S]*?)</code></pre>', restore_mermaid_blocks, text, flags=re.IGNORECASE)
        
    # Final Restore
    for placeholder, original in protected_blocks.items():
        text = text.replace(placeholder, original)
    
    return re.sub(r'\n{3,}', '\n\n', text).strip()
