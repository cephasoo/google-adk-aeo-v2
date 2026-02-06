# --- /worker-story/main.py ---
import functions_framework
from flask import jsonify
import vertexai
import litellm
from litellm import completion
from vertexai.generative_models import GenerativeModel, HarmCategory, HarmBlockThreshold
import google.cloud.logging
import logging
import base64

litellm.drop_params = True  # CRITICAL: Handle unsupported params for new models (like gpt-5)
from vertexai.language_models import TextEmbeddingModel
import os
import json
import re
import uuid
import datetime
import requests
import certifi # Explicitly kept to avert legacy SSL issues
import google.auth
import google.auth.transport.requests
from google.oauth2 import id_token
from google.cloud import firestore, secretmanager
from google.cloud.firestore_v1.vector import Vector
import concurrent.futures

# --- Logging Setup ---
try:
    logging_v_client = google.cloud.logging.Client()
    logging_v_client.setup_logging()
except Exception:
    logging_v_client = None # Fallback to standard logging if local

# --- Configuration ---
# --- Configuration ---
try:
    _, project_id_auth = google.auth.default()
    PROJECT_ID = os.environ.get("PROJECT_ID", project_id_auth)
except:
    PROJECT_ID = os.environ.get("PROJECT_ID")

LOCATION = os.environ.get("LOCATION")
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-2.5-pro") 
MODEL_PROVIDER = os.environ.get("MODEL_PROVIDER", "vertex_ai")
FLASH_MODEL_NAME = os.environ.get("FLASH_MODEL_NAME", "gemini-2.5-flash-lite")
RESEARCH_MODEL_NAME = os.environ.get("RESEARCH_MODEL_NAME", "gemini-2.5-flash-lite") # NEW: For High Context
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
N8N_PROPOSAL_WEBHOOK_URL = os.environ.get("N8N_PROPOSAL_WEBHOOK_URL") 
INGESTION_API_KEY = os.environ.get("INGESTION_API_KEY")
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")  # Explicit fallback for code analysis
MAX_LOOP_ITERATIONS = 2
DEFAULT_GEO = os.environ.get("DEFAULT_GEO", "Global")
TRACKER_SERVER_URL = os.environ.get("TRACKER_SERVER_URL", "http://localhost:8081")
MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://localhost:8080")

# --- Safety Configuration (ADK/RAI Compliant) ---
safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
}


# --- Global Clients ---
unimodel = None
flash_model = None
research_model = None # NEW: The High-Context Specialist
specialist_model = None
search_api_key = None
db = None
mcp_client = None
secret_client = None

# --- SECRET MANAGER ---
def get_secret(secret_id):
    """
    Retrieves a secret from Google Cloud Secret Manager.
    Includes an environment variable fallback for local development or manual overrides.
    """
    global secret_client
    
    # 1. Check environment variables first (High Speed / Local Dev)
    env_key = secret_id.upper().replace("-", "_")
    env_val = os.environ.get(env_key)
    if env_val:
        return env_val
        
    # 2. Fallback to Secret Manager
    try:
        if secret_client is None:
            secret_client = secretmanager.SecretManagerServiceClient()
        
        name = f"projects/{PROJECT_ID}/secrets/{secret_id}/versions/latest"
        response = secret_client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8")
    except Exception as e:
        print(f"⚠️ Secret Manager error for {secret_id}: {e}")
        return None

# --- MCP CLIENT ---
class RemoteTools:
    """
    Acts as a bridge to the MCP Sensory Tools Server.
    """
    def __init__(self, server_url="http://localhost:8080"):
        self.server_url = server_url.rstrip("/")
        # Automatically detect if we need authentication (Cloud Run or Cloud Functions)
        self.is_gcp = "run.app" in server_url or "cloudfunctions.net" in server_url

    def _get_id_token(self):
        """Fetches an OIDC ID token from the metadata server (only works on GCP)."""
        try:
            auth_req = google.auth.transport.requests.Request()
            # The audience must be the service URL
            return id_token.fetch_id_token(auth_req, self.server_url)
        except Exception as e:
            print(f"Auth Hint: If local, ensure you are authenticated. Error: {e}")
            return None

    def call_tool(self, tool_name, arguments):
        """Standard MCP call interface."""
        print(f"MCP: Calling remote tool '{tool_name}' with args {arguments}")
        
        headers = {}
        if self.is_gcp:
            token = self._get_id_token()
            if token:
                headers["Authorization"] = f"Bearer {token}"

        try:
            # We use a simple POST to our FastAPI MCP wrapper
            response = requests.post(
                f"{self.server_url}/messages", 
                json={
                    "method": "tools/call",
                    "params": {
                        "name": tool_name,
                        "arguments": arguments
                    }
                },
                headers=headers,
                timeout=300,
                verify=certifi.where()
            )
            if response.status_code != 200:
                logging.error(f"MCP Server Error {response.status_code}: {response.text}")
                return f"Error: MCP Server returned {response.status_code}"
            
            data = response.json()
            # Extract text from MCP response format
            content = data.get("result", {}).get("content", [])
            if content:
                text_out = content[0].get("text", "No text returned.")
                logging.info(f"MCP Tool Success: {tool_name} (Length: {len(text_out)})")
                print(f"  -> MCP OUT [{tool_name}]: {text_out[:150].replace(chr(10), ' ')}...")
                return text_out
            logging.warning(f"MCP Tool '{tool_name}' returned empty content.")
            return "Empty response from tool."
        except Exception as e:
            return f"MCP Error: {str(e)}"

    def call(self, tool_name, arguments):
        """Backward compatibility for legacy calls in the codebase."""
        return self.call_tool(tool_name, arguments)

def get_mcp_client():
    global mcp_client
    if mcp_client is None:
        url = MCP_SERVER_URL
        logging.info(f"MCP CLIENT INIT: Connecting to {url}")
        mcp_client = RemoteTools(url)
    return mcp_client

# --- UNIFIED MODEL ADAPTER (The Brain Switch) ---
class UnifiedModel:
    """
    Routes requests to Vertex AI, OpenAI, or Anthropic based on configuration.
    """
    def __init__(self, provider, model_name):
        self.provider = provider
        self.model_name = model_name
        
        # Native Vertex Initialization (Only if using Google)
        if provider == "vertex_ai":
            vertexai.init(project=PROJECT_ID, location=LOCATION)
            self._native_model = GenerativeModel(model_name, safety_settings=safety_settings)
            print(f"✅ Loaded Native Vertex Model: {model_name} (Safety Active)", flush=True)

    def generate_content(self, prompt, generation_config=None, max_retries=3):
        """
        Universal generation function with safety catching and Exponential Backoff.
        """
        import time
        import random
        
        # PATH A: Native Vertex AI
        if self.provider == "vertex_ai":
            retries = 0
            while retries <= max_retries:
                try:
                    # Ensure max_output_tokens is set for Vertex AI
                    if generation_config is None: generation_config = {}
                    if "max_output_tokens" not in generation_config: generation_config["max_output_tokens"] = 8192
                    
                    response = self._native_model.generate_content(prompt, generation_config=generation_config)
                    
                    # Robust Safety Check: Some SDK versions throw exceptions, others return empty candidates
                    if not response.candidates or response.candidates[0].finish_reason == 3: # 3 = SAFETY
                         raise ValueError("Safety Block via FinishReason")
                    
                    return response
                except Exception as e:
                    error_msg = str(e).lower()
                    # Check for Rate Limit / Quota errors (429)
                    if "429" in error_msg or "resource exhausted" in error_msg:
                        if retries < max_retries:
                            wait_time = (2 ** retries) + random.uniform(0, 1)
                            print(f"⚠️ Vertex 429 (Quota): Retrying in {wait_time:.2f}s... (Attempt {retries+1}/{max_retries})")
                            time.sleep(wait_time)
                            retries += 1
                            continue
                    
                    # If it's a safety block or we've exhausted retries, log and return fallback
                    print(f"⚠️ Vertex AI Safety/SDK Error: {e}")
                    log_safety_event("generation_error", {"prompt": prompt, "error": str(e)})
                    
                    # Return Mock for Fallback
                    class MockResponse:
                        def __init__(self, content): self.text = content
                    return MockResponse("I encountered a safety limit or internal error. Could you rephrase?")

        # PATH B: Universal Route (OpenAI / Anthropic via LiteLLM)
        else:
            print(f"🔄 Switching to {self.model_name} via LiteLLM ({self.provider})...", flush=True)
            
            # Inject Keys for LiteLLM
            if self.provider == "anthropic" and ANTHROPIC_API_KEY:
                os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY
            elif self.provider == "openai" and OPENAI_API_KEY:
                os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
            
            # Normalize Parameters
            temp = generation_config.get('temperature', 0.7) if generation_config else 0.7
            
            try:
                # ADK FIX: Inject Beta Header for 8k Output (Required for 20240620)
                extra_headers = {}
                if "claude-sonnet-4-5" in self.model_name or "claude-3-5-sonnet" in self.model_name:
                    extra_headers = {"anthropic-beta": "max-tokens-3-5-sonnet-2024-07-15"}

                # LiteLLM Call
                response = completion(
                    model=self.model_name, 
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temp,
                    max_tokens=8192,
                    extra_headers=extra_headers
                )
                
                # Create a Mock Response Object to mimic Vertex AI's SDK
                # This ensures downstream code (getting .text) doesn't break
                class MockResponse:
                    def __init__(self, content):
                        self.text = content
                
                content = response.choices[0].message.content
                if not content:
                    print(f"⚠️ LiteLLM Warning: Received EMPTY content from {self.model_name}")
                    content = "The model was unable to generate a response. This might be due to a safety filter or an extremely large context."
                
                return MockResponse(content)
                
            except Exception as e:
                print(f"❌ LiteLLM Error: {e}", flush=True)
                # Fallback to an empty mock handling - Required re-definition due to scope
                class MockResponse:
                    def __init__(self, content): self.text = content
                return MockResponse("Error generating content.")

    def analyze_citation(self, prompt_text, scrape_content):
        """Calls the Sensory Tracker service to audit citations."""
        try:
            url = f"{TRACKER_SERVER_URL.rstrip('/')}/analyze"
            response = requests.post(url, json={
                "prompt": prompt_text,
                "content": scrape_content
            }, timeout=30, verify=certifi.where())
            if response.status_code == 200:
                return response.json()
            return {"cited": False, "error": f"Tracker status {response.status_code}"}
        except Exception as e:
            print(f"⚠️ Sensory Tracker Error: {e}")
            return {"cited": False, "error": str(e)}


# --- Safety Utils ---
def scrub_pii(text):
    """Simple PII scrubbing for memory and logs."""
    if not isinstance(text, str): return text
    # Mask emails
    text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[EMAIL_MASKED]', text)
    # Mask common phone formats
    text = re.sub(r'\+?\d{1,3}[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}', '[PHONE_MASKED]', text)
    return text

def detect_audience_context(history):
    """
    Search conversation history for tone or audience instructions (e.g., "8th-grader").
    Returns a string describing the target audience.
    """
    default = "Likely Senior Marketers, CTOs, and Founders, but you must value clarity and simplicity above all. Use plain English and avoid dense jargon even when discussing complex technical concepts."
    if not history: return default
    
    hist_lower = history.lower()
    if "8-grader" in hist_lower or "8th grade" in hist_lower or "explain like i'm 5" in hist_lower or "eli5" in hist_lower:
         return "An 8th-grader. Use extremely simple words, short sentences, and clear analogies. Avoid jargon."
    elif "non-technical" in hist_lower:
         return "Non-technical business owners. Focus on value and 'what it does' rather than 'how it works'."
    
    return default

def log_safety_event(event_name, data):
    """Logs safety events to Google Cloud Logging for audit traces."""
    global logging_v_client
    try:
        if logging_v_client:
            logger = logging_v_client.logger("safety_audit")
            # CLAMPING: Truncate large payloads to stay under 256KB Cloud Logging limit
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
        else:
            print(f"Safety Event Logged (Local): {event_name} - {data}")
    except Exception as e:
        print(f"Logging Block: {e}")

# --- SYSTEM HARDENING: Centralized Safe Generator ---
def safe_generate_content(model, prompt, generation_config=None):
    """
    A robust wrapper for model.generate_content that:
    1. Catches Exceptions (Legacy Vertex Failures)
    2. Catches Soft Failures (GPT-5 Refusal Strings like 'Error generating content')
    3. Handles Automatic Fallback to Specialist Model (Claude)
    """
    global specialist_model
    
    # Default config if None
    if generation_config is None: generation_config = {"temperature": 0.4}
    
    # 1. Primary Attempt
    try:
        response = model.generate_content(prompt, generation_config=generation_config)
        text = response.text
        
        # 2. Refusal Trap (The "Soft Failure" Check)
        refusal_triggers = ["I encountered a safety limit", "unable to generate", "Error generating content"]
        if any(err in text for err in refusal_triggers):
             raise ValueError(f"Model returned Soft Refusal: {text[:50]}...")
             
        return text.strip()

    except Exception as e:
        print(f"⚠️ SafeGen Primary Failed ({getattr(model, 'model_name', 'Unknown')}): {e}")
        
        # 3. Fallback to Specialist (Claude)
        if specialist_model and model != specialist_model:
            print(f"🔄 SafeGen FAILOVER: Switching to Specialist Model (Anthropic)...")
            try:
                # Ensure Key
                if not os.environ.get("ANTHROPIC_API_KEY") and ANTHROPIC_API_KEY:
                     os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY
                
                # ADK FIX: Add explicit delay for 429 mitigation
                import time
                print(f"🔄 SafeGen FAILOVER: Cooling down for 2s before Anthropic call...")
                time.sleep(2)
                
                # Retry with backup
                fallback_resp = specialist_model.generate_content(prompt, generation_config={"temperature": 0.4})
                return fallback_resp.text.strip()
            except Exception as e2:
                # Catch Anthropic 429 specifically
                if "rate_limit_error" in str(e2).lower():
                    print("⚠️ Anthropic Rate Limit hit. Ultimate fallback triggered.")
                    return "Cognitive overload (429). Repurpose/Draft command received but requires higher quota. Please try a smaller file or wait 60s."
                
                print(f"❌ SafeGen Fallback Failed: {e2}")
                # Ultimate Safety Net: Allow clean failure handled by caller or return generic error
                return "The system is currently maximizing its cognitive load. Please try again in 60 seconds."
        else:
             return "Service is busy (429). Please try again later."

# --- Utils ---
# (Moved ensure_markdown_spacing to line ~960 for consolidation as ensure_slack_compatibility)

# --- Input Santitation Utility ---
def sanitize_input_text(text):
    """Strips common markdown/Slack formatting from the start and end of a string."""
    if not isinstance(text, str):
        return ""
    # Strips leading/trailing whitespace, asterisks (bold), underscores (italics), and tildes (strikethrough)
    return text.strip().strip('*_~')

def fetch_slack_file_content(file_url, file_id, file_mode="hosted"):
    """
    Fetches content from Slack file using the private download URL.
    Improved to handle both snippets and hosted files consistently and detect Slack errors.
    """
    try:
        slack_token = get_secret("slack-bot-token")
        if not slack_token:
            print(f"❌ Failed to fetch Slack token")
            return None
        
        # We always use the file_url (url_private_download) for high reliability
        # Note: if it's a snippet, n8n/slack still provides a valid private download URL
        response = requests.get(
            file_url,
            headers={"Authorization": f"Bearer {slack_token}"},
            timeout=30,
            verify=certifi.where()
        )
        
        response.raise_for_status()
        content = response.text
        
        # CONTENT VALIDATION: Check if we downloaded a Slack error JSON instead of code
        # Example: {"ok":false,"error":"unknown_method"}
        if content.strip().startswith('{"ok":false') or '"error":"' in content[:100]:
            print(f"⚠️ Slack API Error detected in content for file {file_id}: {content[:100]}")
            return None

        print(f"✅ Downloaded file content ({len(content)} chars)")
        return content
        
    except Exception as e:
        print(f"❌ Failed to fetch file: {e}")
        return None

def extract_core_topic(user_prompt, history=""):
    """
    Distills a user's prompt into a high-precision Google Search query using SEO operators.
    """
    global flash_model
    print(f"Distilling core topic from: '{user_prompt[:100]}...'")
    
    extraction_prompt = f"""
    You are an expert Google Search operator.
    Convert the user's natural language request into a single, high-precision Google Advanced Search query.

    ### SEARCH OPERATOR RULES:
    1.  **Grouping with Parentheses:** Use `( )` to group synonyms when using OR. 
    2.  **Alternatives:** Use uppercase `OR` between entities.
    3.  **Exact Phrases:** Use quotes `""` for specific multi-word concepts.
    4.  **Exclusion:** Use `-` to remove noise.
    5.  **No Commas:** Do not use commas.
    6.  **Freshness:** If news, include "news" or "latest".
    7.  **Remove Fluff:** Remove "I want to know", "Please tell me", etc.
    8.  **Site Operator:** Use site: for specific domains.
    9.  **Filetype Operator:** Use filetype: for PDFs/docs.
    10. **Limit Length:** Under 10 words.
    11. **Implicit AND:** No need for AND.
    12. **CRITICAL: FILTER EDITORIAL INSTRUCTIONS.**
        - Remove words like: "outline", "draft", "strategy", "blog post", "article", "word count", "Grade 8", "1500+ words", "logic flow".
        - But PRESERVE discovery intent words: "why", "how", "reasons", "causes", "factors", "impact", "trends".
        - Focus ONLY on the SUBJECT MATTER and DISCOVERY INTENT.

    ### CONTEXT MAPPING:
    Use the provided HISTORY to resolve ambiguous terms like "he", "it", or "that" where the specific entity names aren't specified.

    HISTORY:
    {history}

    USER REQUEST:
    "{user_prompt}"

    SEARCH QUERY:
    """
    
    core_topic = safe_generate_content(unimodel, extraction_prompt)

    print(f"Distilled Core Topic: '{core_topic}'")
    return core_topic

def extract_target_word_count(text, history=""):
    """
    Semantically extracts the user's desired word count using a Dedicated Anthropic Model (Claude).
    Handles nuance: "1600+", "at least 2000", "short summary", "< 500".
    """
    global specialist_model
    try:
        # Fallback default
        default_count = 1500
        
        # 1. Use the Global Specialist Brain (Anthropic)
        prompt = f"""
        Analyze the current user request: "{text}"
        
        And the conversation history:
        {history}
        
        TASK:
        Extract the implied target word count as a single integer.
        
        ### INTERPRETATION RULES:
        1. **Preference Persistence**: If the history shows an earlier length preference (e.g., 'Make all these 2000 words'), and the current request doesn't override it, INHERIT that preference.
        2. **Explicit Numbers (Current)**: If the current request specifies a number (e.g., '1500 words'), use it as the primary target.
        2. **Floors/Expansion (+, >, at least)**: If the user implies a minimum (e.g., '1600+', '>1500', 'at least 2000'), provide a target slightly ABOVE that number to satisfy the request.
        3. **Ceilings/Constraint (<, less than, max)**: If the user implies a limit (e.g., '<1000', 'max 500'), provide a target BELOW that number.
        4. **Vague Descriptors**: 
           - "Deep dive" / "Long form" -> ~2000
           - "Short" / "Brief" -> ~800
           - "Standard" / "Default" -> 1500
        
        OUTPUT: Return ONLY the integer (e.g., 1800). Do not write sentences.
        """
        
        # 2. Generate
        response = safe_generate_content(specialist_model, prompt)
        
        # 3. Clean and Parse
        digits = re.sub(r'\D', '', response)
        
        if digits:
            return int(digits)
        else:
            return default_count
            
    except Exception as e:
        print(f"Semantic Extraction Failed: {e}. Using Default.")
        # Regex Fallback (Just in case)
        match = re.search(r'(\d+)\s*(?:word|token)s?', text, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return 1500

def strip_html_tags(text):
    """Removes HTML tags for clean context stitching and plain-text views."""
    return re.sub(r'<[^>]+>', '', text).strip()

def convert_html_to_markdown(html):
    """Converts architectural HTML into Slack-friendly Markdown with a clear hierarchy."""
    # 0. Handle H1 Title (The Master Label)
    text = re.sub(r'<h1>(.*?)</h1>', r'*\1*\n\n', html)
    
    # 1. Headers (H2 and H3)
    text = re.sub(r'<h2>(.*?)</h2>', r'\n*\1*\n', text)
    text = re.sub(r'<h3>(.*?)</h3>', r'_ \1 _\n', text)
    
    # 2. Sections to separators (More space)
    text = text.replace('</section>', '\n\n---\n\n')
    text = re.sub(r'<section[^>]*>', '', text)
    
    # 3. Code Blocks (<pre><code> to markdown backticks)
    text = re.sub(r'<pre><code[^>]*>(.*?)</code></pre>', r'```\n\1\n```', text, flags=re.DOTALL)

    # 4. Lists (Improved conversion)
    def list_item_handler(match):
        content = match.group(1).strip()
        # If content already starts with a number or bullet, just indent it
        if re.match(r'^(\d+\.|•|-|\*)', content):
            return f"  {content}\n"
        return f"  • {content}\n"

    text = re.sub(r'<li>(.*?)</li>', list_item_handler, text, flags=re.DOTALL)
    text = text.replace('<ul>', '\n').replace('</ul>', '\n').replace('<ol>', '\n').replace('</ol>', '\n')
    
    # 5. Paragraphs & Line Breaks
    text = text.replace('<p>', '').replace('</p>', '\n\n')
    text = text.replace('<br>', '\n').replace('<br/>', '\n')
    
    # 6. Final Strip of remaining tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Clean up excess newlines (Max 2)
    text = re.sub(r'\n{4,}', '\n\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def extract_trend_term(user_prompt, history=""):
    """
    Extracts the single core entity (Brand, Person, Product) for Google Trends.
    Removes locations, dates, and analytical intent words.
    """
    global flash_model
    print(f"Extracting Trend Entity from: '{user_prompt}'")
    
    prompt = f"""
    Extract the primary subject for a Google Trends "Interest Over Time" query.
    
    RULES:
    1. Identify the Core Entity (Person, Company, Product, or Concept).
    2. Remove all locations (e.g., 'in Nigeria', 'US').
    3. Remove all timeframes (e.g., 'last year', '2024').
    4. Remove analytical intent words (e.g., 'growth', 'trend', 'analysis', 'stats', 'performance').
    5. Remove quotes and special characters.
    6. Return ONLY the clean term.

    HISTORY (For Context):
    {history}

    USER INPUT: "{user_prompt}"
    
    Examples:
    - Input: "Analyze Starlink growth in Nigeria" -> Output: Starlink
    - Input: "How is iPhone 15 doing?" -> Output: iPhone 15
    - Input: "Inflation rates trends" -> Output: Inflation
    
    ENTITY:
    """
    try:
        # Use unimodel for speed/cost since this is a simple extraction
        entity = safe_generate_content(unimodel, prompt).replace('"', '').replace("'", "")
        print(f"Distilled Trend Entity: '{entity}'")
        return entity
    except Exception as e:
        print(f"Entity Extraction Failed: {e}")
        return user_prompt # Fallback

def mcp_detect_geo(query, history=""):
    """MCP Refactored: Calls the sensory hub to detect regional focus."""
    return get_mcp_client().call("detect_geo", {"query": query, "history": history})

def mcp_detect_intent(query):
    """MCP Refactored: Calls the sensory hub to detect research intent."""
    return get_mcp_client().call("detect_intent", {"query": query})

def get_search_api_key():
    global search_api_key
    if search_api_key is None:
        search_api_key = get_secret("google-search-api-key")
    return search_api_key

def extract_json(text):
    """
    Super-Listener: Hardened extraction of JSON from LLM responses.
    Handles markdown fences (```json) and raw braced strings.
    """
    if not text: return None
    
    # 1. Look for Markdown-wrapped blocks first
    markdown_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if markdown_match:
        content = markdown_match.group(1).strip()
    else:
        # 2. Fall back to finding the outermost braces
        brace_match = re.search(r'(\{[\s\S]*\})', text)
        if not brace_match: return None
        content = brace_match.group(1).strip()
    
    try:
        return json.loads(content)
    except Exception as e:
        print(f"FAILED INITIAL JSON PARSE: {e}. Attempting repair...")
        # REPAIR STRATEGY: Handle unescaped internal quotes
        # We look for quotes that are NOT part of a "key": or "value",/,"value"} pair
        try:
            # Simple heuristic: Escape quotes that aren't preceded by : or followed by , or }
            repaired = re.sub(r'(?<![:])\s*"\s*(?![,}])', r'\"', content)
            # Second attempt at parsing after repair
            return json.loads(repaired)
        except:
            print(f"FAILED REPAIRED JSON PARSE: Raw Content: {content[:200]}")
            return None

def extract_urls_from_text(text):
    """Extracts ALL URLs from a given string."""
    url_pattern = r'(https?://[^\s<>|]+)'
    return re.findall(url_pattern, text)

def chunk_text(text, chunk_size=1500, overlap=300):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start = end - overlap 
    return [c for c in chunks if c]

# --- Tools ---
#1. The Specialist Web Scraper Tool (MCP Refactored)
def fetch_article_content(url):
    return get_mcp_client().call("scrape_article", {"url": url})
    
#2. The Internal Knowledge Retrieval Tool (MCP Refactored)
def search_long_term_memory(query, session_id):
    return get_mcp_client().call("rag_search", {"query": query, "session_id": session_id})

# 2.5 The Compliance Knowledge Retrieval Tool (MCP Refactored)
def search_compliance_knowledge(query, doc_type=None, geo_scope=None):
    return get_mcp_client().call("compliance_rag_search", {
        "query": query,
        "doc_type": doc_type,
        "geo_scope": geo_scope
    })

# --- [Removed unused local parser _parse_serp_features (now handled by MCP Server)] ---

#4. The Specialist Google Web Search Tool (MCP Refactored)
def search_google_web(query):
    return get_mcp_client().call("google_web_search", {"query": query})

#5. The Specialist Google Trends Tool (MCP Refactored)
def search_google_trends(geo="US"):
    return get_mcp_client().call("google_trends", {"geo": geo})
    
# --- 6. ANALYSIS TOOL: Trend History (MCP Refactored) ---
def analyze_trend_history(query, geo="US"):
    return get_mcp_client().call("trend_analysis", {"query": query, "geo": geo})

#7. The Specialist Image Search Tool (MCP Refactored)
def search_google_images(query, num_results=5):
    return get_mcp_client().call("google_images_search", {"query": query})

#8. The Specialist Video Search Tool (MCP Refactored)
def search_google_videos(query, num_results=5):
    return get_mcp_client().call("google_videos_search", {"query": query})

#9. The Specialist Scholar Search Tool (MCP Refactored)
def search_google_scholar(query, num_results=3):
    return get_mcp_client().call("google_scholar_search", {"query": query})

# Fallback Search (MCP Refactored)
def google_simple_search(query):
    return get_mcp_client().call("google_simple_search", {"query": query})

# 9.5 The Vision Tool (MCP Refactored)
def analyze_image(image_data, prompt="Describe this image in detail."):
    return get_mcp_client().call("analyze_image", {"image": image_data, "prompt": prompt})

def generate_studio_image(prompt):
    return get_mcp_client().call("generate_image", {"prompt": prompt})

# --- Deep Navigator Logic ---
def analyze_deep_navigation(links_text, mission_topic, history_context):
    """
    Uses the Specialist Model to decide if we should recursively scrape a 'Child Link'.
    Returns: The URL to scrape (str) or None.
    """
    global specialist_model
    if not links_text or "No links found" in links_text: return None
    
    print(f"Deep Navigator: Analyzing {links_text.count('h')} potential links for topic '{mission_topic}'...")

    prompt = f"""
    You are a Senior Research Navigator.
    MISSION: We are researching '{mission_topic}'.
    CONTEXT: We just scraped a page and found these 'Child Links'.
    
    TASK:
    Analyze the links below. Determine if ONE of them is likely to contain CRITICAL information missing from our current understanding.
    
    CRITERIA:
    1. Relevance: Must be directly related to the core mystery/topic.
    2. Value Add: Must likely contain data, specs, or details not usually found on the home page.
    3. Safety: Do not click login, signup, or social media share links.
    
    LINKS FOUND:
    {links_text}
    
    DECISION:
    Return ONLY the URL of the single best link to click.
    If none are critical, return "NO_ACTION".
    """
    
    try:
        response = safe_generate_content(specialist_model, prompt)
        if "http" in response and "NO_ACTION" not in response:
            # Extract clean URL (handling potential markdown wrapper)
            url_match = re.search(r'(https?://[^\s<>"]+)', response)
            if url_match:
                return url_match.group(1)
        return None
    except Exception as e:
        print(f"Deep Navigator Error: {e}")
        return None

# 10. The Router (Updated with "Double-Tap" Analysis)
def find_trending_keywords(raw_topic, history_context="", session_id=None, images=None, mission_topic=None, session_metadata=None, initial_context=None, triage_intent=None):
    """
    An intelligent meta-tool that routes a query to the best specialized search tool.
    Uses 'raw_topic' for routing to allow for "USE_CONVERSATIONAL_CONTEXT" selection on conversational turns.
    
    UPGRADE: The 'ANALYSIS' tool now performs a "Double-Tap":
    1. Fetches Trend Stats (Quant)
    2. Fetches Top News (Qual)
    This prevents "blind" statistical answers (e.g., seeing a spike but not knowing why).
    """
    global unimodel, flash_model
    # --- MONITORING: Router Invocation ---
    # MISSION VS COMMAND: Grounding remains on mission_topic, routing on raw_topic
    grounding_subject = mission_topic or raw_topic
    print(f"TELEMETRY: find_trending_keywords starting. Topic='{raw_topic}' | GroundingSubject='{grounding_subject}'")
    if images: print(f"TELEMETRY: Grounding with {len(images)} images.")
    
    tool_logs = []
    context_snippets = [] # Return ONLY novel results found in this phase
    base_grounding = (initial_context or []).copy() # For internal decision making
    
    # 1. Internal RAG (We use the grounding_subject here)
    internal_context = search_long_term_memory(grounding_subject, session_id) if session_id else "No session_id provided for memory search."

    # Prepare RAG text for the prompt
    rag_text = "No relevant long-term memories found."
    if internal_context:
        # Flatten the list of strings into a single block of text
        rag_content_str = "\n".join([str(item) for item in internal_context])
        rag_text = rag_content_str
        
        context_snippets.append(internal_context)
        base_grounding.append(internal_context)
        tool_logs.append({"event_type": "tool_call", "tool_name": "internal_knowledge_retrieval", "status": "success", "content": rag_text})

    # 1.5 Visual Grounding (Recollective Vision)
    all_images = (images or []).copy()
    
    # Modernized regex for social media assets (matches ?format=jpg or standard extensions)
    # Allows extensions or social media format parameters
    img_pattern = r'(https?://[^\s<>|]+(?:\.(?:jpg|jpeg|png|webp|gif|pdf)|format=(?:jpg|jpeg|png|webp|gif))(?:\?[^\s<>|]*)?)'
    
    # Scan history for previously detected images
    if not all_images and history_context:
        history_images = re.findall(img_pattern, history_context)
        if history_images:
            print(f"Sensory Router: Found {len(history_images)} images in history. Using top 2 for recollective grounding.")
            all_images.extend(history_images[:2])
            
    # DEEP RECOLLECTION: If no images found, we use a semantic check to see if we SHOULD look for them in history.
    # 2. Automated Signal Detection (Smart Triage Inheritance)
    print("Sensory Router: Detecting Signals (Geo & Intent) via MCP Hub...")
    
    if session_metadata is None: session_metadata = {}
    prev_geo = session_metadata.get('detected_geo', DEFAULT_GEO)
    
    # ADK FIX: Use the flash-classified intent from Triage (triage_intent) if available, 
    # falling back to session metadata only if it's a cold restart.
    prev_intent = triage_intent or session_metadata.get('intent', 'FORMAT_GENERAL')

    # 3. CONSOLIDATED INTELLIGENCE ROUTER (Triple-Threat Prompt)
    router_prompt = f"""
    You are the 'Intelligence Router'. Assess the user's command and decide the path.
    USER COMMAND: '{raw_topic}'
    MISSION SUBJECT: '{grounding_subject}'
    CURRENT INTENT: '{prev_intent}'
    CURRENT GEO: '{prev_geo}'
    
    COLLECTED KNOWLEDGE:
    {history_context[:8000]}
    ---
    RESEARCH/MEMORY SAMPLES: 
    {str(base_grounding)[:15000]}
    
    PROTOCOL:
    1. VISUAL_INTENT: Require seeing asset? [YES/NO]
    2. GEO_PIVOT: If the user refers to a specific country, you MUST return the **ISO-3166-1 alpha-2** code.
       EXAMPLES: Nigeria=NG, South Africa=ZA, United States=US, United Kingdom=GB, Ghana=GH, Kenya=KE, India=IN, Germany=DE, France=FR, Canada=CA.
       If no pivot, return 'Inherit'.
    3. ADEQUACY_AUDIT: Have enough info? [SUFFICIENT/INSUFFICIENT]
    4. TOOL_RECRUITMENT: List tools (WEB, IMAGES, VIDEOS, TRENDS, ANALYSIS, COMPLIANCE, USE_CONVERSATIONAL_CONTEXT).
    
    OUTPUT FORMAT (Raw JSON Only):
    {{"visual_intent": "YES/NO", "new_geo": "...", "adequacy": "...", "selected_tools": [], "rationale": "..."}}
    """
    
    router_raw = safe_generate_content(specialist_model, router_prompt, generation_config={"temperature": 0.2})
    try:
        router_data = extract_json(router_raw)
        is_recollective_visual_query = router_data.get("visual_intent") == "YES"
        new_geo_signal = router_data.get("new_geo", "Inherit")
        
        # --- GEO NORMALIZATION (ISO-2 Enforcement for Trends/Analysis) ---
        if new_geo_signal == "Inherit":
            detected_geo = prev_geo
            target_geos = [prev_geo] if prev_geo else ["NG"]
        else:
            # Extract ALL 2-letter ISO codes using word boundaries
            iso_codes = re.findall(r'\b([A-Z]{2})\b', str(new_geo_signal).upper())
            if iso_codes:
                target_geos = iso_codes
                detected_geo = iso_codes[0] # Use the first one as a primary signal for single-use logic
                print(f"  -> Geo Pivots Detected: {target_geos} (Extracted from '{new_geo_signal}')")
            else:
                # Fallback to the raw string if no ISO codes found (Search might still use it)
                detected_geo = new_geo_signal
                target_geos = [new_geo_signal]
                print(f"  ⚠️ Warning: Non-ISO Geo detected: '{detected_geo}'. This may impact Trends/Analysis tools.")

        adequacy_score = router_data.get("adequacy")
        selected_tools = router_data.get("selected_tools", [])
        research_intent_raw = json.dumps({"intent": prev_intent, "rationale": router_data.get("rationale")})
        print(f"  -> Intelligence Router: {router_data.get('rationale')} | Target Geo: {detected_geo}")
    except Exception as e:
        print(f"⚠️ Router failed: {e}. Falling back.")
        is_recollective_visual_query = False
        detected_geo = prev_geo
        target_geos = [prev_geo] if prev_geo else ["NG"]
        adequacy_score = "INSUFFICIENT"
        selected_tools = ["WEB"]
        research_intent_raw = json.dumps({"intent": prev_intent})
    
    # 4. EXECUTION BRANCHING
    
    # 4.1 Recollective Vision (Conditional)
    if not all_images and is_recollective_visual_query and history_context:
        print("Sensory Router: Visual intent detected. Attempting Deep Recollection...")
        history_web_links = re.findall(r'(https?://[^\s<>|]+(?:twitter\.com|x\.com|instagram\.com|facebook\.com|threads\.net|tiktok\.com|youtube\.com|status/|p/)[^\s<>|]*)', history_context)
        if history_web_links:
            top_link = history_web_links[0].strip('><')
            scrape_data = get_mcp_client().call("scrape_article", {"url": top_link})
            found_img_links = re.findall(img_pattern, scrape_data)
            if found_img_links:
                all_images.extend(found_img_links[:2])
                tool_logs.append({"event_type": "tool_call", "tool_name": "recollective_scrape", "status": "success"})

    if all_images:
        print(f"Sensory Router: Analyzing {len(all_images)} images for visual grounding...")
        for i, img in enumerate(all_images):
            # Only analyze if we don't already have a valid insight for this specific image in the recent history
            # This avoids redundant expensive vision calls
            if f"[VISUAL_INSIGHT]" in history_context and i == 0 and "failed" not in history_context.lower():
                continue
                
            visual_context = analyze_image(img, prompt=f"Analyze this image in the context of: {grounding_subject}")
            insight = f"[VISUAL_INSIGHT]: {visual_context}"
            context_snippets.append(insight)
            base_grounding.append(insight)
            tool_logs.append({"event_type": "tool_call", "tool_name": "analyze_image", "status": "success", "content": insight})

    # 4.2 Gatekeeper Short-Circuit
    if adequacy_score == "SUFFICIENT" and "TRENDS" not in str(selected_tools):
        print("SENSORY GATEKEEPER: Knowledge is sufficient. Skipping search.")
        return {"context": context_snippets, "tool_logs": tool_logs, "research_intent": research_intent_raw, "detected_geo": detected_geo}

    # --- 0. CONVERSATIONAL SHORT-CIRCUIT ---
    # We only return early if USE_CONVERSATIONAL_CONTEXT is the ONLY tool selected.
    # If other tools (WEB, TRENDS, etc.) are present, we must execute them.
    if len(selected_tools) == 1 and "USE_CONVERSATIONAL_CONTEXT" in selected_tools[0]:
        return {"context": context_snippets, "tool_logs": tool_logs, "research_intent": research_intent_raw, "detected_geo": detected_geo}

    # --- CATEGORICAL SHARED EXTRACTION (Efficiency + Verbose Debugging) ---
    distilled_seo_query = None
    distilled_trend_term = None

    if any(t in ["WEB", "IMAGES", "VIDEOS", "SCHOLAR", "SIMPLE"] for t in selected_tools):
        print("Sensory Router: Category [SEO] active. Distilling high-precision query...")
        distilled_seo_query = extract_core_topic(raw_topic, history=history_context)
        print(f"  -> SEO Query: '{distilled_seo_query}'")

    if any("ANALYSIS" in t for t in selected_tools):
        print("Sensory Router: Category [TREND] active. Distilling trend entity...")
        distilled_trend_term = extract_trend_term(raw_topic, history=history_context)
        print(f"  -> Trend Entity: '{distilled_trend_term}'")
    
    # SEMANTIC ANCHOR: We only allow visual grounding to skip web research if the user's query
    # explicitly references visual elements AND we are not dealing with a NEW set of images.
    is_visual_query = any(word in raw_topic.lower() for word in ["boy", "eagle", "color", "look like", "see", "image", "promo", "flag", "picture", "photo", "this", "it"])
    
    # Check if we have new images to analyze vs. just relying on history
    has_new_images = len(all_images) > 0
    has_visuals_in_history = any("[VISUAL_INSIGHT]" in str(s) for s in context_snippets)
    
    if has_visuals_in_history and is_visual_query and not has_new_images:
        if "SIMPLE_QUESTION" in str(research_intent_raw):
            print("Sensory Router: Explicit visual grounding (history) detected. Skipping redundant WEB search.")
            selected_tools = [t for t in selected_tools if t not in ["WEB", "SIMPLE"]]
            if not selected_tools: selected_tools = ["USE_CONVERSATIONAL_CONTEXT"]
    # --- 1. GLOBAL KNOWLEDGE TOOLS (Independent of Trending Geos) ---
    knowledge_tools = [t.split(':')[0].strip() for t in selected_tools if t.split(':')[0].strip() in ["WEB", "IMAGES", "VIDEOS", "SCHOLAR", "SIMPLE"]]
    
    for choice in knowledge_tools:
        print(f"Executing Global Sensory Tool: '{choice}'")
        research_context = None
        tool_name = "unknown"

        if choice == "WEB":
            research_context = search_google_web(distilled_seo_query)
            tool_name = "serpapi_web_search_global"
        elif choice == "IMAGES":
            research_context = search_google_images(distilled_seo_query)
            tool_name = "serpapi_image_search"
        elif choice == "VIDEOS":
            research_context = search_google_videos(distilled_seo_query)
            tool_name = "serpapi_video_search"
        elif choice == "SCHOLAR":
            research_context = search_google_scholar(distilled_seo_query)
            tool_name = "serpapi_scholar_search"
        elif choice == "SIMPLE":
            pass # Fallback will handle it

        if research_context:
            context_snippets.append(research_context)
            tool_logs.append({"event_type": "tool_call", "tool_name": tool_name, "input": raw_topic, "status": "success"})

    # --- 1.5 COMPLIANCE KNOWLEDGE TOOLS ---
    if "COMPLIANCE" in selected_tools:
        print("Sensory Router: Category [COMPLIANCE] active. Searching Regulatory Hub (Multi-Geo)...")
        # We iteration through all detected geos for compliance grounding
        for geo_code in target_geos:
            print(f"  + Fetching Compliance Data for Region: {geo_code}")
            compliance_context = search_compliance_knowledge(grounding_subject, geo_scope=geo_code)
            if compliance_context and "No relevant" not in str(compliance_context):
                context_snippets.append(f"[Compliance Info - {geo_code}]:\n{compliance_context}")
                tool_logs.append({
                    "event_type": "tool_call", 
                    "tool_name": f"compliance_knowledge_retrieval_{geo_code}", 
                    "input": raw_topic, 
                    "status": "success"
                })

    # --- 2. REGIONAL TREND TOOLS (Region-Specific Discovery) ---
    trend_tools = [t.split(':')[0].strip() for t in selected_tools if t.split(':')[0].strip() in ["TRENDS", "ANALYSIS"]]
    
    if trend_tools:
        # Use target_geos list gathered by the Intelligence Router
        for geo_code in target_geos:
            print(f"--- Processing Regional Trends: '{geo_code}' ---")
            for choice in trend_tools:
                research_context = None
                tool_name = "unknown"

                if "TRENDS" in choice:
                    research_context = search_google_trends(geo=geo_code)
                    tool_name = f"serpapi_trends_search_{geo_code}"
                elif "ANALYSIS" in choice:
                    search_query = distilled_trend_term
                    print(f"  + Fetching Quantitative Trend Stats for '{search_query}' in '{geo_code}'...")
                    stats_context = analyze_trend_history(search_query, geo=geo_code)
                    print(f"DEBUG: analyze_trend_history returned {len(stats_context)} chars.", flush=True)

                    
                    # Qualitative Data Double-Tap
                    has_web_tool = any(t in ["WEB", "SIMPLE"] for t in selected_tools)
                    try:
                        if not has_web_tool:
                            print(f"  + Fetching Qualitative News Context for '{search_query}' in '{geo_code}' via MCP...")
                            news_text = get_mcp_client().call("google_news_search", {"query": search_query, "geo": geo_code})
                            research_context = f"[Region: {geo_code} Stats]:\n{stats_context}\n{news_text}"
                        else:
                            research_context = f"[Region: {geo_code} Stats]:\n{stats_context}"
                    except Exception: 
                        research_context = stats_context
                    tool_name = f"serpapi_trend_analysis_{geo_code}"

                if research_context:
                    context_snippets.append(research_context)
                    tool_logs.append({"event_type": "tool_call", "tool_name": tool_name, "input": raw_topic, "status": "success"})

    # --- 3. FALLBACK (If no context was gathered) ---
    if not context_snippets and "USE_CONVERSATIONAL_CONTEXT" not in selected_tools:
        fallback_query = distilled_seo_query or distilled_trend_term or raw_topic
        print(f"  ? Sensory research yielded no snippets. Initiating global fallback.")
        research_context = google_simple_search(fallback_query)
        context_snippets.append(research_context)
        tool_logs.append({"event_type": "tool_call", "tool_name": "google_simple_search_fallback", "input": raw_topic, "status": "success"})

    return { "context": context_snippets, "tool_logs": tool_logs, "research_intent": research_intent_raw, "detected_geo": detected_geo }


#11. The Topic Cluster Generator
def generate_topic_cluster(topic, context, history="", is_initial=True):
    global unimodel
    
    state_context = "This is a NEW proposal for a new thread." if is_initial else "This is a REVISION or EXTENSION of a previous strategy in an existing thread."
    
    prompt = f"""
    You are an expert SEO Architect. Create a comprehensive topic cluster for: "{topic}"
    
    STATE: {state_context}
    CONTEXT: {context}
    HISTORY: {history}
    
    ### EXPERT SEO PRINCIPLES:
    1. **Semantic Ecosystems (Entities over Strings):** Do not just match keywords. Identify the core concepts and entities required to be seen as an expert.
    2. **User Journey Maps:** Align clusters with the user's path from "Problem Awareness" (Top-of-funnel) to "Conversion/Solution" (Middle/Bottom-of-funnel).
    3. **Topical Authority Graphs:** Cover the topic's "nooks and crannies" to prove comprehensive expertise.
    4. **Information Architecture (The Library Model):** Ensure structure follows logical internal linking from the Pillar to sub-topics.
    
    ### GUIDELINES:
    1. If this is a REVISION (is_initial=False), you MAY start with a header like ":robot_face: The proposal has been REVISED" or similar context-aware greetings if appropriate.
    2. If this is an INITIAL proposal, be structured and authoritative.
    3. Ensure the output is strictly valid JSON for extraction.

    JSON SCHEMA:
    {{
      "pillar_page_title": "[The optimized H1 title for the pillar page]",
      "clusters": [
        {{ 
          "cluster_title": "[Core Entity/Broad Category Title]", 
          "sub_topics": [
            "[Long-tail query / Specific concept - Stage: (Awareness/Comparison/Transactional)]", 
            "..." 
          ] 
        }}
      ]
    }}
    """
    return extract_json(safe_generate_content(unimodel, prompt))

# 12. The SEO Metadata Generator (Using Specialist Model)
def generate_seo_metadata(article_html, topic):
    """
    Uses the UnifiedModel adapter to route this specific task to Anthropic (Claude).
    """
    print(f"Tool: Delegating SEO Metadata to Specialist Model (Anthropic)...")
    
    # TITLE PERSISTENCE: Try to extract a title from the HTML <h1> or the topic
    title_hint = ""
    h1_match = re.search(r'<h1[^>]*>(.*?)</h1>', article_html, re.IGNORECASE)
    if h1_match:
        title_hint = h1_match.group(1).strip()
    elif len(topic) > 15 and "publish" not in topic.lower() and "repurpose" not in topic.lower():
        title_hint = topic
        
    prompt = f"""
    You are a World-Class SEO Strategist and Copywriter.
    
    INPUT CONTEXT:
    Topic: "{topic}"
    Title Hint (Priority): "{title_hint}"
    Article Content (HTML):
    {article_html[:15000]} # Truncate to avoid context limits
    
    TASK:
    Generate highly optimized, click-worthy metadata for this article.
    
    REQUIREMENTS:
    1. **title**: The primary H1 title. If a clear, substantive title exists in the provided HTML (<h1>) or the 'Topic', PRESERVE IT EXACTLY. Do not rewrite if the input is high-quality.
    2. **meta_title**: The SEO Title tag (Max 60 chars). Try to align this with the main title but ensure the primary keyword is at the front.
    3. **meta_description**: A compelling summary for SERPs (Max 155 chars). Must include a call-to-action or hook.
    4. **custom_excerpt**: A social-media friendly summary (Max 200 chars).
    5. **featured_image_prompt**: A detailed Midjourney/DALL-E prompt. 
       - Style: Cinematic, Photorealistic, High-Tech Office, Warm Lighting. 
       - Subject: Abstract representation of the topic.
    6. **tags**: Not-more-than two relevant tag, selected from list below:
         - AI-Era Search
         - AI Ethics Dialogues
         - Club 10
         - The Builder's Journey
         - The Leading Edge
         - Workflow Architecture

    OUTPUT FORMAT:
    Return ONLY a raw JSON object. Do not use Markdown blocks.
    CRITICAL: You MUST escape any double quotes inside string values (e.g., "title": "The \"Big\" Deal").
    Example:
    {{
        "title": "A Title with \"Quotes\"",
        "meta_title": "SEO Title",
        "meta_description": "Description...",
        "custom_excerpt": "Social summary...",
        "featured_image_prompt": "Prompt...",
        "tags": ["AI-Era Search"]
    }}
    """
    
    try:
        # 1. Use the Global Specialist Brain (Anthropic)
        
        # 2. Generate Content using the Universal Method
        # The adapter returns a response object where .text is the content
        content = safe_generate_content(specialist_model, prompt, generation_config={"temperature": 0.7})
        
        # 3. Use the Unified Super-Listener
        final_json = extract_json(content)
        if final_json:
            return final_json
        
        raise ValueError("Unified parser failed to extract valid metadata.")

    except Exception as e:
        print(f"⚠️ Specialist Anthropic Model Failed: {e}")
        
        # 4. Improved Fallback Logic
        fallback_title = topic.replace("Draft a pSEO article about ", "").replace("Publish as pSEO: ", "").strip('"')
        
        # Try to find an H1 in the article HTML if available
        if article_html:
            h1_match = re.search(r'<h1[^>]*>(.*?)</h1>', article_html, re.IGNORECASE)
            if h1_match:
                fallback_title = h1_match.group(1).strip()
                print(f"  + Fallback recovered title from H1: {fallback_title}")

        return {
            "title": fallback_title,
            "meta_title": fallback_title[:60],
            "meta_description": f"Deep dive into {fallback_title}.",
            "custom_excerpt": f"Analysis of {fallback_title}.",
            "featured_image_prompt": "Abstract digital landscape representing AI ethics.",
            "tags": ["The Leading Edge"]
        }

def ensure_slack_compatibility(text):
    """
    Ensures text is formatted correctly for Slack:
    1. Enforces double newlines for paragraphs.
    2. Converts Markdown bold (**) to Slack bold (*).
    
    Skips code blocks to prevent breaking syntax.
    """
    if not text: return ""
    lines = text.split('\n')
    new_lines = []
    in_code_block = False
    
    for i, line in enumerate(lines):
        # Toggle code block state - RESTORED TO STABLE START-ONLY LOGIC
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            new_lines.append(line)
            continue
            
        if in_code_block:
            new_lines.append(line)
            continue
            
        # Detect clear paragraph breaks (non-list, non-header)
        is_list_item = line.strip().startswith(('-', '*', '1.', '•'))
        is_header = line.strip().startswith('#')
        is_empty = not line.strip()
        
        # FIX BOLDING: Convert **text** to *text* only if not in code block
        # Use regex to replace double asterisks with single, preserving inner content
        processed_line = re.sub(r'\*\*(.+?)\*\*', r'*\1*', line)
        
        new_lines.append(processed_line)
        
        # Add extra newline if it's a paragraph end and next line isn't empty
        if not is_empty and not is_list_item and not is_header and i < len(lines) - 1:
            next_line = lines[i+1]
            if next_line.strip() and not next_line.strip().startswith(('-', '*', '1.', '•')):
                new_lines.append("")  

    return "\n".join(new_lines)





# 13a. The Natural Answer Generator (Conversational & Fluid)
def generate_natural_answer(topic, context, history=""):
    """
    Handles SIMPLE_QUESTION and DIRECT_ANSWER with native intelligence.
    Bypasses the heavy 'Strategist' persona for a more natural flow.
    """
    global unimodel, research_model
    print(f"DEBUG: generate_natural_answer (Native) starting... [Topic: {topic[:50]}]")
    
    # --- ROUTING LOGIC: Context Size Check ---
    # Smart Switch: If context is massive (Likely Research/Code), use Gemini Flash (Research Model)
    # If context is small/conversational, use GPT-5 (Unimodel) for nuance.
    total_context_size = len(str(context)) + len(str(history))
    
    # FIX: Check BOTH context and history for the [CODE_ANALYSIS] tag
    has_code = "[CODE_ANALYSIS]" in str(context) or "[CODE_ANALYSIS]" in str(history)
    use_research_model = total_context_size > 30000 or has_code
    
    active_model = research_model if (use_research_model and research_model) else unimodel
    model_name = "Research Model (Flash)" if active_model == research_model else "Unimodel (GPT-5)"
    print(f"DEBUG: Routing Natural Answer to [{model_name}] | Context Size: {total_context_size} chars")

    prompt = f"""
    You are a helpful, knowledgeable AI assistant.
    
    ### CONVERSATION HISTORY:
    {history}
    
    ### GROUNDING DATA:
    {context}
    
    ### USER INSTRUCTION:
    {topic}
    
    TASK:
    Answer naturally and conversationally. 
    Use the GROUNDING DATA to ensure accuracy.
    
    ### MULTI-NICHE AUDIT PROTOCOL:
    1. **Signal Sifting**: Carefully examine the entire GROUNDING DATA for signals matching the user's specific sub-topic (e.g., 'Politics', 'Education', 'Law').
    2. **Avoid "Sports Blindness"**: High-volume categories like 'Sports' often dominate trend feeds. You must look past global volume to find signals that relate to the user's specific query. 
    3. **Nuance**: If the grounding data contains niche items like 'Iyabo Obasanjo' or 'NELFUND', prioritize explaining their context over simply reporting high-volume sports scores.
    
    ### FORMATTING RULES:
    1. Use **bold** for key concepts.
    2. Ensure paragraphs are separated by blank lines.
    3. **Code Blocks (CRITICAL)**: 
       - For any code snippets, use markdown fenced code blocks: ```language ... ```
       - **NEVER** include normal explanatory text INSIDE a code block.
       - **ALWAYS** close a code block immediately after the code snippet ends.
       - If you have multiple snippets, use separate code blocks for each.
       - Supported languages: javascript, python, java, go, rust, typescript, html, css, json, yaml, bash, sql, etc.
    """
    
    # Use Safe Gen Wrapper
    text = safe_generate_content(active_model, prompt, generation_config={"temperature": 0.4})
    
    # Intent detection for metadata consistency
    # STABILITY: Remove auto-repair here; let ensure_slack_compatibility handle line-by-line
    final_text = ensure_slack_compatibility(text.strip())
    intent = "SIMPLE_QUESTION"
    if "```" in final_text: intent = "TECHNICAL_EXPLANATION"
    
    return {
        "text": final_text,
        "intent": intent,
        "directive": ""
    }


# 13b. The Comprehensive Answer Generator (AEO-AWARE CONTENT STRATEGIST)
def generate_comprehensive_answer(topic, context, history="", intent_metadata=None, context_topic=""):
    """
    Standardizes the logic for generating a direct answer.
    Uses 'detect_research_intent' signal to adjust formatting (Tables/Lists).
    """
    global model, unimodel, research_model
    print(f"DEBUG: generate_comprehensive_answer starting... [Topic: {topic[:50]}]")
    
    # --- ROUTING LOGIC: Context Size Check ---
    total_context_size = len(str(context)) + len(str(history))
    
    # FIX: Check BOTH context and history for the [CODE_ANALYSIS] tag
    has_code = "[CODE_ANALYSIS]" in str(context) or "[CODE_ANALYSIS]" in str(history)
    use_research_model = total_context_size > 30000 or has_code
    
    active_model = research_model if (use_research_model and research_model) else unimodel
    model_name = "Research Model (Flash)" if active_model == research_model else "Unimodel (GPT-5)"
    print(f"DEBUG: Routing Comprehensive Answer to [{model_name}] | Context Size: {total_context_size} chars")
    
    # NEW: Context check includes history if researchers marked it as sufficient
    context_str = str(context)
    is_grounded = "GROUNDING_CONTENT" in context_str or "IN-CONTEXT HISTORY" in context_str
    has_visuals = "VISUAL_INSIGHT" in context_str
    
    # 1. PERSONA & AUDIENCE
    audience_context = detect_audience_context(history)
    persona_instruction = f"""
    You are a Senior Content Strategist. 
    AUDIENCE: {audience_context}
    """

    # 2. INTENT DETECTION
    try:
        signal_data = json.loads(intent_metadata) if intent_metadata else {}
        research_intent = signal_data.get("intent", "FORMAT_GENERAL")
        formatting_directive = signal_data.get("directive", "")
    except:
        research_intent = "FORMAT_GENERAL"
        formatting_directive = ""

    # 3. FORMATTING SENTINEL (Hard-coded prioritization)
    intent_instruction = ""
    target_temp = 0.3 # Lower temperature for better formatting adherence
    
    if research_intent in ["TIMELINE", "FORMAT_TIMELINE"]:
        intent_instruction = "CRITICAL: Output a structured TIMELINE table."
    elif research_intent in ["TABLE", "FORMAT_TABLE"]:
        intent_instruction = "CRITICAL: Output a Markdown TABLE."
    elif research_intent in ["LISTICLE", "FORMAT_LISTICLE", "OUTLINE"]:
        intent_instruction = f"CRITICAL: Output a DETAILED OUTLINE/LISTICLE. Use headings and bullet points. {formatting_directive}"
    elif formatting_directive:
        intent_instruction = f"FORMATTING DIRECTIVE: {formatting_directive}"

    # 4. DYNAMIC PROMPT ASSEMBLY
    prompt = f"""
    {persona_instruction}
    
    ### CONTEXT & MISSION:
    The user is engaged in a thread about: "{context_topic}"
    (Audience is already defined in persona above)
    
    ### LATEST INSTRUCTION (PRIORITY):
    "{topic}"
    
    ### GROUNDING DATA:
    {context}
    
    ### CONVERSATION HISTORY:
    {history}
    
    ### FORMATTING GUIDANCE:
    {intent_instruction}
    
    ### CRITICAL RULES:
    1. Generate a response that directly addresses the LATEST INSTRUCTION while using the GROUNDING DATA as supporting evidence.
    2. **Code Blocks**: For any code snippets, use markdown fenced code blocks:
       - Use triple backticks with language identifier: ```language
       - Example: ```python\ndef hello(): return \"world\"\n```
       - Always specify the language for proper syntax highlighting.
    """

    # Use active_model (Dynamic routing) for the final synthesis
    text = safe_generate_content(active_model, prompt, generation_config={"temperature": target_temp})
    
    # Enforce spacing on the output
    final_text = ensure_slack_compatibility(text.strip())
    
    return {
        "text": final_text,
        "intent": research_intent,
        "directive": formatting_directive
    }

# 14. The Dedicated pSEO Article Generator
def generate_pseo_article(topic, context, history="", history_events=None, is_initial_post=True):
    global unimodel
    print(f"Tool: Generating pSEO Article for '{topic}'")
    
    # AUDIENCE DETECTION: Search history for tone instructions
    audience_context = detect_audience_context(history)

    # 1. Determine System Instruction (Grounding Logic)
    is_grounded = any("GROUNDING_CONTENT" in str(c) for c in context)
    
    # 2. History Analysis (Tri-State Logic: REFACTOR vs EXPAND vs AUTHOR)
    # State A: REFACTOR (Repurpose existing full draft) -> High Strictness, No Synthesis
    # State B: EXPAND (Synthesize from Outline) -> High Synthesis, Strict Structure
    # State C: AUTHOR (Greenfield) -> High Synthesis, Free Structure
    
    start_mode = "AUTHOR"
    
    # Signals
    topic_lower = topic.lower()
    has_pseo_keyword = "pseo" in topic_lower
    
    # Path A Signals (Verbatim Finalization)
    is_repurpose_cmd = any(kw in topic_lower for kw in ["dump", "repurpose", "publish the", "provision the", "for the draft", "finalize the"])
    
    # Path B Signals (Creative Iteration)
    is_expand_cmd = any(kw in topic_lower for kw in ["expand", "flesh out", "write from", "based on outline", "refine", "polish", "rewrite"])
    
    has_outline_structure = "## " in str(history) and len(str(history)) < 3000 # Rough heuristic: Outlines are shorter
    has_full_draft_structure = "## " in str(history) and len(str(history)) > 3000
    
    # FUTURE-PROOFING: If follow-up, insist on "pseo" keyword for Path A/B to avoid accidental refactors of general feedback
    if not is_initial_post and not has_pseo_keyword:
        print("  + Follow-up turn without 'pSEO' keyword. Defaulting to AUTHOR mode for safety.")
        start_mode = "AUTHOR"
    elif is_repurpose_cmd and has_full_draft_structure:
        start_mode = "REFACTOR"
    elif (is_expand_cmd and "## " in str(history)) or (has_outline_structure and not is_repurpose_cmd):
        start_mode = "EXPAND"
    elif is_repurpose_cmd and has_outline_structure:
        # User said "Repurpose this outline" -> They likely mean "Expand this outline into an article"
        start_mode = "EXPAND" 
        
    print(f"  + Generation Mode Selected: {start_mode}")

    # --- FAST-TRACK: DIRECT PROVISIONING (Bypass LLM) ---
    # Case: "Publish this draft" - Extract the existing HTML from history_events.
    best_source_text = None
    if is_repurpose_cmd and history_events:
        for event in reversed(history_events):
            # 1. Look for recent agent_proposal (pseo_article) with clean HTML
            proposal_data = event.get('proposal_data')
            if isinstance(proposal_data, dict) and proposal_data.get('article_html'):
                print(f"  + FAST-TRACK: Found clean HTML in proposal_data. Provisioning directly.")
                return proposal_data['article_html']
            
            # 2. Look for existing text (Markdown) to REFACTOR
            content = event.get('text') or event.get('content')
            if content and isinstance(content, str) and len(content) > 500:
                # HARDENING: Reject Fast-Track if it contains raw markdown artifacts
                has_markdown_artifacts = "```" in content or "\n- " in content or "\n* " in content
                
                if "<section" in content and "</section>" in content and not has_markdown_artifacts:
                    print(f"  + FAST-TRACK: Found existing high-fidelity HTML draft. Provisioning directly.")
                    return content
            
                # Otherwise, this is our best source for the Refactor Engine
                best_source_text = content
                print(f"  + FAST-TRACK: Found potential source text ({len(content)} chars) for Refactoring.")
                break
        
        if best_source_text:
            print("  + FAST-TRACK: Shifting to REFACTOR mode using found source text.")
            start_mode = "REFACTOR"
        else:
            print("  + FAST-TRACK: No suitable source found. Falling back to EXPAND/AUTHOR.")

    if start_mode == "REFACTOR":
        # --- PATH A: STRICT REFACTOR (Specialist Model) ---
        print("  + Executing PATH A: REFACTOR with Specialist Model (Claude Sonnet 4.5)")
        
        # Use best_source_text if found, otherwise the whole history (risky but fallback)
        source_to_refactor = best_source_text or history
        
        refactor_prompt = f"""
        You are a Content Refactor Engine specializing in high-fidelity Ghost CMS delivery.
        
        TARGET AUDIENCE: {audience_context}
        
        TASK: Convert the 'SOURCE TEXT' into target GHOST-FRIENDLY HTML format.
        
        STRUCTURE REQUIREMENTS (Strict HTML):
        1.  **Semantic Structuring**: Use <section> wrappers with appropriate classes (e.g., class="intro", class="body", class="deep-dive", class="methodology").
        2.  **Semantic Headers**: Use <h1> for the main title, <h2> for major sections, and <h3> for sub-sections.
        3.  **NO NUMBERED HEADINGS**: Do NOT include numbers (e.g., "1. ", "2. ") in your <h2> or <h3> headers.
        4.  **Code Blocks**: Use semantic HTML: `<pre><code class="language-...">...</code></pre>`.
        5.  **Lists**: Use semantic `<ul><li>` or `<ol><li>`.
        6.  **Paragraphs**: Wrap all text in <p> tags.
        
        CRITICAL CONTENT RULES:
        - **Semantic Integrity**: Ensure the article flows logically and follows the structure of the source text.
        - **Acronym Protocol**: Define all acronyms in parentheses on the first use.
        
        CRITICAL FORMATTING RULES:
        -   **NO MARKDOWN**: Absolutely NO markdown backticks (```), asterisks for bold (**), or hyphens for lists (- or *).
        -   **CLEAN CONTENT**: Remove any internal instructions, metalabels, or architect-facing strings (e.g., "writing_instruction", "blueprint", "architect notes") found in the source text.
        -   **NO PREAMBLE**: Start directly with the first HTML tag.
        
        SOURCE TEXT:
        {source_to_refactor}
        """
        # Use Specialist (Claude Sonnet 4.5) for high-fidelity refactoring
        raw_html = safe_generate_content(specialist_model, refactor_prompt, generation_config={"temperature": 0.2})
        
        # FIX: Strip Conversational Filler (The "Okay, here is..." preamble)
        # We look for the first occurrence of <h... or <section... and the last occurrence of a major closing tag
        match = re.search(r'(<(?:h1|section|article)[\s\S]*</(?:section|article|h1|p)>)', raw_html, re.IGNORECASE)
        if match:
            print("  + Sanitized pSEO Output: Removed preamble/postscript.")
            return match.group(1)
            
        return raw_html # Fallback if regex fails (likely clean enough)

    # --- PATH B & C: EXPAND / AUTHOR (Creative Synthesis) ---
    elif start_mode == "EXPAND":
        # EXPANSION PROMPT: Use history as a skeleton.
        print("  + Executing PATH B: EXPAND with Unimodel")
        system_instruction = f"""
        You are a Specialized Technical Writer.
        TASK: Write a comprehensive pSEO article based on the provided OUTLINE/BLUEPRINT.
        
        RULES:
        1.  **FOLLOW STRUCTURE**: Strictly follow the headers defined in the 'BLUEPRINT'.
        2.  **SYNTHESIZE CONTENT**: Flesh out each bullet point into full, detailed paragraphs.
        3.  **USE RESEARCH**: Use the provided Research Context to fill in facts/data.
        
        BLUEPRINT/OUTLINE:
        {history}
        """
        tone_instruction = "Tone: High-authority, professional, and detailed. Match the depth implied by the blueprint."
        context_block = f"Research Context:\n{context}"
        
    else: # AUTHOR MODE
        print("  + Executing PATH C: AUTHOR with Unimodel")
        if is_grounded:
            system_instruction = "You are a Clear Technical Communicator. Base the article PRIMARILY on the provided 'Research Context'."
        else:
            system_instruction = "You are a Clear Technical Communicator. Use the provided context to write a high-authority article."
        tone_instruction = "Tone: Clear, engaging, professional English. Use analogies to explain complex ideas."
        context_block = f"Research Context:\n{context}\n\nConversation History:\n{history}"

    # Shared Prompt for B & C (Creative Paths)
    prompt = f"""
    {system_instruction}
    
    AUDIENCE: 
    {audience_context}
    
    TONE & STYLE:
    {tone_instruction}
    - **SIMPLICITY PROTOCOL**: Use simple, accessible terminology. Explain complex concepts using plain English analogies. Avoid academic or overly dense jargon.
    
    STRUCTURE REQUIREMENTS (Strict HTML):
    1.  **<section class="intro">**: A compelling hook. **MEAT-FIRST**: Deliver core findings or high-density technical insights immediately in the first paragraph.
    2.  **<section class="body">**: Detailed analysis using <h2> and <h3> tags.
    3.  **NO NUMBERED HEADINGS**: Do NOT include numbers (e.g., "1. ", "2. ") in your <h2> or <h3> headers.
    4.  **Code Blocks**: Use semantic HTML: `<pre><code class="language-...">...</code></pre>`.
    5.  **Lists**: Use semantic `<ul><li>` or `<ol><li>`.
    6.  **<section class="methodology">**: A TECHNICAL TRANSPARENCY FOOTER. 
    
    CRITICAL CONTENT RULES:
    - **Contextual Specificity**: Actively identify and expand on points of interest, debates, cultural markers, or specific real-world examples found in the research context.
    - **Technical-External Mapping**: Identify relevant external frameworks (regulatory, structural, or conceptual) and explicitly link technical solutions to those specific anchors.
    - **Technical Accuracy**: Ensure all technical claims are supported by the provided context.
    - **Acronym Protocol**: Define all acronyms in parentheses on the first use (e.g. "Electronic Data Interchange (EDI)").
    
    CRITICAL RULES:
    - Do NOT hallucinate.
    - Return ONLY valid semantic HTML content. Absolutely NO MARKDOWN.
    - Always specify the programming language in the code class.
    
    CONTEXTUAL DATA:
    Current Topic: "{topic}"
    
    {context_block}
    
    Article Draft:
    """
    
    return safe_generate_content(unimodel, prompt, generation_config={"temperature": 0.4})

# 14.5 The Recursive Deep-Dive Generator (Dynamic Room-by-Room Construction)
def generate_deep_dive_article(topic, context, history="", history_events=None, target_length=1500, target_geo="Global"):
    global unimodel
    print(f"Tool: Initiating Recursive Deep-Dive for '{topic}' (Region: {target_geo}, Target: {target_length} words)")
    
    # CLAMP: Prevent unbounded generation that crashes SSL/Webhooks
    target_length = min(3500, max(500, target_length))
    
    is_grounded = any("GROUNDING_CONTENT" in str(c) for c in context)
    audience_context = detect_audience_context(history)
    
    # PHASE 1: Generate the Blueprint & Title (The Architect)
    blueprint_prompt = f"""
    You are the Lead Architect. Create a strategic content blueprint for: "{topic}".
    Target Length: ~{target_length} words.
    
    ### STEP 1: HISTORY CHECK (CONSENSUS)
    Check the `HISTORY` below. 
    - **IF** the user provided a **specific Title** or **Outline** (e.g. "Title: ..."), YOU MUST USE IT. Do NOT invent a new title if one is given.
    - **IF** a specific Outline structure exists, ADHERE TO IT.
    - **OTHERWISE (Default)**: Design an optimal structure from scratch based on the topic, collected context, and conversation history.
    
    ### STEP 2: STYLE ANALYSIS
    Analyze the user's request ("{topic}") for specific tone/style instructions (e.g., "sarcastic", "storytelling", "academic", "debate").
    - **IF DETECTED**: Apply that style to the writing instructions below.
    - **IF SILENT**: Use the DEFAULT STYLE: "Clear, engaging, professional English. Use analogies to explain complex ideas, but maintain technical accuracy."
    
    ### STEP 3: BLUEPRINT GENERATION
    Design the optimal structure.
    - **Sections**: Create between 3 and 8 H2 sections depending on the complexity of the topic.
    - **NO REDUNDANCY**: Do NOT include a "Lede", "Introduction", or "Conclusion" section in your list. These parts are handled by special drafting phases. Start directly with the core problem or first technical insight.
    - **WORD COUNT BUDGETING**: You MUST assign a `target_word_count` to each section. The SUM of all these counts MUST be approximately {target_length - 300} (to allow space for the intro and conclusion). 
    - **Strategic Allocation**: Allocate more words to complex case studies/ROOT CAUSE analysis and fewer to simple definitions/hooks.
    - **Contextual Root Cause Analysis**: Mandate exploration of the systemic factors behind the topic's core challenges in the architecture.
    - **Solution Framing**: Explicitly frame provided technical concepts or solutions as context-sensitive measures.
    - **Relevant Technical States**: Highlight specific system states (e.g., approval, failure, transitions) that are most critical to the current context.
    - **Flow**: Ensure a logical narrative arc.
    
    AUDIENCE: {audience_context}
    RESEARCH CONTEXT: {context}
    HISTORY: {history}
    
    OUTPUT: Return ONLY a JSON object:
    {{
        "title": "A Compelling Title (Or the Agreed Title)",
        "sections": [ 
            {{
                "title": "Section Title", 
                "writing_instruction": "Specific instruction for the writer (e.g., 'Tell a story about X', 'List 5 facts about Y'). Ensure it aligns with the detected style.",
                "target_word_count": "Optional. Number of words for this section (e.g. 200, 500). If omitted, the system will auto-allocate."
            }} 
        ]
    }}
    """
    
    try:
        blueprint_raw = safe_generate_content(unimodel, blueprint_prompt)
        if "```json" in blueprint_raw: blueprint_raw = blueprint_raw.split("```json")[1].split("```")[0].strip()
        blueprint = json.loads(blueprint_raw)
        title = blueprint.get("title", f"The Analysis of {topic}")
        sections = blueprint.get("sections", [])
    except Exception as e:
        print(f"Blueprint Error: {e}")
        # Fallback to pSEO if blueprinting fails
        return generate_pseo_article(topic, context, history, history_events=history_events, is_initial_post=True)

    # PHASE 2: Recursive Room Building (The Writer - PARALLELIZED)
    article_parts = [f"<h1>{title}</h1>"]
    
    # Intro (Sequential - Sets the stage)
    print(f"  + Drafting Hook Intro for {target_geo}...")
    intro_words = max(150, target_length // (len(sections) + 1))
    intro_prompt = f"""
    Write a compelling {intro_words}-word intro for: '{title}'.
    REGION: {target_geo}
    
    AUDIENCE: {audience_context}
    STYLE: Clear, engaging, professional English. Use analogies for complex ideas.
    GOAL: {topic}
    
    OUTPUT: Return ONLY the content in semantic HTML (no markdown). Wrap paragraphs in <p>. Do NOT include <h1> or <h2> headers here.
    """
    article_parts.append(f'<section class="intro">\n{safe_generate_content(unimodel, intro_prompt)}\n</section>')
    
    # Define Worker Function for Parallel Processing
    def generate_section(index, section, total_sections, previous_title):
        print(f"  + [THREAD {index+1}] Drafting Room: {section['title']} (Region: {target_geo})...")
        
        instruction = section.get('writing_instruction', 'Explain this concept clearly.')
        
        # Determine dynamic word count target
        section_words = section.get('target_word_count')
        if not section_words or not str(section_words).isdigit():
            # Heuristic fallback: Divide remaining length among sections
            remaining_length = target_length - intro_words
            section_words = remaining_length // total_sections
        else:
            section_words = int(str(section_words).strip())
            
        # NOTE: In parallel mode, we rely on the Blueprint logic rather than the literal previous text.
        room_prompt = f"""
        Write a {section_words}-word section for: "{title}".
        REGION: {target_geo}
        
        CHAPTER {index+1}/{total_sections}: {section['title']}
        INSTRUCTION: {instruction}
        CONTEXT: This section strictly follows the concept: "{previous_title}". Ensure logical continuity.
        
        STRICT LIMIT: Stay under {section_words + 50} words.
        **MEAT-FIRST PROTOCOL**: Deliver core findings or high-density technical/systemic insights immediately in the first paragraph.
        **SIMPLICITY MANDATE**: Use simple terms and descriptions. Avoid dense jargon; explain technical concepts using analogies that a non-specialist can grasp.
        **TECHNICAL-EXTERNAL MAPPING**: Instruct the model to identify relevant external frameworks (regulatory, structural, or conceptual) and explicitly link technical solutions to those specific anchors.
        **CONTEXTUAL SPECIFICITY**: Actively identify and expand on points of interest, debates, cultural markers, or specific real-world examples found in the research context.
        
        GROUNDING DATA: {context if is_grounded else "Internal Knowledge base"}
        
        OUTPUT: Start with <h2>{section['title']}</h2> then the content in semantic HTML (no markdown). Wrap paragraphs in <p>, use <ul>/<li> for lists, and `<pre><code class="language-...">` for code.
        """
        content = safe_generate_content(unimodel, room_prompt)
        return index, f'<section class="body-part">\n{content}\n</section>'

    # Execute Parallel Pool
    print(f"  + Launching Parallel Construction Crew ({len(sections)} sections)...")
    body_parts = [None] * len(sections) # Placeholder array
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_index = {}
        for i, section in enumerate(sections):
            # Determine "Previous Title" for context simulation
            prev_title = "Introduction" if i == 0 else sections[i-1]['title']
            future = executor.submit(generate_section, i, section, len(sections), prev_title)
            future_to_index[future] = i
            
        for future in concurrent.futures.as_completed(future_to_index):
            idx, content = future.result()
            body_parts[idx] = content
            print(f"  + [THREAD {idx+1}] Room Complete.")
            
    # Assembly
    article_parts.extend(body_parts)

    # Conclusion
    print("  + Drafting Final Reflection...")
    conc_words = intro_words // 2
    conc_prompt = f"""
    Write a {conc_words}-word concluding thought for '{title}'. 
    Audience: {audience_context}. 
    Summary of key impact.
    
    OUTPUT: Return ONLY the content in semantic HTML (no markdown). Wrap paragraphs in <p>. Do NOT include a header.
    """
    article_parts.append(f'<section class="conclusion">\n<h2>Final Reflection</h2>\n{safe_generate_content(unimodel, conc_prompt)}\n</section>')

    # PHASE 3: Methodology
    print("  + Finalizing Methodology & Transparency...")
    if is_grounded:
        # STRIP DEBRIS: Clean external metadata from grounding context before listing sources
        found_urls = list(set(re.findall(r'https?://[^\s<>"]+', str(context))))
        # Filter for real article links, avoid ghost-api/image/internal artifacts
        clean_urls = [u for u in found_urls if not any(x in u for x in ["ghost-api", "image", "google-adk", "localhost"])]
        source_text = ", ".join(clean_urls[:5]) if clean_urls else "Verified research sources."
    else:
        source_text = "AEO Synthesis from Knowledge Base."

    # Calculate interim word count for the footer
    current_content = " ".join(article_parts)
    est_words = len(current_content.split())

    methodology_text = f"""
    <section class="methodology">
    <h2>Methodology & Transparency</h2>
    This {est_words}-word analysis was recursively architected.
    Sources: {source_text}
    </section>
    """
    article_parts.append(methodology_text)

    full_html = "\n\n".join(article_parts)
    print(f"  -> Deep-Dive Complete. Est words: {len(full_html.split())}")
    return full_html

#15. The Euphemistic 'Then vs Now' Linker
def create_euphemistic_links(keyword_context, is_initial=True):
    global unimodel
    
    state_context = "This is a NEW proposal for a new thread." if is_initial else "This is a REVISION or EXTENSION of a previous strategy in an existing thread."
    
    prompt = f"""
    Topic: "{keyword_context['clean_topic']}". 
    
    STATE: {state_context}
    CONTEXT: {keyword_context['context']}
    
    Identify 4-10 core keyword clusters for 'Then' and 6-10 for 'Now'.
    
    ### STRATEGIC PRINCIPLES (Euphemistic Contrast):
    1. **Historical Trajectory:** Contrast the 'Static/Manual' past with the 'Dynamic/AI-Driven' future.
    2. **Value Shift:** Highlight how the user's role evolves from 'Execution' to 'Architecture'.
    3. **Semantic Anchoring:** Ensure the links between Then and Now are logically sound and establish immediate authority.
    
    ### GUIDELINES:
    1. If this is a REVISION (is_initial=False), you MAY start with a header like ":robot_face: The proposal has been REVISED" or similar context-aware greetings if appropriate.
    2. If this is an INITIAL proposal, be structured and authoritative.
    3. Ensure the output is strictly valid JSON for extraction.

    CRITICAL SCHEMA: Exact keys: "then_concept", "now_concept", "link".
    Structure: {{ "interlinked_concepts": [ {{ "then_concept": "...", "now_concept": "...", "link": "..." }} ] }}
    """
    return extract_json(safe_generate_content(unimodel, prompt))

#16. The Proposal Critic and Refiner
def critique_proposal(topic, current_proposal):
    global unimodel
    prompt = f"Review proposal for '{topic}': {json.dumps(current_proposal, indent=2)}. If excellent, respond: APPROVED. Else, provide concise feedback to improve 'Then vs Now' contrast."
    return safe_generate_content(unimodel, prompt)

#17. The Proposal Refiner
def refine_proposal(topic, current_proposal, critique):
    global unimodel
    prompt = f"""
    REWRITE proposal. Topic: {topic}. Draft: {json.dumps(current_proposal)}. Critique: {critique}. 
    Preserve keys: 'then_concept', 'now_concept', 'link'. Ensure JSON format.
    """
    return extract_json(safe_generate_content(unimodel, prompt))

# 18. Phase 1: Sales-to-Content Pipeline
def process_sales_transcript(transcript_text, session_id=None):
    """
    Extracts customer objections and generates solution brief.
    """
    global specialist_model, db
    
    # Step 1: Extract objections using Claude
    objection_prompt = f"""
    Analyze this sales call transcript and extract the following (omit any field if not mentioned):
    1. Core customer objections (max 5)
    2. Pain points mentioned
    3. Competitor comparisons
    4. Budget/timeline concerns
    
    Transcript:
    {transcript_text}
    
    Format as JSON (omit fields if not present in transcript):
    {{
      "objections": ["objection 1", "objection 2"],
      "pain_points": ["pain 1", "pain 2"],
      "competitors": ["competitor 1"],
      "budget_timeline": "summary"
    }}
    
    CRITICAL: Return ONLY valid JSON. If a category has no data, use an empty array [] or "Not discussed".
    """
    
    try:
        objections_raw = safe_generate_content(specialist_model, objection_prompt)
        # Handle potential markdown wrapping
        if "```json" in objections_raw:
            objections_raw = objections_raw.split("```json")[1].split("```")[0].strip()
        elif "```" in objections_raw:
            objections_raw = objections_raw.split("```")[1].split("```")[0].strip()
            
        objections = json.loads(objections_raw)
    except Exception as e:
        print(f"Error extracting objections: {e}")
        objections = {}
    
    # Normalize with defaults (defensive programming)
    objections_list = objections.get("objections", [])
    pain_points = objections.get("pain_points", [])
    competitors = objections.get("competitors", [])
    budget_timeline = objections.get("budget_timeline", "Not discussed")
    
    # Step 2: Generate solution brief
    brief_prompt = f"""
    Create a "Solution Brief" that addresses these customer insights:
    
    Objections: {objections_list if objections_list else "None identified"}
    Pain Points: {pain_points if pain_points else "None identified"}
    Competitors: {competitors if competitors else "None mentioned"}
    Budget/Timeline: {budget_timeline}
    
    Format:
    # Solution Brief
    ## Executive Summary
    ## Objection Responses
    ## Competitive Advantages
    ## Implementation Timeline
    ## ROI Projection
    
    ### FORMATTING RULES:
    1. Use **bold** for emphasis.
    2. Use markdown tables where appropriate.
    3. **Code Blocks**: If including technical/code examples, use markdown fenced code blocks:
       - Use triple backticks with language identifier: ```language
    """
    
    
    brief_raw = safe_generate_content(specialist_model, brief_prompt)
    brief_html = ensure_slack_compatibility(brief_raw)  # Convert **bold** to *bold*
    
    # Step 3: Normalize content for return (Auto-save REMOVED to align with approval logic)
    normalized_objections = {
        'objections': objections_list,
        'pain_points': pain_points,
        'competitors': competitors,
        'budget_timeline': budget_timeline
    }
    
    return {
        'objections': normalized_objections,
        'brief': brief_html
    }

# --- THE STATEFUL MAIN WORKER FUNCTION (FINAL V6 - SOCIAL AWARE) ---
@functions_framework.http
def process_story_logic(request):
    global unimodel, flash_model, research_model, specialist_model, db

    # 0. ENTRY TELEMETRY (Observability)
    now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[MISSION START] {now_str} | Function: process_story_logic")

    # 0.5 CORE SERVICES (Pre-flight init)
    if any(m is None for m in [unimodel, flash_model, research_model, specialist_model]):
        vertexai.init(project=PROJECT_ID, location=LOCATION)

    if unimodel is None:
        unimodel = UnifiedModel(MODEL_PROVIDER, MODEL_NAME)
    
    if flash_model is None:
        flash_model = GenerativeModel(FLASH_MODEL_NAME, safety_settings=safety_settings)

    if research_model is None:
        research_model = UnifiedModel("vertex_ai", RESEARCH_MODEL_NAME)
        
    if specialist_model is None:
        # STRICT: User enforced Claude 3.5 Sonnet (using 4.5 alias as requested)
        specialist_model = UnifiedModel(provider="anthropic", model_name="claude-sonnet-4-5")
        
    if db is None:
        db = firestore.Client(project=PROJECT_ID)

    req = request.get_json(silent=True)
    if isinstance(req, list): req = req[0] # Add safety for n8n list payloads
    print(f"DEBUG: Story Worker received payload keys: {list(req.keys()) if req else 'None'}")
    
    # --- SYSTEM FILTER: Prevent Infinite Loops ---
    # Case: The message is a confirmation notification (e.g., from N8N/Ghost)
    topic_text = req.get('topic', "").lower() or req.get('feedback_text', "").lower()
    if "created in ghost" in topic_text or "ready for ghost" in topic_text:
        print("🛑 SYSTEM FILTER: Ignoring Ghost/N8N status notification.")
        return jsonify({"msg": "Status notification ignored"}), 200
    
    # --- IDEMPOTENCY CHECK (Prevent Double-Trigger) ---
    # Construct a unique event ID from the Slack Timestamp
    slack_context = req.get('slack_context', {})
    unique_event_id = slack_context.get('ts') or req.get('event_ts') or req.get('client_msg_id')
    
    # ADK FIX: Removed bypass for feedback loops. Slack provides unique 'ts' for every reply, so they MUST be deduced.
    is_feedback_loop = bool(req.get('feedback_text'))
    
    if unique_event_id:
        # Check if we've already seen this event ID in the last 30 minutes
        dedup_ref = db.collection('processed_events').document(str(unique_event_id))
        doc = dedup_ref.get()
        if doc.exists:
            # Check timestamp to allow re-runs after 5 minutes
            data = doc.to_dict()
            last_run = data.get('timestamp')
            
            if last_run:
                cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=5)
                if last_run > cutoff:
                    print(f"🔒 IDEMPOTENCY: Skipping duplicate event {unique_event_id} (Processed < 5m ago)")
                    return "Event already processed", 200
                else:
                    print(f"🔄 IDEMPOTENCY: Reprocessing old event {unique_event_id} (Age > 5m)")
            else:
                 pass 
        
        # Mark as pending immediately
        try:
            dedup_ref.set({
                'timestamp': datetime.datetime.now(datetime.timezone.utc),
                'status': 'processing',
                'session_id': req.get('session_id')
            })
        except Exception as e:
            print(f"⚠️ Warning: Failed to set idempotency lock: {e}")
            
    session_id, original_topic = req['session_id'], req.get('topic', "")
    feedback_text = req.get('feedback_text') # Captured from worker-feedback
    images = req.get('images', [])
    code_files = req.get('code_files', []) # ADD: Extract code_files
    
    # Use the Slack-mimicking logic: if there's no thread_ts AND no feedback_text, it's the root message (initial).
    is_initial_post = not slack_context.get('thread_ts') and not feedback_text
    print(f"DEBUG: Thread State -> is_initial_post: {is_initial_post} (ts: {slack_context.get('ts')}, thread_ts: {slack_context.get('thread_ts')})")
    
    # --- STEP 1: PERCEPTION (Sanitize Input) ---
    sanitized_topic = sanitize_input_text(feedback_text or original_topic)
    
    # --- STEP 2: LOAD SHORT-TERM MEMORY ---
    doc_ref = db.collection('agent_sessions').document(session_id)
    session_doc = doc_ref.get()
    
    session_data = {}
    if session_doc.exists:
        session_data = session_doc.to_dict()
        # SAFETY NET: Recover Slack Context if stripped by n8n (Non-Destructive)
        if session_data.get('slack_context'):
            db_ctx = session_data['slack_context']
            for key in ['channel', 'ts', 'thread_ts']:
                if not slack_context.get(key) and db_ctx.get(key):
                    print(f"INFO: Recovering missing {key} from Firestore...")
                    slack_context[key] = db_ctx[key]

    history_events = []
    history_text = ""
    if session_doc.exists:
        # MEMORY EXPANSION: Removed .limit(20) to ensure deep context recall
        events_ref = doc_ref.collection('events')
        
        # Chronological order (Old -> New)
        query = events_ref.order_by('timestamp', direction=firestore.Query.ASCENDING)
        
        # CLAMPING: Limit history to last 15 turns to prevent context bloat
        # 30 matches (15 turns of User/Agent pairs)
        all_events = [doc.to_dict() for doc in query.stream()]
        history_events = all_events[-30:] if len(all_events) > 30 else all_events


    # RESTORED: Removed the MAX_HISTORY_EVENTS limit to preserve the full conversation thread.
    # The 429 errors are now handled by Exponential Backoff in UnifiedModel.
    recent_events = history_events
    
    # 2. Extract context from history for Triage and Routing
    formatted_history = []
    for event in recent_events:
        etype = event.get('event_type')
        if etype == 'user_request':
            formatted_history.append(f"USER: {event.get('text')}")
        elif etype == 'user_feedback':
            # FIX: Include code analysis in history text if it was part of feedback
            code_context = ""
            if event.get('code_files'):
                # Use .get('name') to handle raw Slack file objects or normalized ones safely
                code_context = f" (Attached Files: {', '.join([f.get('name', f.get('file_name', 'unknown_file')) for f in event['code_files']])})"
            formatted_history.append(f"USER FEEDBACK: {event.get('text')}{code_context}")
        elif etype in ['agent_answer', 'agent_reply', 'agent_proposal']:
            content = event.get('text') or event.get('content') or str(event.get('data', ''))
            formatted_history.append(f"AGENT: {content}") 
    
    history_text = "\n".join(formatted_history)
    
    # --- MONITORING: Context Loading Statistics ---
    print(f"DEBUG: Worker loading {len(history_events)} events for session: {session_id}")
    print(f"DEBUG: Active Grounding assets manually passed: {len(images or [])}")

    # --- Step 2.5: Code Perception (HOISTED) ---
    # We analyze code files BEFORE Triage to ensure "Fast Exit" paths like Structural Formatting
    # still have access to the code context.
    code_analysis_snippets = []
    
    # Initialize MCP Client (Early Init)
    mcp = get_mcp_client()

    # 1. Recover from history
    processed_filenames = set() # To prevent redundant re-analysis
    if history_events:
        recovered_count = 0
        unique_snippets = {} # Deduplication by filename
        for event in history_events:
            if event.get('event_type') == 'code_analysis' and event.get('analysis_result'):
                fname = event.get('file_name', 'unknown')
                unique_snippets[fname] = event['analysis_result']
                processed_filenames.add(fname)
                recovered_count += 1
        
        # CLAMPING: Take only last 5 unique snippets
        sorted_keys = list(unique_snippets.keys())
        for k in sorted_keys[-5:]:
            code_analysis_snippets.append(unique_snippets[k])
            
        if recovered_count > 0:
            print(f"TELEMETRY: Recovered {len(code_analysis_snippets)} unique code insights from history (Clamping: {recovered_count} -> {len(code_analysis_snippets)}).")

    # 2. Analyze New Files
    if code_files:
        print(f"TELEMETRY: Code File Analysis Mode detected ({len(code_files)} files) - Pre-Triage Execution")
        
        for file_info in code_files:
            file_name = file_info.get('name', 'unknown_file')
            
            # ADK FIX: Skip analysis if we already recovered it from history
            if file_name in processed_filenames:
                print(f"⏭️ Skipping re-analysis of {file_name} (Already in history)")
                continue
                
            file_url = file_info.get('url')
            file_id = file_info.get('id')
            file_mode = file_info.get('mode', 'hosted')
            
            # Download file content from Slack
            file_content = fetch_slack_file_content(file_url, file_id, file_mode)
            
            if not file_content:
                print(f"⚠️ No content downloaded for {file_name}")
                code_analysis_snippets.append(f"CODE_FILE ({file_name}): [Download failed]")
                continue
            
            # Analyze code file via MCP tool
            try:
                analysis_context = mcp.call_tool(
                    "analyze_code_file",
                    {
                        "file_content": file_content,
                        "file_name": file_name,
                        "user_prompt": sanitized_topic,
                        "history_context": history_text
                    }
                )
                
                # Prepend special tag for Routing Logic detection
                tagged_context = f"[CODE_ANALYSIS] (File: {file_name})\n{analysis_context}"
                code_analysis_snippets.append(tagged_context)
                
                # Log the analysis event (with result for persistence)
                # Note: We append to new_events later, but we need to track it now
                # We'll just define the event dict here and append it after new_events is init
                # Actually, we can just append to history_text immediately for Triage visibility
                history_text += f"\n[SYSTEM: Analyzed Code File '{file_name}'. Content Summary: {analysis_context[:500]}...]"
                
            except Exception as e:
                print(f"⚠️ Code analysis failed for {file_name}: {e}")
                code_analysis_snippets.append(f"CODE_FILE ({file_name}): [Analysis failed: {str(e)}]")
        
        print(f"✅ Code analysis complete. Adding {len(code_analysis_snippets)} code insights to working context.")

    try:
        # --- STEP 3: TRIAGE (Now includes SOCIAL category) ---
        # Note: We triage BEFORE distilling or researching to save cost/latency.
        
        triage_prompt = f"""
            Analyze the user's latest message in the context of the conversation history.
            Classify the PRIMARY GOAL into one of seven categories.
            
            ### DECISION PRINCIPLES:
            1.  **Fresh Start Rule**: Treat every message as a fresh decision point. Do not assume the goal remains the same as the previous turn just because a specific mode (like pSEO) was active previously.
            2.  **Post-Artifact Default**: If the conversation history shows a major artifact (pSEO Draft, Deep Dive, Topic Cluster) was delivered in the previous turn, the user's next message is usually seeking **feedback, clarification, or minor refinement**. Favor **SIMPLE_QUESTION** or **DIRECT_ANSWER** in these cases.
            3.  **Intent Overlap**: If a request could be multiple categories, pick the one that represents the *simplest* necessary action based on the USER REQUEST text.

            ### HYBRID REQUEST RULE (CRITICAL):
            - If the request contains NEW URLs, Attached Files (Code), or asks to "Research" something new, **DO NOT** select OPERATIONAL_REFORMAT.
            - Even if the user asks for an "Outline" or "Checklist", if there are NEW signals (links/files), select **DIRECT_ANSWER** or **WEB** so the agent performs the necessary grounding first.
            - Select **OPERATIONAL_REFORMAT** *only* if the context is ALREADY fully present in the history or code analysis snippets.

            1.  **SOCIAL_CONVERSATION**: Greetings, small talk, simple feedback, or **casual replies to the agent's questions**.
                *   *Triggers:* "Okay", "Thanks", "Hello", "How are you", or providing a short answer to an agent's conversational prompt (e.g., answering "What's brewing?" with "Heineken").
                *   *Rule:* If the user is just continuing a friendly chat without asking for a draft, research, or a direct answer, pick this.

            2.  **DEEP_DIVE**: **LONG-FORM BODY CONTENT (800-2000+ Words).** Select this ONLY if the user asks for a complete article body or specific word counts.
                *   *CRITICAL:* Select this if the user says "Draft a post using this outline" or "Expand this structure".
                *   *EXCEPTION:* Do NOT select this if the user asks to *CREATE* an outline. Only if they ask to *WRITE* the full text.
                *   *Triggers:* "800 words", "1500 words", "Write the full post", "Draft the entire article", "Expand this point".

            3.  **TOPIC_CLUSTER_PROPOSAL**: **SEMANTIC ARCHITECTURE GENERATION.** The user SPECIFICALLY ASKS to *create, build, or generate* the full hierarchical map of keywords. 
                *   *POSITIVE TRIGGERS:* "Generate", "Build", "Construct cluster", "Map out", "Create the strategy".
                *   *Rule:* Use this ONLY for **BUILD COMMANDS**. If the user is asking *if* they should do a cluster, asking *how* to extend a topic, or asking for an expert opinion *before* building, do **NOT** pick this. Use DIRECT_ANSWER instead.
                *   *Expert SEO Lenses:*
                    - **Semantic Architecture:** Branching topics into specific long-tail queries.
                    - **Semantic Ecosystems:** Identifying concepts and entities (vocabulary of authority).
                    - **User Journey Maps:** Aligning sub-topics with the path from "Awareness" to "Solution".
                    - **Topical Authority Graphs:** Covering every "nook and cranny" to prove expertise and create a defensive moat.
                    - **Information Architecture:** Providing a logical structure (Dewey Decimal System) for internal linking from the Master Pillar to supporting docs.

            4.  **THEN_VS_NOW_PROPOSAL**: Specifically asking for a human-centric 'Then vs Now' structured comparison.
                *   *POSITIVE TRIGGERS:* "Compare human-centric", "Do a then vs now", "Show trajectory", "Contrast eras".
                *   *Rule:* Use this ONLY for explicit trajectory/comparison requests.

            5.  **PSEO_ARTICLE**: **Ghost CMS Content Hook.** Only select this if the message is a NEW request to create a pSEO article or EXPLICITLY mentions "**pSEO**", "**Ghost**", or "**Ghost CMS**" as a command. 
                *   *CRITICAL:* If the user is asking *about* a previous pSEO article (e.g., "How did you integrate X?", "Explain the delivery", "What's the status?"), **DO NOT** select this. Use **SIMPLE_QUESTION** or **DIRECT_ANSWER** instead.

            6.  **SALES_TRANSCRIPT**: **Sales Analysis.** Select this if the user provides a transcript or asks to analyze a sales call for objections, solution briefs, or deal coaching.
                *   *Triggers:* "Analyze this call", "Create a solution brief", "Extract objections", "What were the pain points?".
                *   *Rule:* If the user requests a "Solution Brief" from text, select this.

            7.  **SIMPLE_QUESTION**: **The Factual Scout.** For fact-checks, definitions, or retrieving specific data points that require external grounding or RAG search.
                *   *Triggers:* "What is...", "Explain the concept of...", "Find me the rules for...", "Check the status of...".
                *   *Rule:* Use this for initial questions about the world. If the user asks for a STRUCTURE (Outline/Brief), pick **OPERATIONAL_REFORMAT** instead.

            8.  **DIRECT_ANSWER**: **The Research Architect.** Select this ONLY for **NOVEL SYNTHESIS** or **TECHNICAL AUDITS** that require combining multiple external data sources into a new expert opinion. 
                *   *Example:* "Analyze the security layer of AP2 vs standard PCI-DSS." 

            9.  **OPERATIONAL_REFORMAT**: **The Efficiency Engine.** Select this for any structural command whose COMPLETE context is already available in the history.
                *   *Triggers:* "Summarize our progress", "Reformat these points as bullets", "Make it more professional", "Convert this into a brief".
                *   *CRITICAL:* If the request is purely about structural formatting of existing knowledge, select this.
                *   *EXCEPTION:* If the user says "Repurpose... for pSEO" or "Create a pSEO article", select **PSEO_ARTICLE**. 

            10. **BLOG_OUTLINE**: **The Content Blueprint.** Select this for requests to create, repurpose, or draft a Blog Outline, Content Structure, or Strategy Document.
                *   *Triggers:* "Repurpose the strategy into a proper blog outline", "Create an outline", "Develop a content structure", "Draft a plan".
                *   *Rule:* If the user asks for a structural blueprint (not the full 2500 word article yet), pick this.

            ### NEGATIVE CONSTRAINTS (DO NOT CLASSIFY AS DEEP_DIVE IF):
            - The word "Outline" is present.
            - The user is asking for a "Brief" or "Checklist".
            - The request is purely for the formatting or retrieval of previously discussed information.

            CONVERSATION HISTORY:
            {history_text}

            USER REQUEST (THE COMMAND): "{sanitized_topic}"
            MISSION SUBJECT: "{original_topic}"

            CRITICAL: Respect the "No Outlines in Deep_Dive" rule. Select OPERATIONAL_REFORMAT for Outlines, checklists, or formatting requests using existing data. Select DIRECT_ANSWER ONLY for new, complex synthesis of external data. Respond with ONLY the category name.

            """
        # --- EXECUTION: Triage Model Selection ---
        # Triage on the Command (sanitized_topic) but with history context
        # Use Flash Model for Speed/Cost Efficiency (Gemini 2.0 Flash)
        intent = safe_generate_content(flash_model, triage_prompt)
        print(f"TELEMETRY: Triage V6.0 -> Intent classified as: [{intent}] for command: '{sanitized_topic[:50]}...'")

        # Initialize variables for the response
        new_events = [] 
        
        # Retroactive Event Logging for Code Analysis (since we ran it early)
        if code_files and code_analysis_snippets:
             for i, snippet in enumerate(code_analysis_snippets):
                # reconstruct filename from input if possible, or generic
                f_name = code_files[i].get('name') if i < len(code_files) else "unknown_code"
                new_events.append({
                    "event_type": "code_analysis",
                    "file_name": f_name,
                    "analysis_result": snippet,
                    "status": "success",
                    "timestamp": datetime.datetime.now(datetime.timezone.utc)
                })

        expire_time = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=30)
        
        # --- STEP 4: ROUTING & EXECUTION ---

        # === PATH A: SOCIAL (Fast Exit) ===
        if intent == "SOCIAL_CONVERSATION":
            # Just reply naturally using the history context. No research tools.
            print("Executing Social Response (No Research)")
            social_prompt = f"""
            You are a helpful, witty, and professional AI Research Assistant.
            The user said: "{sanitized_topic}"
            
            Based on the conversation history, provide a brief, warm, and appropriate conversational reply. 
            Do not perform research. Do not write a blog post. Just chat.
            
            History:
            {history_text}
            """
            reply_text = ensure_slack_compatibility(safe_generate_content(unimodel, social_prompt))
            
            new_events.append({"event_type": "agent_reply", "text": reply_text})
            
            # --- Writing to a Sub-collection ---
            session_ref = db.collection('agent_sessions').document(session_id)
            events_ref = session_ref.collection('events')

            # 1. Write Events
            for event in new_events:
                if 'timestamp' not in event: event['timestamp'] = datetime.datetime.now(datetime.timezone.utc)
                events_ref.add(event)

            # 2. Update Parent Metadata (Preserving Context)
            session_ref.update({
                "status": "completed",
                "type": "social",
                "last_updated": expire_time
            })
            requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={"session_id": session_id, "type": "social", "message": reply_text, "channel_id": slack_context['channel'], "thread_ts": slack_context['ts'], "is_initial_post": is_initial_post}, verify=certifi.where(), timeout=10)
            return jsonify({"msg": "Social reply sent"}), 200

        # === PATH B: OPERATIONAL REFORMAT / FAST FOLLOW-UP (Fast Exit) ===
        elif intent == "OPERATIONAL_REFORMAT":
            # GROUNDING GUARD: We only "Fast-Exit" if we actually have history to reformat.
            # If this is the first turn and the topic is new, we MUST research.
            has_substantive_history = len(history_events) > 0 or len(sanitized_topic) > 100 
            
            if has_substantive_history:
                print(f"Executing Operational Reformat (Fast Exit) for command: {sanitized_topic[:50]}")
                # RESTORED: Use simpler natural answer generator to avoid persona-induced bloat
                answer_data = generate_natural_answer(sanitized_topic, "COMPLETE_CONTEXT_IN_HISTORY", history=history_text)
                answer_text = answer_data['text']
                research_intent = answer_data.get('intent', "OPERATIONAL_REFORMAT")
                
                new_events.append({"event_type": "agent_answer", "text": answer_text, "intent": "OPERATIONAL_REFORMAT"})
                
                # Writing to Session Memory
                session_ref = db.collection('agent_sessions').document(session_id)
                events_ref = session_ref.collection('events')
                for event in new_events:
                    if 'timestamp' not in event: event['timestamp'] = datetime.datetime.now(datetime.timezone.utc)
                    events_ref.add(event)

                session_ref.update({
                    "status": "completed",
                    "type": "operational_answer",
                    "last_updated": expire_time
                })
                
                requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={
                    "session_id": session_id, 
                    "type": "social", 
                    "message": answer_text, 
                    "intent": "OPERATIONAL_REFORMAT",
                    "channel_id": slack_context['channel'], 
                    "thread_ts": slack_context['ts'],
                    "is_initial_post": is_initial_post
                }, verify=certifi.where(), timeout=15)
                return jsonify({"msg": "Operational reformat sent"}), 200
            else:
                print(f"OPERATIONAL_REFORMAT fallback: Insufficient grounding. Routing to Research Path.")
                # Fall through to Path D (Work/Research)
                pass

        # === PATH CX: BLOG OUTLINE (High-Fidelity Structure) ===
        elif intent == "BLOG_OUTLINE":
            print(f"Executing Blog Outline generation for command: {sanitized_topic[:50]}")
            # Use Comprehensive Content Strategist for outlines
            answer_data = generate_comprehensive_answer(sanitized_topic, "BLOG_OUTLINE_MODE", history=history_text, context_topic=original_topic)
            answer_text = answer_data['text']
            
            new_events.append({"event_type": "agent_answer", "text": answer_text, "intent": "BLOG_OUTLINE"})
            
            session_ref = db.collection('agent_sessions').document(session_id)
            events_ref = session_ref.collection('events')
            for event in new_events:
                if 'timestamp' not in event: event['timestamp'] = datetime.datetime.now(datetime.timezone.utc)
                events_ref.add(event)

            session_ref.update({
                "status": "completed",
                "type": "work_answer",
                "last_updated": expire_time
            })
            
            requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={
                "session_id": session_id, 
                "type": "social", 
                "message": answer_text, 
                "intent": "BLOG_OUTLINE",
                "channel_id": slack_context['channel'], 
                "thread_ts": slack_context['ts'],
                "is_initial_post": is_initial_post
            }, verify=certifi.where(), timeout=15)
            return jsonify({"msg": "Blog outline sent"}), 200

        # === PATH C: SALES TRANSCRIPT (Fast Exit) ===
        elif intent == "SALES_TRANSCRIPT" or req.get('type') == 'sales_transcript':
            print("Executing Sales Transcript Processing (Fast Exit)")
            transcript_text = req.get('transcript', sanitized_topic)
            
            result = process_sales_transcript(transcript_text, session_id=session_id)
            
            # SUCCESS: Log the event to subcollection for Feedback Worker retrieval
            ts = datetime.datetime.now(datetime.timezone.utc)
            new_events.append({
                "event_type": "agent_proposal", 
                "proposal_type": "solution_brief", 
                "text": result['brief'],
                "proposal_data": {
                    "objections": result['objections'],
                    "brief": result['brief']
                },
                "timestamp": ts
            })
            
            # Persist to Firestore
            print(f"DEBUG: Persisting Solution Brief event for session {session_id}")
            session_ref = db.collection('agent_sessions').document(session_id)
            events_ref = session_ref.collection('events')
            for event in new_events:
                if 'timestamp' not in event: event['timestamp'] = ts
                events_ref.add(event)

            # Update session status
            session_ref.update({
                "status": "awaiting_feedback",
                "type": "solution_brief",
                "last_updated": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=30)
            })

            # Send to N8N for distribution
            requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={
                "session_id": session_id,
                "type": "solution_brief",
                "objections": result['objections'],
                "brief": result['brief'],
                "channel_id": slack_context.get('channel'),
                "thread_ts": slack_context.get('ts'),
                "is_initial_post": is_initial_post
            }, verify=certifi.where(), timeout=30)
            
            return jsonify({"msg": "Solution brief generated"}), 200


        # === PATH B: WORK/RESEARCH (The Heavy Lifting) ===
        # Only now do we pay the cost to distill and research.
        
        # 0. CODE FILE ANALYSIS (Legacy Slot - Now Handled in Step 2.5)
        # We assume code_analysis_context is already populated from the hoisted block.
        code_analysis_context = code_analysis_snippets 
        
        # (Legacy loop removed, as we did it pre-triage)
        
        # 1. Research (Distinguishing between Direct URLs and Exploratory Keywords)
        urls = extract_urls_from_text(sanitized_topic)
        combined_context = []
        
        # Add code analysis context first (highest priority grounding)
        if code_analysis_context:
            combined_context.extend(code_analysis_context)
            print(f"TELEMETRY: Added {len(code_analysis_context)} code analysis insights to grounding context")
        
        if urls:
            print(f"TELEMETRY: URL Mode detected ({len(urls)} links). Bypassing exploratory search.")
            print(f"Detected {len(urls)} URLs. Processing sequentially (Browserless Tier Safety)...")
            for url in urls:
                article_text = fetch_article_content(url)
                combined_context.append(f"GROUNDING_CONTENT (Source: {url}):\n{article_text}")
                # FIX: Persist the scrape into the event history
                new_events.append({"event_type": "tool_call", "tool_name": "scrape_url", "status": "success", "content": f"Source: {url}\n{article_text}"})
                
            # --- SENSORY "DOUBLE-TAP": Analyze detected images from scrape ---
            # Reuse the image pattern from find_trending_keywords
            img_pattern = r'(https?://[^\s<>|]+(?:\.(?:jpg|jpeg|png|webp|gif|pdf)|format=(?:jpg|jpeg|png|webp|gif))(?:\?[^\s<>|]*)?)'
            for ctx in combined_context:
                found_images = re.findall(img_pattern, ctx)
                if found_images:
                    print(f"Sensory Double-Tap: Analyzing {len(found_images[:2])} images found in scrape...")
                    for img_url in found_images[:2]:
                        visual_context = analyze_image(img_url, prompt=f"Analyze this image in the context of the user request: {sanitized_topic}")
                        combined_context.append(f"[VISUAL_INSIGHT]: {visual_context}")
                        new_events.append({"event_type": "tool_call", "tool_name": "analyze_scraped_image", "status": "success", "content": f"[VISUAL_INSIGHT]: {visual_context}"})
        
        # TIER 2: Grounding & Signal Detection (Images or URLs)
        if images or urls:
            print(f"TELEMETRY: Media-Rich Request detected. Analyzing signals and grounding assets...")
            # We call find_trending_keywords to get Geo/Intent signals and analyze any direct images
            r_result = find_trending_keywords(sanitized_topic, history_context=history_text, session_id=session_id, images=images, mission_topic=original_topic, session_metadata=session_data, initial_context=code_analysis_context, triage_intent=intent)
            
            # Merge context: URL scrapes (combined_context) + keyword/image insights (r_result)
            combined_context.extend(r_result['context'])
            research_data = {
                "context": combined_context, 
                "tool_logs": r_result['tool_logs'], 
                "research_intent": "URL_PLUS_SENSORY" if (urls and images) else ("URL_PROCESSING" if urls else "IMAGE_GROUNDING"), 
                "detected_geo": r_result.get('detected_geo')
            }

        else:
            print(f"TELEMETRY: Keyword Mode detected. Launching Sensory Array Router...")
            # FIX: Pass code_analysis_context so the hub can see what we already know
            r_result = find_trending_keywords(sanitized_topic, history_context=history_text, session_id=session_id, images=images, mission_topic=original_topic, session_metadata=session_data, initial_context=code_analysis_context, triage_intent=intent)
            research_data = r_result
            # Unified Merging: Base context (Code Analysis) + Novel Findings (r_result)
            research_data['context'] = code_analysis_context + r_result['context']
            # Ensure Key Persistence
            if "research_intent" not in research_data: research_data["research_intent"] = "KEYWORD_SENSORY"
        
        # --- METADATA EXTRACTION (Thread Memory Optimization) ---
        final_geo = research_data.get("detected_geo", session_data.get("detected_geo", DEFAULT_GEO))
        final_intent_key = "FORMAT_GENERAL"
        try:
            ri = research_data.get("research_intent", "")
            if isinstance(ri, str) and "{" in ri:
                final_intent_key = json.loads(ri).get("intent", "FORMAT_GENERAL")
        except:
             pass
        print(f"TELEMETRY: Signals for Persistence | Geo: {final_geo} | Intent: {final_intent_key}")
        
        # --- SAFETY KILL SWITCH (Hardening) ---
        intent_metadata = research_data.get("research_intent", "")
        is_blocked = False
        reply_text = "I'm sorry, I cannot fulfill this request due to safety guardrails."
        
        if isinstance(intent_metadata, str):
            meta_lower = intent_metadata.lower()
            # 1. Check for the new explicit SIGNAL_BLOCK or legacy VIOLATION intent
            if '"intent": "signal_block"' in meta_lower or '"intent": "violation"' in meta_lower:
                is_blocked = True
            
            # 2. Check for broad refusal keywords using word boundaries
            refusal_keywords = [r"\bharm\b", r"\brefuse\b", r"\billegal\b", r"\bviolence\b", r"\bsensitive\b", r"\bprohibited\b", r"\bviolate\b"]
            if any(re.search(kw, meta_lower) for kw in refusal_keywords):
                is_blocked = True

        if is_blocked:
            print(f"🛑 Safety Kill Switch activated: {intent_metadata}")
            log_safety_event("kill_switch_activated", {"query": original_topic, "signal": intent_metadata})
            
            # Try to extract a specific directive if available
            try:
                msg_json = json.loads(intent_metadata)
                reply_text = msg_json.get("directive", reply_text)
            except:
                pass
                
            new_events.append({"event_type": "safety_kill", "text": reply_text})
            
            # Finalize Session (Same logic as Social PATH)
            session_ref = db.collection('agent_sessions').document(session_id)
            events_ref = session_ref.collection('events')
            for event in new_events:
                if 'timestamp' not in event: event['timestamp'] = datetime.datetime.now(datetime.timezone.utc)
                events_ref.add(event)

            session_ref.update({
                "status": "blocked",
                "type": "safety_violation",
                "detected_geo": final_geo,
                "intent": final_intent_key,
                "last_updated": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=30)
            })
            requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={
                "session_id": session_id, 
                "type": "social", 
                "message": reply_text, 
                "channel_id": slack_context['channel'], 
                "thread_ts": slack_context['ts']
            }, verify=certifi.where(), timeout=10)
            return jsonify({"msg": "Safety block applied"}), 200

        # --- MONITORING: Signal Propagation ---
        if "tool_logs" in research_data: new_events.extend(research_data["tool_logs"])
        print(f"TELEMETRY: Research Phase concluded with {len(research_data.get('context', []))} context snippets.")

        # CHANGE: Ensure downstream functions use the raw topic too, so they see the full request
        clean_topic = sanitized_topic

        # 3. Generate Output based on Intent
        # UPGRADE: Handle structural intents (FORMAT_*) and provide a robust fallback
        # UPGRADE: Handle structural intents (FORMAT_*) and provide a robust fallback
        if intent in ["DIRECT_ANSWER", "SIMPLE_QUESTION", "BLOG_OUTLINE"] or str(intent).startswith("FORMAT_"):
            if intent == "BLOG_OUTLINE": print(f"Executing Blog Outline generation for command: {clean_topic[:50]}")
            
            # ROUTING CHANGE: use generate_natural_answer for fluid interactions
            answer_data = generate_natural_answer(clean_topic, research_data['context'], history=history_text)
            answer_text = answer_data['text']
            research_intent = answer_data['intent']
            formatting_directive = answer_data['directive']
            
            new_events.append({"event_type": "agent_answer", "text": answer_text, "intent": research_intent})
            
            # Writing to a Sub-collection
            session_ref = db.collection('agent_sessions').document(session_id)
            events_ref = session_ref.collection('events')

            for event in new_events:
                if 'timestamp' not in event: event['timestamp'] = datetime.datetime.now(datetime.timezone.utc)
                events_ref.add(event)

            # UPDATE METADATA: Only update 'topic' if it's the first turn to prevent command-hijacking
            update_data = {
                "status": "awaiting_feedback", 
                "type": "work_answer", 
                "slack_context": slack_context,
                "detected_geo": final_geo,
                "intent": final_intent_key,
                "last_updated": expire_time
            }
            if is_initial_post: update_data["topic"] = clean_topic
            session_ref.update(update_data)
            requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={
                "session_id": session_id, 
                "type": "social", 
                "message": answer_text, 
                "intent": research_intent,
                "directive": formatting_directive,
                "channel_id": slack_context['channel'], 
                "thread_ts": slack_context['ts'],
                "is_initial_post": is_initial_post
            }, verify=certifi.where(), timeout=15)
            return jsonify({"msg": "Answer sent"}), 200

        elif intent == "DEEP_DIVE":
            target_count = extract_target_word_count(sanitized_topic, history=history_text)
            print(f"TELEMETRY: Executing Dynamic Deep-Dive Recursive Expansion (Target: {target_count})...")
            # PASS THE sanitzied_topic as the DIRECTIVE to prioritize feedback
            article_html = generate_deep_dive_article(sanitized_topic, research_data['context'], history=history_text, history_events=history_events, target_length=target_count, target_geo=final_geo)
            
            # Use Claude for metadata even for deep dives
            seo_data = generate_seo_metadata(article_html, original_topic)
            
            # SUCCESS: Log the event to subcollection for Feedback Worker retrieval
            new_events.append({
                "event_type": "agent_proposal", 
                "proposal_type": "deep_dive", 
                "text": convert_html_to_markdown(article_html),
                "proposal_data": {"article_html": article_html, "seo": seo_data}
            })
            
            session_ref = db.collection('agent_sessions').document(session_id)
            events_ref = session_ref.collection('events')
            for event in new_events:
                if 'timestamp' not in event: event['timestamp'] = datetime.datetime.now(datetime.timezone.utc)
                events_ref.add(event)

            update_data = {
                "status": "awaiting_feedback", 
                "type": "work_proposal", 
                "slack_context": slack_context,
                "detected_geo": final_geo,
                "intent": final_intent_key,
                "last_updated": expire_time
            }
            if is_initial_post: update_data["topic"] = seo_data.get('title', clean_topic)
            session_ref.update(update_data)

            requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={
                "session_id": session_id, 
                "type": "social", 
                "message": convert_html_to_markdown(article_html), 
                "channel_id": slack_context['channel'], 
                "thread_ts": slack_context['ts'],
                "is_initial_post": is_initial_post
            }, verify=certifi.where(), timeout=30)
            return jsonify({"msg": "Deep-Dive article sent"}), 200

        elif intent == "TOPIC_CLUSTER_PROPOSAL":
            try:
                cluster_data = generate_topic_cluster(sanitized_topic, research_data['context'], history=history_text, is_initial=is_initial_post)
                if not cluster_data or "clusters" not in cluster_data: raise ValueError("Failed to parse valid cluster JSON.")
                
                new_events.append({
                    "event_type": "agent_proposal", 
                    "proposal_type": "topic_cluster", 
                    "proposal_data": cluster_data
                })
                
                session_ref = db.collection('agent_sessions').document(session_id)
                events_ref = session_ref.collection('events')
                for event in new_events:
                    if 'timestamp' not in event: event['timestamp'] = datetime.datetime.now(datetime.timezone.utc)
                    events_ref.add(event)

                update_data = {
                    "status": "awaiting_feedback", 
                    "type": "work_proposal", 
                    "slack_context": slack_context,
                    "detected_geo": final_geo,
                    "intent": final_intent_key,
                    "last_updated": expire_time
                }
                if is_initial_post: update_data["topic"] = clean_topic
                session_ref.update(update_data)
                
                # Align with N8N Parser: Wrap JSON in markdown for regex extraction
                cluster_msg = f"```json\n{json.dumps(cluster_data, indent=2)}\n```"
                requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={
                    "session_id": session_id, 
                    "type": "answer", 
                    "message": cluster_msg,
                    "intent": "TOPIC_CLUSTER_PROPOSAL",
                    "channel_id": slack_context['channel'], 
                    "thread_ts": slack_context['ts'],
                    "is_initial_post": is_initial_post
                }, verify=certifi.where(), timeout=30) # Increased timeout for large clusters
                return jsonify({"msg": "Topic cluster sent"}), 200

            except Exception as e:
                print(f"TELEMETRY: ⚠️ TOPIC_CLUSTER Fallback: {e}")
                answer_data = generate_comprehensive_answer(clean_topic, research_data['context'], history=history_text, context_topic=original_topic)
                answer_text = answer_data['text']
                new_events.append({"event_type": "agent_answer", "text": answer_text})
                
                session_ref = db.collection('agent_sessions').document(session_id)
                events_ref = session_ref.collection('events')
                for event in new_events:
                    if 'timestamp' not in event: event['timestamp'] = datetime.datetime.now(datetime.timezone.utc)
                    events_ref.add(event)

                update_data = {
                    "status": "awaiting_feedback", 
                    "type": "work_answer", 
                    "slack_context": slack_context,
                    "detected_geo": final_geo,
                    "intent": final_intent_key,
                    "last_updated": expire_time
                }
                if is_initial_post: update_data["topic"] = clean_topic
                session_ref.update(update_data)
                requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={"session_id": session_id, "type": "social", "message": answer_text, "channel_id": slack_context['channel'], "thread_ts": slack_context['ts'], "is_initial_post": is_initial_post}, verify=certifi.where(), timeout=15)
                return jsonify({"msg": "Cluster Fallback Answer sent"}), 200
        elif intent == "THEN_VS_NOW_PROPOSAL":
            try:
                # FIX: Pass original_topic (Mission) for tool consistency
                current_proposal = create_euphemistic_links({**research_data, "clean_topic": original_topic}, is_initial=is_initial_post)
                if not current_proposal or "interlinked_concepts" not in current_proposal:
                     raise ValueError("Failed to generate Then-vs-Now proposal.")
                
                approval_id = f"approval_{uuid.uuid4().hex[:8]}"
                new_events.append({"event_type": "loop_draft", "iteration": 0, "proposal_data": current_proposal})

                loop_count = 0
                while loop_count < MAX_LOOP_ITERATIONS:
                    critique = critique_proposal(original_topic, current_proposal)
                    if "APPROVED" in critique.upper(): break
                    try: 
                         refined = refine_proposal(original_topic, current_proposal, critique)
                         if refined and "interlinked_concepts" in refined:
                              current_proposal = refined
                         else:
                              break
                    except Exception: break
                    loop_count += 1

                approval_id = f"approval_{uuid.uuid4().hex[:8]}"
                new_events.append({"event_type": "adk_request_confirmation", "approval_id": approval_id, "payload": current_proposal['interlinked_concepts']})
                
                session_ref = db.collection('agent_sessions').document(session_id)
                events_ref = session_ref.collection('events')
                for event in new_events:
                    if 'timestamp' not in event: event['timestamp'] = datetime.datetime.now(datetime.timezone.utc)
                    events_ref.add(event)

                update_data = {
                    "status": "awaiting_feedback", 
                    "type": "work_proposal", 
                    "slack_context": slack_context,
                    "detected_geo": final_geo,
                    "intent": final_intent_key,
                    "last_updated": expire_time
                }
                if is_initial_post: update_data["topic"] = clean_topic
                session_ref.update(update_data)

                requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={
                    "session_id": session_id, 
                    "type": "answer_proposal", 
                    "proposal": current_proposal['interlinked_concepts'], 
                    "approval_id": approval_id, 
                    "channel_id": slack_context['channel'], 
                    "thread_ts": slack_context['ts'],
                    "is_initial_post": is_initial_post
                }, verify=certifi.where(), timeout=20)
                return jsonify({"msg": "Then-vs-Now proposal sent"}), 200

            except ValueError as e:
                # Fallback
                answer_text = generate_comprehensive_answer(clean_topic, research_data['context'], history=history_text, context_topic=original_topic)['text']
                new_events.append({"event_type": "agent_answer", "text": answer_text})
                
                session_ref = db.collection('agent_sessions').document(session_id)
                events_ref = session_ref.collection('events')

                for event in new_events:
                    if 'timestamp' not in event: event['timestamp'] = datetime.datetime.now(datetime.timezone.utc)
                    events_ref.add(event)

                update_data = {
                    "status": "awaiting_feedback", 
                    "type": "work_answer", 
                    "slack_context": slack_context,
                    "detected_geo": final_geo,
                    "intent": final_intent_key,
                    "last_updated": expire_time
                }
                if is_initial_post: update_data["topic"] = clean_topic
                session_ref.update(update_data)
                requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={"session_id": session_id, "type": "social", "message": answer_text, "channel_id": slack_context['channel'], "thread_ts": slack_context['ts'], "is_initial_post": is_initial_post}, verify=certifi.where(), timeout=15)
                return jsonify({"msg": "Fallback answer sent"}), 200

        # === PATH C: PSEO ARTICLE GENERATION (Dual-Agent Path) ===
        elif intent == "PSEO_ARTICLE":
            try:
                # 1. AGENT A (Gemini): Generate the Body Content (Prioritizing Latest Feedback)
                article_html = generate_pseo_article(sanitized_topic, research_data['context'], history=history_text, history_events=history_events, is_initial_post=is_initial_post)
                
                # 2. AGENT B (Claude): Generate the Semantic Metadata
                seo_data = generate_seo_metadata(article_html, original_topic)
                
                # SUCCESS: Log the event to subcollection
                new_events.append({
                    "event_type": "agent_proposal", 
                    "proposal_type": "pseo_article", 
                    "text": convert_html_to_markdown(article_html),
                    "proposal_data": {"article_html": article_html}
                })
                
                # 4. Write to Database
                session_ref = db.collection('agent_sessions').document(session_id)
                events_ref = session_ref.collection('events')
                for event in new_events:
                    if 'timestamp' not in event: event['timestamp'] = datetime.datetime.now(datetime.timezone.utc)
                    events_ref.add(event)

                # UPDATE METADATA: Preserve Mission Topic
                update_data = {
                    "status": "awaiting_feedback", 
                    "type": "work_proposal", 
                    "slack_context": slack_context,
                    "detected_geo": final_geo,
                    "intent": final_intent_key,
                    "last_updated": expire_time
                }
                if is_initial_post: update_data["topic"] = seo_data.get('title', clean_topic)
                session_ref.update(update_data)
                
                # 5. Send STRUCTURED DATA to N8N (Restored Wrapper for GhostNode Harmony)
                requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={
                    "session_id": session_id, 
                    "type": "pseo_draft", 
                    "payload": {
                        "title": seo_data.get("title", "Untitled Draft"),
                        "html": article_html,
                        "tags": seo_data.get("tags", []),
                        "custom_excerpt": seo_data.get("custom_excerpt"),
                        "meta_title": seo_data.get("meta_title"),
                        "meta_description": seo_data.get("meta_description"),
                        "featured_image_prompt": seo_data.get("featured_image_prompt"),
                        "status": "draft"
                    },
                    "channel_id": slack_context['channel'], 
                    "thread_ts": slack_context['ts'],
                    "is_initial_post": is_initial_post
                }, verify=certifi.where(), timeout=30)

                return jsonify({"msg": "pSEO Draft sent"}), 200

            except Exception as e:
                print(f"⚠️ PSEO_ARTICLE Fallback: {e}")
                answer_data = generate_comprehensive_answer(clean_topic, research_data['context'], history=history_text, context_topic=original_topic)
                answer_text = answer_data['text']
                new_events.append({"event_type": "agent_answer", "text": answer_text})
                
                session_ref = db.collection('agent_sessions').document(session_id)
                events_ref = session_ref.collection('events')
                for event in new_events:
                    if 'timestamp' not in event: event['timestamp'] = datetime.datetime.now(datetime.timezone.utc)
                    events_ref.add(event)

                update_data = {
                    "status": "awaiting_feedback", 
                    "type": "work_answer", 
                    "slack_context": slack_context,
                    "detected_geo": final_geo,
                    "intent": final_intent_key,
                    "last_updated": expire_time
                }
                if is_initial_post: update_data["topic"] = clean_topic
                session_ref.update(update_data)
                requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={"session_id": session_id, "type": "social", "message": answer_text, "channel_id": slack_context['channel'], "thread_ts": slack_context['ts'], "is_initial_post": is_initial_post}, verify=certifi.where(), timeout=15)
                return jsonify({"msg": "pSEO Fallback Answer sent"}), 200


        else: 
            # ROBUST FALLBACK: If intent is unknown, default to a natural answer rather than crashing with 500
            print(f"TELEMETRY: ⚠️ Unknown Intent Detected: '{intent}'. Defaulting to Natural Answer fallback.")
            answer_data = generate_natural_answer(clean_topic, research_data['context'], history=history_text)
            answer_text = answer_data['text']
            research_intent = answer_data['intent']
            formatting_directive = answer_data['directive']
            
            new_events.append({"event_type": "agent_answer", "text": answer_text, "intent": research_intent, "status": "intent_fallback"})
            
            session_ref = db.collection('agent_sessions').document(session_id)
            events_ref = session_ref.collection('events')
            for event in new_events:
                if 'timestamp' not in event: event['timestamp'] = datetime.datetime.now(datetime.timezone.utc)
                events_ref.add(event)

            # UPDATE METADATA (Fallback Persistence)
            update_data = {
                "status": "awaiting_feedback", 
                "type": "work_answer", 
                "slack_context": slack_context,
                "detected_geo": final_geo,
                "intent": final_intent_key,
                "last_updated": expire_time
            }
            if is_initial_post: update_data["topic"] = clean_topic
            session_ref.update(update_data)

            requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={
                "session_id": session_id, 
                "type": "social", 
                "message": answer_text, 
                "intent": research_intent,
                "directive": formatting_directive,
                "channel_id": slack_context['channel'], 
                "thread_ts": slack_context['ts'],
                "is_initial_post": is_initial_post
            }, verify=certifi.where(), timeout=15)
            
            return jsonify({"msg": "Unknown intent fallback answer sent"}), 200

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"❌ WORKER ERROR: {e}")
        print(f"STACK TRACE:\n{error_trace}")
        return jsonify({"error": str(e)}), 500

# --- FUNCTION: Ingest Knowledge ---
@functions_framework.http
def ingest_knowledge(request):
    """
    Ingests story data and solution briefs into the knowledge base.
    Secured with API key authentication.
    """
    # Security check: Validate API key
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({"error": "Missing or invalid Authorization header"}), 401
    
    provided_key = auth_header.replace('Bearer ', '')
    if provided_key != INGESTION_API_KEY:
        print(f"⚠️ Unauthorized access attempt from IP: {request.remote_addr}")
        return jsonify({"error": "Invalid API key"}), 403

    global db, unimodel
    if db is None: db = firestore.Client(project=PROJECT_ID)
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")

    req = request.get_json(silent=True) or {}
    session_id = req.get('session_id', 'unknown')
    topic = req.get('topic', 'unknown')
    final_story = req.get('story', '')
    
    # Unified Ingestion Router (Determines destination collection)
    ingest_type = req.get('type', 'knowledge') # 'knowledge' or 'solution_brief'
    
    if ingest_type == 'solution_brief':
        # Solution Briefs are structured, not chunked for vector search (Session Memory focus)
        db.collection('solution_briefs').add({
            'session_id': session_id,
            'topic': topic,
            'objections': req.get('objections'),
            'brief': req.get('story'),
            'created_at': datetime.datetime.now(datetime.timezone.utc)
        })
        return jsonify({"msg": "Solution brief ingested."}), 200

    # Knowledge Base Path: Chunk and Embed
    chunks = chunk_text(final_story)
    embeddings = embedding_model.get_embeddings([c for c in chunks])
    
    batch = db.batch()
    count = 0
    for i, (text_segment, embedding_obj) in enumerate(zip(chunks, embeddings)):
        doc_ref = db.collection('knowledge_base').document(f"{session_id}_{i}")
        batch.set(doc_ref, {
            "content": scrub_pii(text_segment),
            "embedding": Vector(embedding_obj.values),
            "topic_trigger": scrub_pii(topic), 
            "source_session": session_id,
            "chunk_index": i,
            "created_at": datetime.datetime.now(datetime.timezone.utc)
        })
        count += 1
        if count >= 499:
            batch.commit()
            batch = db.batch()
            count = 0

    if count > 0: batch.commit()

    print(f"Ingested {len(chunks)} chunks.")
    return jsonify({"msg": "Knowledge ingested."}), 200



# --- COMPLIANCE DOCUMENT INGESTION ENDPOINT ---
@functions_framework.http
def ingest_compliance_docs(request):
    """
    Ingests compliance PDFs from external URLs.
    Zero-cost implementation: No backup storage, re-run script if URLs fail.
    Secured with API key authentication.
    """
    # Security check: Validate API key
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({"error": "Missing or invalid Authorization header"}), 401
    
    provided_key = auth_header.replace('Bearer ', '')
    if provided_key != INGESTION_API_KEY:
        print(f"⚠️ Unauthorized access attempt from IP: {request.remote_addr}")
        return jsonify({"error": "Invalid API key"}), 403
    
    # Initialize clients
    global db
    if db is None:
        db = firestore.Client(project=PROJECT_ID)
        
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
    
    req = request.get_json(silent=True)
    doc_url = req.get('doc_url')
    doc_type = req.get('doc_type')
    doc_version = req.get('version', '1.0')
    tenant_id = req.get('tenant_id', 'global')
    geo_scope = req.get('geo_scope', [])
    industry_scope = req.get('industry_scope', ['all'])
    
    if not doc_url or not doc_type:
        return jsonify({"error": "Missing doc_url or doc_type"}), 400
    
    # Fetch PDF content via optimized MCP sensory hub (Hybrid HTML/PDF scraper)
    try:
        print(f"Fetching {doc_type} via Smart MCP Hub: {doc_url}")
        scrape_data = get_mcp_client().call("scrape_article", {"url": doc_url})
        
        if not scrape_data or "Error" in scrape_data:
            return jsonify({"error": "MCP Hub Error", "details": scrape_data}), 500
            
        pdf_content = scrape_data
        
        if not pdf_content or len(pdf_content) < 100:
            raise ValueError("Smart Hub failed to extract meaningful text")
            
        print(f"✅ Successfully fetched {len(pdf_content)} characters via Smart Hub")
        
    except Exception as e:
        error_msg = f"Failed to fetch content via Smart Hub: {str(e)}"
        print(f"❌ {error_msg}")
        return jsonify({"error": error_msg}), 500
    
    # Chunk and embed (reuse existing logic)
    chunks = chunk_text(pdf_content)
    embeddings = embedding_model.get_embeddings(chunks)
    
    # Store in Firestore with multi-tenant metadata
    batch = db.batch()
    count = 0
    
    for i, (chunk, embedding_obj) in enumerate(zip(chunks, embeddings)):
        doc_ref = db.collection('compliance_knowledge').document(f"{doc_type}_{doc_version}_{i}")
        batch.set(doc_ref, {
            "content": scrub_pii(chunk),
            "embedding": Vector(embedding_obj.values),
            "doc_type": doc_type,
            "doc_source": doc_url,
            "doc_version": doc_version,
            "tenant_id": tenant_id,
            "geo_scope": geo_scope,
            "industry_scope": industry_scope,
            "chunk_index": i,
            "created_at": datetime.datetime.now(datetime.timezone.utc)
        })
        
        count += 1
        
        if count >= 499:
            batch.commit()
            batch = db.batch()
            count = 0
    
    if count > 0:
        batch.commit()
    
    print(f"✅ Ingested {len(chunks)} chunks from {doc_type}")
    
    return jsonify({
        "msg": f"Ingested {len(chunks)} chunks from {doc_type}",
        "doc_type": doc_type,
        "chunks_count": len(chunks),
        "doc_source": doc_url
    }), 200
