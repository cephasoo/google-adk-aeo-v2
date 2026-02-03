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
    client = google.cloud.logging.Client()
    client.setup_logging()
except Exception:
    pass # Fallback to standard logging if local

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
                timeout=60,
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
        url = os.environ.get("MCP_SERVER_URL", "http://localhost:8080")
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
                # LiteLLM Call
                response = completion(
                    model=self.model_name, 
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temp,
                    max_tokens=2048
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
        # Compatibility stub for worker-tracker alignment
        return {"cited": False, "snippet": "Feature not active in worker-story."}


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
    default = "Senior Marketers, CTOs, and Founders. They value depth, nuance, and honesty over hype."
    if not history: return default
    
    hist_lower = history.lower()
    if "8-grader" in hist_lower or "8th grade" in hist_lower or "explain like i'm 5" in hist_lower or "eli5" in hist_lower:
         return "An 8th-grader. Use extremely simple words, short sentences, and clear analogies. Avoid jargon."
    elif "non-technical" in hist_lower:
         return "Non-technical business owners. Focus on value and 'what it does' rather than 'how it works'."
    
    return default

def log_safety_event(event_name, data):
    """Logs safety events to Google Cloud Logging for audit traces."""
    try:
        logging_client = google.cloud.logging.Client()
        logger = logging_client.logger("safety_audit")
        logger.log_struct({
            "event": event_name,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            **data
        }, severity="WARNING")
    except Exception as e:
        print(f"Logging Block: {e}")

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
    
    core_topic = unimodel.generate_content(extraction_prompt).text.strip()

    print(f"Distilled Core Topic: '{core_topic}'")
    return core_topic

def extract_target_word_count(text):
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
        Analyze the user's request: "{text}"
        
        Extract the implied target word count as a single integer.
        
        ### INTERPRETATION RULES:
        1. **Explicit Numbers**: '1500 words' -> 1500.
        2. **Floors/Expansion (+, >, at least)**: If the user implies a minimum (e.g., '1600+', '>1500', 'at least 2000'), provide a target slightly ABOVE that number to satisfy the request.
        3. **Ceilings/Constraint (<, less than, max)**: If the user implies a limit (e.g., '<1000', 'max 500'), provide a target BELOW that number.
        4. **Vague Descriptors**: 
           - "Deep dive" / "Long form" -> ~2000
           - "Short" / "Brief" -> ~800
           - "Standard" / "Default" -> 1500
        
        OUTPUT: Return ONLY the integer (e.g., 1800). Do not write sentences.
        """
        
        # 2. Generate
        response = specialist_model.generate_content(prompt).text.strip()
        
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
    
    # 3. Lists (Simple conversion)
    text = text.replace('<ul>', '').replace('</ul>', '')
    text = text.replace('<li>', '• ').replace('</li>', '\n')
    
    # 4. Final Strip of remaining tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Clean up excess newlines (Max 2)
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
        entity = unimodel.generate_content(prompt).text.strip().replace('"', '').replace("'", "")
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
        client = secretmanager.SecretManagerServiceClient()
        response = client.access_secret_version(name=f"projects/{PROJECT_ID}/secrets/google-search-api-key/versions/latest")
        search_api_key = response.payload.data.decode("UTF-8")
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
        print(f"FAILED JSON PARSE: {e}\nRaw Content: {content[:200]}")
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
        response = specialist_model.generate_content(prompt).text.strip()
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
def find_trending_keywords(raw_topic, history_context="", session_id=None, images=None, mission_topic=None, session_metadata=None, initial_context=None):
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
    visual_check_prompt = f"Analyze if the user's query requires seeing an image, PDF, or visual asset to answer accurately. Query: '{raw_topic}'. Respond ONLY with 'YES' or 'NO'."
    is_recollective_visual_query = "YES" in unimodel.generate_content(visual_check_prompt).text.upper()
    
    if not all_images and is_recollective_visual_query and history_context:
        print("Sensory Router: Semantic visual intent detected. Attempting Deep Recollection (Re-scraping URLs)...")
        # ... (rest of the logic remains the same)
        history_web_links = re.findall(r'(https?://[^\s<>|]+(?:twitter\.com|x\.com|instagram\.com|facebook\.com|threads\.net|tiktok\.com|youtube\.com|status/|p/)[^\s<>|]*)', history_context)
        if history_web_links:
            top_link = history_web_links[0].strip('><')
            print(f"Sensory Router: Re-scraping historical link for visual assets: {top_link}")
            scrape_data = get_mcp_client().call("scrape_article", {"url": top_link})
            found_img_links = re.findall(img_pattern, scrape_data)
            if found_img_links:
                print(f"Sensory Router: Deep Recollection successful. Found {len(found_img_links)} assets.")
                all_images.extend(found_img_links[:2])
                tool_logs.append({"event_type": "tool_call", "tool_name": "recollective_scrape", "status": "success", "content": f"Found images: {found_img_links[:2]}"})

            # --- TURBO CHARGED: Deep Navigator (Recursive Link Analysis) ---
            # If the scrape was successful, check if we should go deeper
            if scrape_data and "[DETECTED_LINKS]:" in scrape_data:
                links_section = scrape_data.split("[DETECTED_LINKS]:")[1]
                deep_url = analyze_deep_navigation(links_section, grounding_subject, history_context)
                
                if deep_url:
                    print(f"Sensory Router: Deep Navigator decided to click: {deep_url}")
                    deep_data = get_mcp_client().call("scrape_article", {"url": deep_url})
                    if deep_data:
                         # Append the recursive finding as high-value context
                        insight = f"[DEEP_NAVIGATOR_INSIGHT (Source: {deep_url})]:\n{deep_data[:8000]}"
                        context_snippets.append(insight)
                        base_grounding.append(insight)
                        tool_logs.append({"event_type": "tool_call", "tool_name": "deep_navigator_scrape", "status": "success", "content": f"Recursively scraped: {deep_url}"})
                    else:
                        print("Sensory Router: Deep click returned no data.")
                else:
                    print("Sensory Router: Deep Navigator decided NO_ACTION (No critical link found).")

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
    # 2. Automated Signal Detection (Always-On Parallel + Smart Fallback)
    print("Sensory Router: Detecting Signals (Geo & Intent) via MCP Hub...")
    
    # SAFETY: Ensure session_metadata is a dict to prevent NoneType errors
    if session_metadata is None:
        session_metadata = {}
    
    prev_geo = session_metadata.get('detected_geo', 'Global')
    prev_intent = session_metadata.get('intent', 'FORMAT_GENERAL')
    
    # ALWAYS EXECUTE PARALLEL: Use ThreadPoolExecutor for concurrent MCP Hub calls
    # This ensures we are ALWAYS listening for pivots (e.g. "Why is it different in France?")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_geo = executor.submit(mcp_detect_geo, raw_topic, history=history_context)
        future_intent = executor.submit(mcp_detect_intent, raw_topic)
        
        try:
            detected_geo = future_geo.result(timeout=8)  # Reduced from 10s to prevent cascading timeouts
            research_intent_raw = future_intent.result(timeout=8)
        except concurrent.futures.TimeoutError:
            print(f"⚠️ Parallel Signal Detection timed out. Using graceful fallback to session memory.")
            # GRACEFUL DEGRADATION: If MCP is slow, inherit from session memory entirely
            detected_geo = prev_geo
            research_intent_raw = json.dumps({"intent": prev_intent, "directive": "Fallback due to timeout"})
        except Exception as e:
            print(f"⚠️ Parallel Signal Detection failed: {e}. Attempting sequential fallback...")
            try:
                detected_geo = mcp_detect_geo(raw_topic, history=history_context)
                research_intent_raw = mcp_detect_intent(raw_topic)
            except Exception as fallback_error:
                print(f"❌ Sequential fallback also failed: {fallback_error}. Using session memory.")
                detected_geo = prev_geo
                research_intent_raw = json.dumps({"intent": prev_intent, "directive": "Fallback due to error"})

    # --- SMART INHERITANCE LOGIC ---
    # Rule: If new signal is generic ("Global") but memory is specific, inherit.
    # Rule: If new signal is specific, prioritize it (Pivot).
    if detected_geo == "Global" and prev_geo != "Global":
        print(f"  -> Smart Inheritance: New signal is Global. Falling back to session memory: '{prev_geo}'")
        detected_geo = prev_geo
    elif detected_geo != "Global" and detected_geo != prev_geo:
        print(f"  -> Topic Pivot: New specific Geo signal detected: '{detected_geo}'. Overriding memory.")

    print(f"  -> Signal Detected | Geo: '{detected_geo}' | Intent: '{research_intent_raw}'")
    
    # Parse for the prompt context to avoid JSON clutter
    try:
        intent_json = json.loads(research_intent_raw)
        detected_intent_key = intent_json.get("intent", "FORMAT_GENERAL")
    except:
        detected_intent_key = "FORMAT_GENERAL"
        
    print(f"  -> Signal Detected | Geo: '{detected_geo}' | Intent: '{detected_intent_key}'")
    
    # --- 2.5 SENSORY GATEKEEPER: Grounding Adequacy Audit ---
    # We check if we ALREADY have enough info in context_snippets + history to satisfy the request.
    # This specifically addresses the "Token Economy" requirement.
    audit_prompt = f"""
    Evaluate if the following knowledge is SUFFICIENT to answer the user's request without further web research.
    
    USER REQUEST: "{raw_topic}"
    
    COLLECTED KNOWLEDGE (RAG + HISTORY + VISUALS + CODE ANALYSIS):
    {history_context}
    ---
    {base_grounding}
    
    RULES:
    1. If the user is asking to reorganize, summarize, edit, or COMPOSE (create a draft/outline) based on information whose COMPLETE context has already been retrieved and is visible in the history or snippets, it is SUFFICIENT.
    2. WEB RESEARCH VS COMPOSITION: If the user commands an action (e.g., "Draft a post") that can be fulfilled using the EXISTING history or MISSION SUBJECT knowledge, select SUFFICIENT. We only need more research (INSUFFICIENT) if the request requires fresh signals or facts not yet established.
    3. PROFUNDITY MATCH: Surface-level context is INSUFFICIENT if the user request demands high-fidelity technical depth not already grounded in history.
    4. If the user's query is purely conversational/social, it is SUFFICIENT.

    OUTPUT FORMAT:
    RATIONALE: <1-sentence explanation>
    DECISION: [SUFFICIENT or INSUFFICIENT]
    """
    audit_raw = specialist_model.generate_content(audit_prompt).text.strip().upper()
    print(f"SENSORY GATEKEEPER: {audit_raw}")
    
    adequacy_score = "INSUFFICIENT"
    if "DECISION: SUFFICIENT" in audit_raw:
        adequacy_score = "SUFFICIENT"
    elif "DECISION: INSUFFICIENT" in audit_raw:
        adequacy_score = "INSUFFICIENT"

    if adequacy_score == "SUFFICIENT" and "TRENDS" not in detected_intent_key:
        print("SENSORY GATEKEEPER: Knowledge is sufficient. Skipping AI Router and forcing CONVERSATIONAL_CONTEXT.")
        selected_tools = ["USE_CONVERSATIONAL_CONTEXT"]
        return {"context": context_snippets, "tool_logs": tool_logs, "research_intent": research_intent_raw, "detected_geo": detected_geo}

    # 3. Sensory Array Router (Decides based on RAW input)
    tool_choice_prompt = f"""
    Analyze the user's query to select the most appropriate research tool(s).

    CONVERSATION HISTORY:
    {history_context}

    GROUNDING ASSETS (Code Analysis / Visuals / Prior RAG):
    {base_grounding}

    LONG-TERM MEMORY (RAG Results):
    {rag_text}

    USER'S CORE QUERY (THE COMMAND): '{raw_topic}'
    MISSION SUBJECT (THE CONTEXT): '{grounding_subject}'
    
    ### DETECTED CONTEXT (MCP Hub Signals):
    - PRIMARY GEO TARGET: '{detected_geo}'
    - REQUESTED FORMAT INTENT: '{detected_intent_key}'

    ### DECISION PROTOCOL (CHECK IN ORDER):
    1. **CHECK HISTORY (Priority #1):**
        - If the user is asking for information (or a REFORMATTING like an OUTLINE) that CAN BE GATHERED from the "LONG-TERM MEMORY" or "CONVERSATION HISTORY", select **USE_CONVERSATIONAL_CONTEXT**.
        - If the history contains the FACTUAL CONTENT needed but the user wants a new structure (e.g. "Create an outline for the above"), select **USE_CONVERSATIONAL_CONTEXT**.
        - **INTERPRETIVE EXCEPTION:** If the user asks for **REASONS**, **DRIVERS**, **CAUSES**, or **"WHY"** (and the history only contains stats/numbers), **DO NOT** select USE_CONVERSATIONAL_CONTEXT. Proceed to research.

    2. **CHECK CREATIVE INTENT (Priority #2):**
        - If the user asks you to **DRAFT**, **CREATE**, or **WRITE** content for an entity that is clearly **INVENTED**, **FICTIONAL**, or defined in history, select **USE_CONVERSATIONAL_CONTEXT**.

    3. **NEW RESEARCH (Priority #3):**
        - If the user wants you to gather NEW information not in history, use a tool below.
        - If the user wants you to explain a trend (and the history contains only numbers/stats but not the *explanation*), select a tool below.

    ### AVAILABLE TOOLS:
    - WEB: Use this whenever the request requires **Factual Discovery** or **External Grounding** not available in history.
        *   Examples: "Research X", "Find data on Y", "Ground this in technical specifics", "Use latest signals".
    - IMAGES: For visual inspiration.
    - VIDEOS: For tutorials/clips.
    - TRENDS: For viral awareness. Select ONLY for measuring "What is trending" or "Viral topics" in a specific region.
    - ANALYSIS: For historical trajectory. Select ONLY for "Interest over time" or "Growth metrics".
    - SIMPLE: General web searches for broad definitions.
    - COMPLIANCE: **CRITICAL.** Select this for ANY question involving **Data Privacy**, **Breach Reporting**, **Laws**, or **Regulations** (GDPR, NDPA, SOC2, etc.).
        *   Trigger this whenever the user asks "What do I need to do?", "How long do I have?", or "What are the rules?" in a business or technical context.
    - USE_CONVERSATIONAL_CONTEXT: Select this for **Composition**, **Formatting**, or **Incremental Refinement** using information already established in the conversation.
        *   Examples: "Write a draft", "Create an outline", "Summarize our notes", "Refine the tone", "Include more detail on [Topic A]", write about a hypothetical or user-defined entity.

    ### DECISION REASONING:
    - If a request combines **Drafting** with **Research Intent** (e.g. "Draft a deep-dive based on new research"), the priority is to select **WEB** + any other necessary tools.
    - If a request is purely about **Execution/Form** based on existing knowledge, select **USE_CONVERSATIONAL_CONTEXT**.

    Respond ONLY with the tool selection(s), comma-separated (e.g., "WEB, ANALYSIS" or "IMAGES" or "USE_CONVERSATIONAL_CONTEXT").
    Do not add country codes; the system handles locations via the MCP signals.
    """
    # 4. Robust Parsing Logic
    try:
        raw_response = specialist_model.generate_content(tool_choice_prompt).text.strip().upper()
        clean_response = raw_response.replace("*", "").replace("`", "").replace("'", "").replace('"', "").rstrip(".")
        
        selected_tools = [t.strip() for t in clean_response.split(',')]
        logging.info(f"TELEMETRY: Router decided on Strategy: {selected_tools} | Target Geo: {detected_geo}")
    except Exception as e:

        logging.error(f"Router Parse Error: {e}. Defaulting to SIMPLE.")
        selected_tools = ["SIMPLE"]

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
        print("Sensory Router: Category [COMPLIANCE] active. Searching Regulatory Hub...")
        # We pass detected_geo as geo_scope to filter results relevant to that region
        compliance_context = search_compliance_knowledge(grounding_subject, geo_scope=detected_geo)
        if compliance_context:
            context_snippets.append(compliance_context)
            tool_logs.append({"event_type": "tool_call", "tool_name": "compliance_knowledge_retrieval", "input": raw_topic, "status": "success", "content": str(compliance_context)[:500]})

    # --- 2. REGIONAL TREND TOOLS (Region-Specific Discovery) ---
    trend_tools = [t.split(':')[0].strip() for t in selected_tools if t.split(':')[0].strip() in ["TRENDS", "ANALYSIS"]]
    
    if trend_tools:
        target_geos = [detected_geo] if detected_geo else ["NG"]
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
    1. **Semantic Ecosystems (Entities over Strings):** Do not just match keywords. Identify the core concepts and entities (vocabulary of authority) required to be seen as an expert.
    2. **User Journey Maps:** Align clusters with the user's path from "Problem Awareness" (Informational/Top-of-funnel) to "Solution" (Transactional/Middle/Bottom-of-funnel).
    3. **Topical Authority Graphs:** Cover the topic's "nooks and crannies" to prove comprehensive expertise and create a defensive moat.
    4. **Information Architecture (The Library Model):** Ensure the structure follows a logical internal linking blueprint from the Master Pillar to supporting sub-topics.
    
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
    response = model.generate_content(prompt)
    return extract_json(response.text)

# 12. The SEO Metadata Generator (Using Specialist Model)
def generate_seo_metadata(article_html, topic):
    """
    Uses the UnifiedModel adapter to route this specific task to Anthropic (Claude).
    """
    print(f"Tool: Delegating SEO Metadata to Specialist Model (Anthropic)...")
    
    prompt = f"""
    You are a World-Class SEO Strategist and Copywriter.
    
    INPUT CONTEXT:
    Topic: "{topic}"
    Article Content (HTML):
    {article_html[:15000]} # Truncate to avoid context limits
    
    TASK:
    Generate highly optimized, click-worthy metadata for this article.
    
    REQUIREMENTS:
    1. **title**: A catchy H1 title (Max 70 chars). different from the input if needed for better CTR.
    2. **meta_title**: The SEO Title tag (Max 60 chars). Must include primary keyword near the front.
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
    {{
        "title": "...",
        "meta_title": "...",
        "meta_description": "...",
        "custom_excerpt": "...",
        "featured_image_prompt": "...",
        "tags": ["...", "..."]
    }}
    """
    
    try:
        # 1. Use the Global Specialist Brain (Anthropic)
        
        # 2. Generate Content using the Universal Method
        # The adapter returns a response object where .text is the content
        response = specialist_model.generate_content(prompt, generation_config={"temperature": 0.7})
        content = response.text.strip()
        
        # 3. Use the Unified Super-Listener
        final_json = extract_json(content)
        if final_json:
            return final_json
        
        raise ValueError("Unified parser failed to extract valid metadata.")

    except Exception as e:
        print(f"⚠️ Specialist Anthropic Model Failed: {e}")
        # Return a safe fallback schema
        return {
            "title": topic.replace("Draft a pSEO article about ", "").strip('"'),
            "meta_title": topic[:60],
            "meta_description": f"Deep dive into {topic}.",
            "custom_excerpt": f"Analysis of {topic}.",
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
        # Toggle code block state
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
    3. **Code Blocks**: For any code snippets, use markdown fenced code blocks:
       - Use triple backticks with language identifier: ```language
       - Example: ```python\ndef hello(): return \"world\"\n```
       - Supported languages: javascript, python, java, go, rust, typescript, html, css, json, yaml, bash, sql, etc.
    """
    
    try:
        response = active_model.generate_content(prompt, generation_config={"temperature": 0.4})
        text = response.text
        
        # Check for Mock Error Response from UnifiedModel
        if "I encountered a safety limit" in text or "unable to generate" in text:
             raise ValueError("Model returned Error Mock")

        final_text = ensure_slack_compatibility(text.strip())
        
        # Simple intent detection for metadata consistency
        intent = "SIMPLE_QUESTION"
        if "```" in final_text: intent = "TECHNICAL_EXPLANATION"
        
        return {
            "text": final_text,
            "intent": intent,
            "directive": ""
        }
    except Exception as e:
        print(f"⚠️ Primary Model Failed ({active_model.model_name}): {e}")
        
        if specialist_model:
            print(f"🔄 FAILOVER: Switching to Specialist Model (Anthropic)...")
            try:
                # Ensure Anthropic Key is present
                if not os.environ.get("ANTHROPIC_API_KEY") and ANTHROPIC_API_KEY:
                     os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY
                
                fallback_resp = specialist_model.generate_content(prompt, generation_config={"temperature": 0.4})
                final_text = ensure_slack_compatibility(fallback_resp.text.strip())
                return {"text": final_text, "intent": "DIRECT_ANSWER_FALLBACK", "directive": ""}
            except Exception as e2:
                print(f"❌ Specialist Failover also failed: {e2}")
                return {"text": "I am currently experiencing high traffic on all neural pathways (429). Please try again in a minute.", "intent": "ERROR", "directive": ""}
        else:
             return {"text": "Service is busy (429). Please try again later.", "intent": "ERROR", "directive": ""}


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
    You are the 'Sonnet & Prose' Senior Content Strategist. 
    AUDIENCE: {audience_context}
    PRINCIPLE: provide technical depth (Prose) wrapped in human-centric perspective (Sonnet).
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

    try:
        # Use active_model (Dynamic routing) for the final synthesis
        response = active_model.generate_content(prompt, generation_config={"temperature": target_temp})
        
        # Enforce spacing on the output
        final_text = ensure_slack_compatibility(response.text.strip())
        
        return {
            "text": final_text,
            "intent": research_intent,
            "directive": formatting_directive
        }
    except Exception as e:
        print(f"Synthesis Error: {e}")
        # Final fallback
        fallback = active_model.generate_content(prompt).text.strip()
        return {"text": fallback, "intent": research_intent, "directive": formatting_directive}

# 14. The Dedicated pSEO Article Generator (High Trust & Transparency)
def generate_pseo_article(topic, context, history="", history_events=None):
    global unimodel
    print(f"Tool: Generating pSEO Article for '{topic}'")
    
    # AUDIENCE DETECTION: Search history for tone instructions
    audience_context = detect_audience_context(history)

    # 1. Determine System Instruction (Grounding Logic)
    is_grounded = any("GROUNDING_CONTENT" in str(c) for c in context)
    
    # 2. History Check for Repurposing (Adherence)
    # Check if a structural draft exists in history (proposal_type="deep_dive")
    has_existing_draft = False
    is_repurpose_command = any(kw in topic.lower() for kw in ["dump", "repurpose", "convert", "use existing", "use the draft", "publish"])
    has_markdown_structure = "## " in str(history) or "Title:" in str(history)
    
    if history_events:
        has_existing_draft = any(e.get('proposal_type') in ['deep_dive', 'pseo_article'] for e in history_events)
    else:
        has_existing_draft = False

    # Combined Check: If explicit command OR strong artifact signals
    if is_repurpose_command and has_markdown_structure:
        has_existing_draft = True
    elif "Deep-Dive Complete" in str(history) or "Here is the draft" in str(history):
        has_existing_draft = True
    
    if has_existing_draft:
        print("  + Detected existing draft (Structural Check). Switching to REPURPOSE MODE.")
        system_instruction = """
        You are a Content Formatter. 
        TASK: Convert the EXISTING DRAFT found in the 'Conversation History' into the specific HTML structure below.
        CRITICAL: Do not rewrite the substance. Keep the original content, analogies, and tone. Just apply the requested HTML tags and organization.
        """
    else:
        print("  + No draft detected. Switching to AUTHOR MODE.")
        if is_grounded:
            system_instruction = "You are a Clear Technical Communicator. Base the article PRIMARILY on the provided 'Research Context'."
        else:
            system_instruction = "You are a Clear Technical Communicator. Use the provided context to write a high-authority article."

    if has_existing_draft:
        tone_instruction = "Tone: Maintain the exact professional tone and depth of the existing draft."
    else:
        tone_instruction = "Tone: Clear, engaging, professional English. Use analogies to explain complex ideas, but maintain technical accuracy. Rule: Define technical acronyms on first use (e.g., 'Hardware Security Module (HSM)')."

    # 3. Strict System Prompt
    prompt = f"""
    {system_instruction}
    
    AUDIENCE: 
    {audience_context}
    
    {tone_instruction}
    
    STRUCTURE REQUIREMENTS (HTML Format):
    1.  **<section class="intro">**: A compelling, simple hook connecting the topic to the reader.
    2.  **<section class="body">**: Detailed analysis using <h2> and <h3> tags. Use specific data/facts from the context.
    3.  **Code Blocks**: When including code snippets, use Ghost's markdown fenced code block format:
        - Use triple backticks with language identifier: ```language
        - Example for JavaScript: ```javascript\\nfunction example() {{ return true; }}\\n```
        - Example for Python: ```python\\ndef hello(): return "world"\\n```
        - Supported languages: javascript, python, java, go, rust, typescript, html, css, json, yaml, bash, sql, etc.
        - CRITICAL: Always specify the language for proper Prism.js syntax highlighting
        - The language identifier must be lowercase and immediately follow the opening backticks
    4.  **<section class="methodology">**: A TRANSPARENCY FOOTER. 
        - Must be titled "## Methodology & Sources".
        - Explicitly list the domains/articles found in the research.
        - Explain *why* you selected them.
    
    CRITICAL RULES:
    - Do NOT hallucinate.
    - Return ONLY the HTML content (no ```html``` markdown wrappers).
    - Do NOT include labels like 'Intro', 'Body', or 'Methodology' as text.
    - For code snippets, use markdown fenced code blocks (```language) NOT HTML <pre><code> tags.
    - Always specify the programming language for syntax highlighting (e.g., ```javascript, ```python).
    - Ensure it is ready for Ghost CMS (Ghost will convert markdown code blocks to HTML automatically).

    CONTEXTUAL DATA:
    Conversation History: 
    {history}
    
    Current Topic: "{topic}"
    
    Research Context (Grounding Data):
    {context}
    
    Article Draft:
    """
    
    
    return unimodel.generate_content(prompt, generation_config={"temperature": 0.3}).text.strip()

# 14.5 The Recursive Deep-Dive Generator (Dynamic Room-by-Room Construction)
def generate_deep_dive_article(topic, context, history="", history_events=None, target_length=1500):
    global unimodel
    print(f"Tool: Initiating Recursive Deep-Dive for '{topic}' (Target: {target_length} words)")
    
    # CLAMP: Prevent unbounded generation that crashes SSL/Webhooks
    target_length = min(3500, max(500, target_length))
    
    is_grounded = any("GROUNDING_CONTENT" in str(c) for c in context)
    audience_context = detect_audience_context(history)
    
    # Calculate architecture: Dynamic Range (3-8 sections)
    # logic: We give the Blueprint Architect the freedom to choose, but provide a heuristic target.
    # We remove the rigid calc: num_sections = max(3, target_length // 300)
    words_per_section = 300 # Baseline heuristic for prompting
    
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
                "writing_instruction": "Specific instruction for the writer (e.g., 'Tell a story about X', 'List 5 facts about Y'). Ensure it aligns with the detected style."
            }} 
        ]
    }}
    """
    
    try:
        blueprint_raw = unimodel.generate_content(blueprint_prompt).text.strip()
        if "```json" in blueprint_raw: blueprint_raw = blueprint_raw.split("```json")[1].split("```")[0].strip()
        blueprint = json.loads(blueprint_raw)
        title = blueprint.get("title", f"The Analysis of {topic}")
        sections = blueprint.get("sections", [])
    except Exception as e:
        print(f"Blueprint Error: {e}")
        # Fallback to pSEO if blueprinting fails
        return generate_pseo_article(topic, context, history, history_events=history_events)

    # PHASE 2: Recursive Room Building (The Writer)
    article_parts = [f"<h1>{title}</h1>"]
    
    # Intro
    print(f"  + Drafting Hook Intro...")
    intro_prompt = f"""
    Write a compelling {words_per_section}-word intro for: '{title}'.
    
    AUDIENCE: {audience_context}
    STYLE: Clear, engaging, professional English. Use analogies for complex ideas.
    GOAL: {topic}
    """
    article_parts.append(f'<section class="intro">\n{unimodel.generate_content(intro_prompt).text.strip()}\n</section>')
    
    for i, section in enumerate(sections):
        print(f"  + Constructing Room {i+1}/{len(sections)}: {section['title']}...")
        
        # CONTEXT STITCHING
        last_plain = strip_html_tags(article_parts[-1])
        prev_context = ". ".join(last_plain.split(". ")[-3:]) if ". " in last_plain else last_plain[-150:]
        
        instruction = section.get('writing_instruction', 'Explain this concept clearly.')
        
        room_prompt = f"""
        Write a {words_per_section}-word section for: "{title}".
        
        CHAPTER: {section['title']}
        INSTRUCTION: {instruction}
        TRANSITION FROM: "...{prev_context}"
        
        STRICT LIMIT: Stay under {words_per_section + 50} words.
        
        GLOBAL CONSTRAINT: Ensure professional but accessible English. Avoid jargon, but do not dumb down technical concepts. Define technical acronyms on first use (e.g., 'Payment Card Industry (PCI)').
        
        GROUNDING DATA: {context if is_grounded else "Internal Knowledge base"}
        
        OUTPUT: Start with <h2>{section['title']}</h2> then the content.
        """
        room_content = unimodel.generate_content(room_prompt).text.strip()
        article_parts.append(f'<section class="body-part">\n{room_content}\n</section>')

    # Conclusion
    print("  + Drafting Final Reflection...")
    conc_prompt = f"Write a {words_per_section // 2}-word concluding thought (Sonnet style) for '{title}'. Audience: {audience_context}. Summary of key impact."
    article_parts.append(f'<section class="conclusion">\n<h2>Final Reflection</h2>\n{unimodel.generate_content(conc_prompt).text.strip()}\n</section>')

    # PHASE 3: Methodology
    print("  + Finalizing Methodology & Transparency...")
    if is_grounded:
        found_urls = list(set(re.findall(r'https?://[^\s<>"]+', str(context))))
        source_text = ", ".join(found_urls[:5]) if found_urls else "Verified research sources."
    else:
        source_text = "AEO Synthesis from Knowledge Base."

    # Calculate interim word count for the footer
    current_content = " ".join(article_parts)
    est_words = len(current_content.split())

    methodology_text = f"""
    <section class="methodology">
    <h2>Methodology & Transparency</h2>
    This {est_words}-word analysis was recursively architected for the 'Sonnet & Prose' publication.
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
    response = unimodel.generate_content(prompt)
    return extract_json(response.text)

#16. The Proposal Critic and Refiner
def critique_proposal(topic, current_proposal):
    global unimodel
    prompt = f"Review proposal for '{topic}': {json.dumps(current_proposal, indent=2)}. If excellent, respond: APPROVED. Else, provide concise feedback to improve 'Then vs Now' contrast."
    return unimodel.generate_content(prompt).text.strip()

#17. The Proposal Refiner
def refine_proposal(topic, current_proposal, critique):
    global unimodel
    prompt = f"""
    REWRITE proposal. Topic: {topic}. Draft: {json.dumps(current_proposal)}. Critique: {critique}. 
    Preserve keys: 'then_concept', 'now_concept', 'link'. Ensure JSON format.
    """
    return extract_json(unimodel.generate_content(prompt).text)

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
        objections_raw = specialist_model.generate_content(objection_prompt).text.strip()
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
    
    
    brief_raw = specialist_model.generate_content(brief_prompt).text.strip()
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

    if unimodel is None:
        unimodel = UnifiedModel(MODEL_PROVIDER, MODEL_NAME)
    
    if flash_model is None:
        # Initialize Vertex AI explicitly for the Flash Model (Triage)
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        flash_model = GenerativeModel(FLASH_MODEL_NAME, safety_settings=safety_settings)

    if research_model is None:
        # Initialize Vertex AI explicitly for the Research Model (High Context)
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        research_model = UnifiedModel("vertex_ai", RESEARCH_MODEL_NAME)
        
    if specialist_model is None:
        # Initialize Anthropic Specialist explicitly
        specialist_model = UnifiedModel(provider="anthropic", model_name="claude-sonnet-4-5")
        
    if db is None:
        db = firestore.Client(project=PROJECT_ID)

    req = request.get_json(silent=True)
    if isinstance(req, list): req = req[0] # Add safety for n8n list payloads
    print(f"DEBUG: Story Worker received payload keys: {list(req.keys()) if req else 'None'}")
    
    # --- IDEMPOTENCY CHECK (Prevent Double-Trigger) ---
    # Construct a unique event ID from the Slack Timestamp (which is unique to the message)
    # Use request-provided ID if available, otherwise fallback to signature
    slack_context = req.get('slack_context', {})
    unique_event_id = slack_context.get('ts') or req.get('event_ts') or req.get('client_msg_id')
    
    if unique_event_id:
        # Check if we've already seen this event ID in the last 5 minutes
        # We store processed events in a separate 'processed_requests' collection or within the session
        # For simplicity and speed, we'll check a centralized deduplication log
        dedup_ref = db.collection('processed_events').document(str(unique_event_id))
        doc = dedup_ref.get()
        if doc.exists:
            # Check timestamp to allow re-runs after 5 minutes (for testing/dev)
            data = doc.to_dict()
            last_run = data.get('timestamp')
            
            # Simple timezone-aware check
            if last_run:
                # Firestore timestamps come back as datetime with tz
                cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=5)
                if last_run > cutoff:
                    print(f"🔒 IDEMPOTENCY: Skipping duplicate event {unique_event_id} (Processed < 5m ago)")
                    return "Event already processed", 200
                else:
                    print(f"🔄 IDEMPOTENCY: Reprocessing old event {unique_event_id} (Age > 5m)")
            else:
                 pass # No timestamp? Overwrite it.
        
        # Mark as pending immediately (TTL handled by expiration policy or ignored for now)
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
        
        history_events = [doc.to_dict() for doc in query.stream()]


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
    if history_events:
        recovered_count = 0
        for event in history_events:
            if event.get('event_type') == 'code_analysis' and event.get('analysis_result'):
                code_analysis_snippets.append(event['analysis_result'])
                recovered_count += 1
        if recovered_count > 0:
            print(f"TELEMETRY: Recovered {recovered_count} code analysis results from thread history.")

    # 2. Analyze New Files
    if code_files:
        print(f"TELEMETRY: Code File Analysis Mode detected ({len(code_files)} files) - Pre-Triage Execution")
        
        for file_info in code_files:
            file_name = file_info.get('name', 'unknown_file')
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
                *   *Triggers:* "Summarize our progress", "Create an outline from our discussion", "Reformat these points as bullets", "Make it more professional", "Convert this into a brief".
                *   *CRITICAL:* If the request is purely about structural formatting of existing knowledge, select this.

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
        intent = flash_model.generate_content(triage_prompt).text.strip()
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
            reply_text = ensure_slack_compatibility(unimodel.generate_content(social_prompt).text.strip())
            
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
            requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={"session_id": session_id, "type": "social", "message": reply_text, "channel_id": slack_context['channel'], "thread_ts": slack_context['ts'], "is_initial_post": is_initial_post}, verify=True, timeout=10)
            return jsonify({"msg": "Social reply sent"}), 200

        # === PATH B: OPERATIONAL REFORMAT / FAST FOLLOW-UP (Fast Exit) ===
        elif intent == "OPERATIONAL_REFORMAT":
            # GROUNDING GUARD: We only "Fast-Exit" if we actually have history to reformat.
            # If this is the first turn and the topic is new, we MUST research.
            has_substantive_history = len(history_events) > 0 or len(sanitized_topic) > 100 
            
            if has_substantive_history:
                print(f"Executing Operational Reformat (Fast Exit) for command: {sanitized_topic[:50]}")
                # Use natural answer generation with full context but NO research
                answer_data = generate_natural_answer(sanitized_topic, "COMPLETE_CONTEXT_IN_HISTORY", history=history_text)
                answer_text = answer_data['text']
                research_intent = answer_data['intent']
                
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
            r_result = find_trending_keywords(sanitized_topic, history_context=history_text, session_id=session_id, images=images, mission_topic=original_topic, session_metadata=session_data, initial_context=code_analysis_context)
            
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
            r_result = find_trending_keywords(sanitized_topic, history_context=history_text, session_id=session_id, images=images, mission_topic=original_topic, session_metadata=session_data, initial_context=code_analysis_context)
            research_data = r_result
            # Unified Merging: Base context (Code Analysis) + Novel Findings (r_result)
            research_data['context'] = code_analysis_context + r_result['context']
            # Ensure Key Persistence
            if "research_intent" not in research_data: research_data["research_intent"] = "KEYWORD_SENSORY"
        
        # --- METADATA EXTRACTION (Thread Memory Optimization) ---
        final_geo = research_data.get("detected_geo", session_data.get("detected_geo", "Global"))
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
            
            # 2. Check for broad refusal keywords
            refusal_keywords = ["cannot fulfill", "illegal", "refuse", "harm", "violence", "sensitive", "prohibited", "violate"]
            if any(kw in meta_lower for kw in refusal_keywords):
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
        if intent in ["DIRECT_ANSWER", "SIMPLE_QUESTION"] or str(intent).startswith("FORMAT_"):
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
            target_count = extract_target_word_count(sanitized_topic)
            print(f"TELEMETRY: Executing Dynamic Deep-Dive Recursive Expansion (Target: {target_count})...")
            # PASS THE sanitzied_topic as the DIRECTIVE to prioritize feedback
            article_html = generate_deep_dive_article(sanitized_topic, research_data['context'], history=history_text, history_events=history_events, target_length=target_count)
            
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
                # Use sanitized_topic (Feedback) for priority cluster subject
                cluster_data = generate_topic_cluster(sanitized_topic, research_data['context'], history=history_text, is_initial=is_initial_post)
                if not cluster_data: raise ValueError("Failed to parse cluster JSON.")
                
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
                approval_id = f"approval_{uuid.uuid4().hex[:8]}"
                new_events.append({"event_type": "loop_draft", "iteration": 0, "proposal_data": current_proposal})

                loop_count = 0
                while loop_count < MAX_LOOP_ITERATIONS:
                    critique = critique_proposal(original_topic, current_proposal)
                    if "APPROVED" in critique.upper(): break
                    try: current_proposal = refine_proposal(original_topic, current_proposal, critique)
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
                article_html = generate_pseo_article(sanitized_topic, research_data['context'], history=history_text, history_events=history_events)
                
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
                
                # 5. Send STRUCTURED DATA to N8N
                requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={
                    "session_id": session_id, 
                    "type": "pseo_draft", 
                    "payload": {
                        "title": seo_data.get("title"),
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
    global db, unimodel
    if db is None: db = firestore.Client(project=PROJECT_ID)
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")

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
        mcp_endpoint = "https://mcp-sensory-server-gjkm6kxlea-uc.a.run.app/messages"
        
        mcp_payload = {
            "method": "tools/call",
            "params": {
                "name": "scrape_article",
                "arguments": {"url": doc_url}
            }
        }
        
        mcp_response = requests.post(mcp_endpoint, json=mcp_payload, timeout=90) # Extended timeout for Gemini OCR
        
        if mcp_response.status_code != 200:
            return jsonify({"error": f"MCP Hub Error ({mcp_response.status_code})", "details": mcp_response.text}), 500
            
        mcp_data = mcp_response.json()
        pdf_content = mcp_data.get("result", {}).get("content", [{}])[0].get("text", "")
        
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
