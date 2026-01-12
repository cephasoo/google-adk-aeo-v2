# --- /worker-story/main.py ---
import functions_framework
from flask import jsonify
import vertexai
import litellm
from litellm import completion
from vertexai.generative_models import GenerativeModel, HarmCategory, HarmBlockThreshold
import google.cloud.logging
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
from google.cloud import firestore

# --- Configuration ---
PROJECT_ID = os.environ.get("PROJECT_ID")
LOCATION = os.environ.get("LOCATION")
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-2.5-pro") 
MODEL_PROVIDER = os.environ.get("MODEL_PROVIDER", "vertex_ai")
FLASH_MODEL_NAME = os.environ.get("FLASH_MODEL_NAME", "gemini-2.5-flash-lite")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
N8N_PROPOSAL_WEBHOOK_URL = os.environ.get("N8N_PROPOSAL_WEBHOOK_URL") 
MAX_LOOP_ITERATIONS = 2

# --- Safety Configuration (ADK/RAI Compliant) ---
safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
}

# --- Global Clients ---
model = None
flash_model = None
search_api_key = None
db = None
mcp_client = None

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

    def call(self, tool_name, arguments):
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
                return f"Error: MCP Server returned {response.status_code}"
            
            data = response.json()
            # Extract text from MCP response format
            content = data.get("result", {}).get("content", [])
            if content:
                return content[0].get("text", "No text returned.")
            return "Empty response from tool."
        except Exception as e:
            return f"MCP Error: {str(e)}"

def get_mcp_client():
    global mcp_client
    if mcp_client is None:
        mcp_client = RemoteTools(os.environ.get("MCP_SERVER_URL", "http://localhost:8080"))
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

    def generate_content(self, prompt, generation_config=None):
        """
        Universal generation function with safety catching.
        """
        # PATH A: Native Vertex AI
        if self.provider == "vertex_ai":
            try:
                response = self._native_model.generate_content(prompt, generation_config=generation_config)
                
                # Robust Safety Check: Some SDK versions throw exceptions, others return empty candidates
                if not response.candidates or response.candidates[0].finish_reason == 3: # 3 = SAFETY
                     raise ValueError("Safety Block via FinishReason")
                
                return response
            except Exception as e:
                # Catch both the ValueError we raised above AND any SDK-specific exceptions
                print(f"⚠️ Vertex AI Safety/SDK Block: {e}")
                log_safety_event("safety_block", {"prompt": prompt, "error": str(e)})
                
                # Create a Fake Vertex Response Object for fallback
                class MockResponse:
                    def __init__(self, content):
                        self.text = content
                
                return MockResponse("I'm sorry, I cannot fulfill this request as it conflicts with my safety guardrails. Let's try rephrasing.")

        # PATH B: Universal Route (OpenAI / Anthropic via LiteLLM)
        else:
            print(f"🔄 Switching to {self.model_name} via LiteLLM ({self.provider})...", flush=True)
            
            # Inject Keys for LiteLLM
            if self.provider == "anthropic" and ANTHROPIC_API_KEY:
                os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY
            elif self.provider == "openai" and OPENAI_API_KEY:
                os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
            
            # Normalize Parameters (Temp default 0.7)
            temp = generation_config.get('temperature', 0.7) if generation_config else 0.7
            
            try:
                # The Universal Call
                response = completion(
                    model=self.model_name, 
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temp,
                    num_retries=3,
                    timeout=60  # Increased for complex recursive drafting
                )
                
                # Create a Fake Vertex Response Object
                class MockResponse:
                    def __init__(self, content):
                        self.text = content
                
                return MockResponse(response.choices[0].message.content)
                
            except Exception as e:
                print(f"❌ LiteLLM Error: {e}", flush=True)
                raise e

# --- Safety Utils ---
def scrub_pii(text):
    """Simple PII scrubbing for memory and logs."""
    if not isinstance(text, str): return text
    # Mask emails
    text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[EMAIL_MASKED]', text)
    # Mask common phone formats
    text = re.sub(r'\+?\d{1,3}[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}', '[PHONE_MASKED]', text)
    return text

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
# --- Input Santitation Utility ---
def sanitize_input_text(text):
    """Strips common markdown/Slack formatting from the start and end of a string."""
    if not isinstance(text, str):
        return ""
    # Strips leading/trailing whitespace, asterisks (bold), underscores (italics), and tildes (strikethrough)
    return text.strip().strip('*_~')

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
    
    core_topic = flash_model.generate_content(extraction_prompt).text.strip()

    print(f"Distilled Core Topic: '{core_topic}'")
    return core_topic

def extract_target_word_count(user_prompt):
    """
    Finds a numeric word count in the prompt (e.g., "1500 words").
    Defaults to 1500 if none found.
    """
    matches = re.findall(r'(\d+)\+?[\s-]*words?', user_prompt, re.IGNORECASE)
    if matches:
        return int(matches[0])
    return 1500

def strip_html_tags(text):
    """Removes HTML tags for clean context stitching and plain-text views."""
    return re.sub(r'<[^>]+>', '', text).strip()

def convert_html_to_markdown(html):
    """Converts architectural HTML into Slack-friendly Markdown."""
    # 1. Headers
    text = re.sub(r'<h2>(.*?)</h2>', r'*\1*\n', html)
    text = re.sub(r'<h3>(.*?)</h3>', r'_\1_\n', text)
    
    # 2. Sections to separators
    text = text.replace('</section>', '\n---\n')
    text = re.sub(r'<section[^>]*>', '', text)
    
    # 3. Lists (Simple conversion)
    text = text.replace('<ul>', '').replace('</ul>', '')
    text = text.replace('<li>', '• ').replace('</li>', '\n')
    
    # 4. Final Strip
    text = re.sub(r'<[^>]+>', '', text)
    
    # Clean up excess newlines
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
        # Use flash_model for speed/cost since this is a simple extraction
        entity = flash_model.generate_content(prompt).text.strip().replace('"', '').replace("'", "")
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
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match: raise ValueError(f"No JSON found in text: {text[:200]}...")
    return json.loads(match.group(0))

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

# 10. The Router (Updated with "Double-Tap" Analysis)
def find_trending_keywords(raw_topic, history_context="", session_id=None):
    """
    An intelligent meta-tool that routes a query to the best specialized search tool.
    Uses 'raw_topic' for routing to allow for "NONE" selection on conversational turns.
    
    UPGRADE: The 'ANALYSIS' tool now performs a "Double-Tap":
    1. Fetches Trend Stats (Quant)
    2. Fetches Top News (Qual)
    This prevents "blind" statistical answers (e.g., seeing a spike but not knowing why).
    """
    global model, flash_model
    print(f"Tool: find_trending_keywords (Sensory Array) for RAW topic: '{raw_topic}'")
    
    tool_logs = []
    context_snippets = []
    
    # 1. Internal RAG (We can use raw_topic here; semantic search handles sentences well)
    internal_context = search_long_term_memory(raw_topic, session_id) if session_id else "No session_id provided for memory search."

    # Prepare RAG text for the prompt
    rag_text = "No relevant long-term memories found."
    if internal_context:
        # Flatten the list of strings into a single block of text
        rag_content_str = "\n".join([str(item) for item in internal_context])
        rag_text = rag_content_str
        
        context_snippets.append(internal_context)
        tool_logs.append({"event_type": "tool_call", "tool_name": "internal_knowledge_retrieval", "status": "success"})

    # 2. Automated Signal Detection (MCP Refactored)
    print("Sensory Router: Detecting Signals (Geo & Intent) via MCP Hub...")
    detected_geo = mcp_detect_geo(raw_topic, history=history_context)
    research_intent = mcp_detect_intent(raw_topic)
    print(f"  -> Signal Detected | Geo: '{detected_geo}' | Intent: '{research_intent}'")
    
    # 3. Sensory Array Router (Decides based on RAW input)
    tool_choice_prompt = f"""
    Analyze the user's query to select the most appropriate research tool(s).

    CONVERSATION HISTORY:
    {history_context}

    LONG-TERM MEMORY (RAG Results):
    {rag_text}

    USER'S CORE QUERY: '{raw_topic}'
    
    ### DETECTED CONTEXT (MCP Hub Signals):
    - PRIMARY GEO TARGET: '{detected_geo}' (System will default to '{detected_geo}')
    - REQUESTED INTENT: '{research_intent}'

    ### DECISION PROTOCOL (CHECK IN ORDER):
    1. **CHECK HISTORY (Priority #1):**
        - If the user is asking for information (or a REFORMATTING like an OUTLINE) that CAN BE GATHERED from the "LONG-TERM MEMORY" or "CONVERSATION HISTORY", select **NONE**.
        - If the history contains the FACTUAL CONTENT needed but the user wants a new structure (e.g. "Create an outline for the above"), select **NONE**.
        - **INTERPRETIVE EXCEPTION:** If the user asks for **REASONS**, **DRIVERS**, **CAUSES**, or **"WHY"** (and the history only contains stats/numbers), **DO NOT** select NONE. Proceed to research.

    2. **CHECK CREATIVE INTENT (Priority #2):**
        - If the user asks you to **DRAFT**, **CREATE**, or **WRITE** content for an entity that is clearly **INVENTED**, **FICTIONAL**, or defined in history, select **NONE**.

    3. **NEW RESEARCH (Priority #3):**
        - If the user wants you to gather NEW information not in history, use a tool below.
        - If the user wants you to explain a trend (and the history contains only numbers/stats but not the *explanation*), select a tool below.

    ### AVAILABLE TOOLS:
    - WEB: Use this whenever the request requires **Factual Discovery** or **External Grounding** not available in history.
        *   Examples: "Research X", "Find data on Y", "Ground this in technical specifics", "Use latest signals".
    - IMAGES: For visual inspiration.
    - VIDEOS: For tutorials/clips.
    - SCHOLAR: For academic research.
    - TRENDS: For viral awareness. Select ONLY for measuring "What is trending" or "Viral topics" in a specific region.
    - ANALYSIS: For historical trajectory. Select ONLY for "Interest over time" or "Growth metrics".
    - SIMPLE: General web searches for broad definitions.
    - NONE: Select this for **Composition**, **Formatting**, or **Incremental Refinement** using information already established in the conversation.
        *   Examples: "Write a draft", "Create an outline", "Summarize our notes", "Refine the tone", "Include more detail on [Topic A]", write about a hypothetical or user-defined entity.

    ### DECISION REASONING:
    - If a request combines **Drafting** with **Research Intent** (e.g. "Draft a deep-dive based on new research"), the priority is to select **WEB** + any other necessary tools.
    - If a request is purely about **Execution/Form** based on existing knowledge, select **NONE**.

    Respond ONLY with the tool selection(s), comma-separated (e.g., "WEB, ANALYSIS" or "IMAGES" or "NONE").
    Do not add country codes; the system handles locations via the MCP signals.
    """
    # 4. Robust Parsing Logic
    try:
        raw_response = flash_model.generate_content(tool_choice_prompt).text.strip().upper()
        clean_response = raw_response.replace("*", "").replace("`", "").replace("'", "").replace('"', "").rstrip(".")
        
        selected_tools = [t.strip() for t in clean_response.split(',')]
        print(f"Sensory Array decided on tools: {selected_tools} (Detected Geo: {detected_geo})")
    except Exception as e:
        print(f"Router Parse Error: {e}. Defaulting to SIMPLE.")
        selected_tools = ["SIMPLE"]

    if any("NONE" in choice for choice in selected_tools):
        return {"context": context_snippets, "tool_logs": tool_logs, "research_intent": research_intent}

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
    
    # --- MULTI-SIGNAL EXECUTION LOOP ---
    # We split detected_geo (e.g., "NG, KE") and iterate.
    # To avoid combinatorial explosion, we limit to 3 geos max.
    target_geos = [g.strip() for g in detected_geo.split(',')][:3]
    
    for geo_code in target_geos:
        print(f"--- Processing Region: '{geo_code}' ---")
        
        # Iterate through each selected tool
        for tool_selection in selected_tools:
            choice = tool_selection.split(':')[0].strip() # Handle legacy format just in case
            print(f"Executing Sensory Tool: '{choice}' (Geo: '{geo_code}')")

            research_context = None
            tool_name = "unknown"

            # --- 1. VIRAL DISCOVERY ---
            if "TRENDS" in choice:
                research_context = search_google_trends(geo=geo_code)
                tool_name = f"serpapi_trends_search_{geo_code}"

            # --- 2. TREND ANALYSIS ---
            elif "ANALYSIS" in choice:
                search_query = distilled_trend_term
                print(f"  + Fetching Quantitative Trend Stats for '{search_query}' in '{geo_code}'...")
                stats_context = analyze_trend_history(search_query, geo=geo_code)
                
                # C. TAP 2: Qualitative Data
                has_web_tool = any(t in ["WEB", "SIMPLE"] for t in selected_tools)
                try:
                    if not has_web_tool:
                        print(f"  + Fetching Qualitative News Context for '{search_query}' in '{geo_code}' via MCP...")
                        news_text = get_mcp_client().call("google_news_search", {"query": search_query, "geo": geo_code})
                        research_context = f"[Region: {geo_code} Stats]:\n{stats_context}\n{news_text}"
                    else:
                        print(f"  - Skipping Supplemental News context (WEB search will provide it) for '{geo_code}'.")
                        research_context = f"[Region: {geo_code} Stats]:\n{stats_context}"
                except Exception as news_err: 
                    print(f"  ! News fetch error for '{geo_code}': {news_err}")
                    research_context = stats_context
                tool_name = f"serpapi_trend_analysis_{geo_code}"

            # --- 3. STANDARD SEARCH TOOLS ---
            elif choice in ["WEB", "IMAGES", "VIDEOS", "SCHOLAR", "SIMPLE"]:
                search_query = distilled_seo_query
                if choice == "WEB":
                    results = search_google_web(search_query)
                    research_context = f"[Region: {geo_code} Results]:\n{results}"
                    tool_name = f"serpapi_web_search_{geo_code}"
                elif choice == "IMAGES":
                    research_context = search_google_images(search_query)
                    tool_name = "serpapi_image_search"
                elif choice == "VIDEOS":
                    research_context = search_google_videos(search_query)
                    tool_name = "serpapi_video_search"
                elif choice == "SCHOLAR":
                    research_context = search_google_scholar(search_query)
                    tool_name = "serpapi_scholar_search"
                elif choice == "SIMPLE":
                    pass 
                
            # --- 4. FALLBACK / SIMPLE SEARCH ---
            if not research_context and choice not in ["IMAGES", "VIDEOS", "SCHOLAR"]:
                fallback_query = distilled_seo_query or distilled_trend_term or raw_topic
                print(f"  ? Primary research failed for '{geo_code}'. Initiating fallback.")
                research_context = google_simple_search(fallback_query)
                tool_name = "google_simple_search"

            if research_context:
                context_snippets.append(research_context)
                tool_logs.append({"event_type": "tool_call", "tool_name": tool_name, "input": raw_topic, "status": "success"})

    return { "context": context_snippets, "tool_logs": tool_logs, "research_intent": research_intent }


#11. The Topic Cluster Generator
def generate_topic_cluster(topic, context, history=""):
    global model
    print("Tool: Topic Cluster Generator")
    prompt = f"""
    You are a master Content Strategist.
    
    Conversation History: {history}.
    
    User wants pillar page on "{topic}".
    
    CONTEXT: {context}

    CRITICAL: Respond with a JSON object following this exact schema:
    {{
    
      "pillar_page_title": "[Topic]",

      "clusters": [
        {{ "cluster_title": "...", "sub_topics": ["...", "..."] }}
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
        # 1. Initialize the Specialist Brain via your Adapter
        # This ensures we use the exact same logic/keys as the rest of the app
        specialist = UnifiedModel(provider="anthropic", model_name="claude-sonnet-4-5")
        
        # 2. Generate Content using the Universal Method
        # The adapter returns a response object where .text is the content
        response = specialist.generate_content(prompt, generation_config={"temperature": 0.7})
        content = response.text.strip()
        
        # 3. Clean and Parse JSON (Standard Logic)
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[0].strip()
            
        return json.loads(content)

    except Exception as e:
        print(f"⚠️ Specialist Anthropic Model Failed: {e}")
        # Fallback to Basic Generation using the DEFAULT global model if Specialist fails
        # or just return static fallback
        return {
            "title": topic.replace("Draft a pSEO article about ", "").strip('"'),
            "meta_title": topic[:60],
            "meta_description": "Read our latest analysis on this topic.",
            "custom_excerpt": "Click to read more.",
            "featured_image_prompt": f"A futuristic abstract representation of {topic}",
            "tags": ["AI", "Tech"]
        }

#13. The Comprehensive Answer Generator (AEO-AWARE CONTENT STRATEGIST)
def generate_comprehensive_answer(topic, context, history="", intent_hint="DIRECT_ANSWER"):
    """
    Standardizes the logic for generating a direct answer.
    Uses 'detect_research_intent' signal to adjust formatting (Tables/Lists).
    """
    global model
    print(f"Tool: generate_comprehensive_answer for topic: '{topic}'")
    is_grounded = any("GROUNDING_CONTENT" in str(c) for c in context)
    
    if is_grounded:
        print(f"  -> Context is GROUNDED. Enforcing STRICT ACTION MODE.")
    else:
        print(f"  -> Context is NOT grounded. Using strategic dialogue mode.")
    
    # UNIFIED PERSONA: The AEO-Aware Content Strategist
    persona_instruction = """
    You are the 'Sonnet & Prose' Senior Content Strategist. 
    You excel at Answer Engine Optimization (AEO) and Conversational Synthesis.
    
    CORE OPERATING PRINCIPLES:
    1. **Dual-Intent Synthesis**: Connect external 'Research Context' with the specific artifacts and nuances found in the SLACK HISTORY. 
    2. **STRICT ACTION MODE**: If research data is provided (GROUNDING_CONTENT), **DO NOT ASK FOR PERMISSION**. Provide the data immediately in the requested format.
    3. **Contextual Formatting (AEO vs. Dialogue)**: 
       - **Production/Draft Mode**: If the user is asking for a **DRAFT**, **OUTLINE**, **TIMELINE**, **TABLE**, **PLAN**, or **STRATEGY**, strictly apply AEO Principles: Lead with an "Inverted Pyramid" structure (40-60 word extraction-ready leads), use modular H2/H3 headers, listicles, and structural tables.
       - **Dialogue/Research Mode**: If the user is asking a direct question or exploring a trend, provide a **Simple, Natural, and Empathetic** response.
    4. **Deliverables Over Dialog**: If the user asks for a 'Timeline', produce a clear table or list with dates and events. If they ask for 'Trends', list them clearly.
    5. **Sonnet & Prose Balance**: Wrap technical depth (Prose) in a human-centric, philosophical intro and a reflective conclusion (Sonnet).
    
    CRITICAL: Do NOT include structural labels like 'Lead', 'Summary', 'H2', 'H3', 'Prose', or 'Sonnetal close' in your response. Use natural headers (##) and bolding to establish structure implicitly.
    """

    # Signal Detection (MCP Refactored)
    hybrid_signal_raw = mcp_detect_intent(topic)
    try:
        signal_data = json.loads(hybrid_signal_raw)
        research_intent = signal_data.get("intent", "NONE")
        formatting_directive = signal_data.get("directive", "")
    except Exception:
        research_intent = "NONE"
        formatting_directive = ""

    if is_grounded:
        temp = 0.0
        # Incorporate Intent into formatting instruction
        intent_instruction = ""
        if research_intent == "TIMELINE":
            intent_instruction = "CRITICAL: The user wants a TIMELINE. Provide a structured table with 'Date', 'Event', and 'Description'."
        elif research_intent == "TABLE":
            intent_instruction = "CRITICAL: Response MUST include a structured Markdown table. Do not include extra horizontal lines (---) beyond the standard Markdown header separator to ensure clean data extraction."
        elif research_intent == "CHART":
            intent_instruction = f"CRITICAL: The user wants a CHART. You MUST output data using Mermaid.js syntax (e.g., pie, graph TD, sequenceDiagram, mindmap). Do not add conversational filler around the code block. {formatting_directive}"
        elif research_intent == "LISTICLE":
            intent_instruction = f"CRITICAL: The user wants a LISTICLE. Use high-impact headers, numbered lists, and bolding for key terms. {formatting_directive}"
        elif formatting_directive:
            intent_instruction = f"FORMATTING DIRECTIVE: {formatting_directive}"
        
        if research_intent in ["CHART", "CSV"]:
            # Visual-Only Mode: Suppress long persona filler but allow a brief summary
            # We allow 2-3 sentences of analytical context to clarify the data.
            instruction = f"CRITICAL: You are in VISUAL MODE. {intent_instruction} DO NOT provide Python code, Google Sheets instructions, or standard conversational filler. Provide a brief analytical summary (2-3 sentences max), followed ONLY by the Mermaid.js or CSV content."
        elif intent_hint == "SIMPLE_QUESTION":
             # NO PERSONA WRAP for simple questions.
             instruction = "Provide a Simple, Natural, and Empathetic response based on the context. Do not use structural headers, intros, or conclusions. Just a direct answer."
        else:
            instruction = f"CRITICAL: Base answer PRIMARILY on 'GROUNDING_CONTENT'. {intent_instruction} {persona_instruction}"
    else:
        temp = 0.7
        if research_intent == "CHART":
             # Even without grounding, the user wants a visualization
             instruction = f"CRITICAL: You are in VISUAL MODE. {intent_instruction} Use your internal knowledge. Provide a brief analytical summary (2-3 sentences max), then ONLY the Mermaid.js content."
        elif intent_hint == "SIMPLE_QUESTION":
             # NO PERSONA WRAP for simple questions.
             instruction = "Provide a Simple, Natural, and Empathetic response based on the context. Do not use structural headers, intros, or conclusions. Just a direct answer."
        else:
             instruction = f"You are a strategic partner and content architect. {persona_instruction}"
        
    prompt = f"""
        {instruction}
        
        CONVERSATION HISTORY:
        {history}
        
        CURRENT REQUEST: "{topic}"
        
        RESEARCH CONTEXT:
        {context}
        
        RESPONSE:
        """
    return {
        "text": model.generate_content(prompt, generation_config={"temperature": temp}).text.strip(),
        "intent": research_intent if intent_hint != "SIMPLE_QUESTION" else "SIMPLE_QUESTION",
        "directive": formatting_directive
    }

# 14. The Dedicated pSEO Article Generator (High Trust & Transparency)
def generate_pseo_article(topic, context, history=""):
    global model
    print(f"Tool: Generating pSEO Article for '{topic}'")
    
    # 1. Determine System Instruction (Grounding Logic)
    # This ensures the model sticks to the facts found by the Researcher (worker-story)
    is_grounded = any("GROUNDING_CONTENT" in str(c) for c in context)
    if is_grounded:
        system_instruction = "CRITICAL INSTRUCTION: You are in READING MODE. Base the article PRIMARILY on the provided 'Research Context'. Do not hallucinate data."
    else:
        system_instruction = "You are an expert Editor. Use the provided context to write a high-authority article."

    # 2. Strict System Prompt for "Sonnet & Prose" Style
    prompt = f"""
    You are the Senior Editor for 'Sonnet & Prose', a publication exploring the intersection of creativity (Sonnet) and automation (Prose).
    
    {system_instruction}
    
    AUDIENCE: 
    Senior Marketers, CTOs, and Founders. They value depth, nuance, and honesty over hype.
    
    TONE:
    - **The Sonnet:** The introduction and conclusion must be human-centric, philosophical, and empathetic.
    - **The Prose:** The body must be technical, architectural, and actionable.
    
    STRUCTURE REQUIREMENTS (HTML Format):
    1.  **<section class="intro">**: A compelling hook connecting the topic to the human experience.
    2.  **<section class="body">**: Detailed analysis using <h2> and <h3> tags. Use specific data/facts from the context.
    3.  **<section class="methodology">**: A TRANSPARENCY FOOTER. 
        - Must be titled "## Methodology & Sources".
        - Explicitly list the domains/articles found in the research.
        - Explain *why* you selected them (e.g., "Used Google Trends data to verify 2025 growth").
    
    CRITICAL RULES:
    - Do NOT hallucinate. If the context is empty, admit it.
    - Return ONLY the HTML content (no ```html``` markdown wrappers).
    - Do NOT include labels like 'Intro', 'Body', or 'Methodology' as text; use the provided HTML class structure for organization.
    - Ensure it is ready for Ghost CMS.

    CONTEXTUAL DATA:
    Conversation History: 
    {history}
    
    Current Topic: "{topic}"
    
    Research Context (Grounding Data):
    {context}
    
    Article Draft:
    """
    
    return model.generate_content(prompt, generation_config={"temperature": 0.3}).text.strip()

# 14.5 The Recursive Deep-Dive Generator (Dynamic Room-by-Room Construction)
def generate_deep_dive_article(topic, context, history="", target_length=1500):
    global model
    print(f"Tool: Initiating Recursive Deep-Dive for '{topic}' (Target: {target_length} words)")
    
    is_grounded = any("GROUNDING_CONTENT" in str(c) for c in context)
    
    # Calculate architecture: Aim for ~300 words per section
    num_sections = max(3, target_length // 300)
    words_per_section = target_length // (num_sections + 1) # Plus one for intro
    
    # PHASE 1: Generate the Blueprint
    blueprint_prompt = f"""
    Create a strategic {target_length} word article blueprint for: "{topic}".
    STRUCTURE: Provide exactly {num_sections} thematic sections (H2) for the body.
    Return ONLY a JSON list of objects:
    [ {{"title": "Section Title", "focus": "What specific data or analogies to use here"}} ]
    
    RESEARCH CONTEXT: {context}
    HISTORY: {history}
    """
    
    try:
        blueprint_raw = model.generate_content(blueprint_prompt).text.strip()
        if "```json" in blueprint_raw: blueprint_raw = blueprint_raw.split("```json")[1].split("```")[0].strip()
        sections = json.loads(blueprint_raw)
    except Exception:
        return generate_pseo_article(topic, context, history)

    # PHASE 2: Recursive Room Building
    article_parts = []
    
    # Intro
    print(f"  + Drafting Architectural Intro...")
    intro_prompt = f"Write a compelling {words_per_section}-word philosophical intro for: '{topic}'. History: {history}"
    article_parts.append(f'<section class="intro">\n{model.generate_content(intro_prompt).text.strip()}\n</section>')
    
    for i, section in enumerate(sections):
        print(f"  + Constructing Room {i+1}/{len(sections)}: {section['title']}...")
        
        # CONTEXT STITCHING: Strip HTML and pass the tail of the previous part
        last_plain = strip_html_tags(article_parts[-1])
        prev_context = ". ".join(last_plain.split(". ")[-3:]) if ". " in last_plain else last_plain[-150:]
        
        room_prompt = f"""
        Write a {words_per_section}-word deep-dive section for: "{topic}".
        
        CHAPTER: {section['title']}
        FOCUS: {section['focus']}
        TONE: Authoritative, architectural, visionary.
        TRANSITION FROM: "...{prev_context}"
        
        GROUNDING DATA: {context if is_grounded else "Internal Knowledge base"}
        CONVERSATIONAL HISTORY: {history}
        
        OUTPUT: Provide the content wrapped in <h2> if there's a title, with clean 1-2 paragraph structure.
        """
        room_content = model.generate_content(room_prompt).text.strip()
        article_parts.append(f'<section class="body-part">\n{room_content}\n</section>')

    # PHASE 3: Methodology
    print("  + Finalizing Methodology & Transparency...")
    
    # Clean source extraction for the footer
    if is_grounded:
        found_urls = list(set(re.findall(r'https?://[^\s<>"]+', str(context))))
        source_text = ", ".join(found_urls[:5]) if found_urls else "Verified research sources."
    else:
        source_text = "AEO Synthesis from Knowledge Base."

    methodology_text = f"""
    <section class="methodology">
    <h2>Methodology & Transparency</h2>
    This {target_length}-word deep-dive was architected recursively to ensure narrative depth.
    Sources included: {source_text}
    </section>
    """
    article_parts.append(methodology_text)

    full_html = "\n\n".join(article_parts)
    print(f"  -> Deep-Dive Complete. Est words: {len(full_html.split())}")
    return full_html

#15. The Euphemistic 'Then vs Now' Linker
def create_euphemistic_links(keyword_context):
    global model
    prompt = f"""
    Topic: "{keyword_context['clean_topic']}". Context: {keyword_context['context']}
    Identify 4-10 core keyword clusters for 'Then' and 6-10 for 'Now'.
    CRITICAL SCHEMA: Exact keys: "then_concept", "now_concept", "link".
    Structure: {{ "interlinked_concepts": [ {{ "then_concept": "...", "now_concept": "...", "link": "..." }} ] }}
    """
    response = model.generate_content(prompt)
    return extract_json(response.text)

#16. The Proposal Critic and Refiner
def critique_proposal(topic, current_proposal):
    global model
    prompt = f"Review proposal for '{topic}': {json.dumps(current_proposal, indent=2)}. If excellent, respond: APPROVED. Else, provide concise feedback to improve 'Then vs Now' contrast."
    return model.generate_content(prompt).text.strip()

#17. The Proposal Refiner
def refine_proposal(topic, current_proposal, critique):
    global model
    prompt = f"""
    REWRITE proposal. Topic: {topic}. Draft: {json.dumps(current_proposal)}. Critique: {critique}. 
    Preserve keys: 'then_concept', 'now_concept', 'link'. Ensure JSON format.
    """
    return extract_json(model.generate_content(prompt).text)

# --- THE STATEFUL MAIN WORKER FUNCTION (FINAL V6 - SOCIAL AWARE) ---
@functions_framework.http
def process_story_logic(request):
    global model, flash_model, db
    if model is None:
        model = UnifiedModel(MODEL_PROVIDER, MODEL_NAME)
    
    if flash_model is None:
        # Initialize Vertex AI explicitly for the Flash Model (Triage)
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        flash_model = GenerativeModel(FLASH_MODEL_NAME, safety_settings=safety_settings)
        
    if db is None:
        db = firestore.Client(project=PROJECT_ID)

    req = request.get_json(silent=True)
    session_id, original_topic, slack_context = req['session_id'], req['topic'], req['slack_context']
    
    # --- STEP 1: PERCEPTION (Sanitize Input) ---
    sanitized_topic = sanitize_input_text(original_topic)
    
    # --- STEP 2: LOAD SHORT-TERM MEMORY ---
    doc_ref = db.collection('agent_sessions').document(session_id)
    session_doc = doc_ref.get()
    
    history_events = []
    history_text = ""
    if session_doc.exists:
        # MEMORY EXPANSION: Removed .limit(20) to ensure deep context recall
        events_ref = doc_ref.collection('events')
        
        # Sort DESCENDING (Newest first)
        query = events_ref.order_by('timestamp', direction=firestore.Query.DESCENDING)
        
        # Chronological order (Old -> New)
        history_events = [doc.to_dict() for doc in query.stream()][::-1]


    # MEMORY EXPANSION: Removed [-10:] slice to see the WHOLE conversation
    formatted_history = []
    for i, e in enumerate(history_events):
        if e.get('event_type') == 'tool_call':
            # MEMORY EXPANSION: Removed char limits to keep full data context
            entry = f"Turn {i+1} (System Search Results): {e.get('content', '')}" 
        else:
            # MEMORY EXPANSION: Removed char limits for full conversational fidelity
            entry = f"Turn {i+1} ({e.get('event_type')}): {e.get('text', '')}"
        formatted_history.append(entry)

    history_text = "\n".join(formatted_history)
    
    print(f"Worker loaded context with {len(history_events)} past events.")

    try:
        # --- STEP 3: TRIAGE (Now includes SOCIAL category) ---
        # Note: We triage BEFORE distilling or researching to save cost/latency.
        
        triage_prompt = f"""
            Analyze the user's latest message in the context of the conversation history.
            Classify the PRIMARY GOAL into one of five categories:

            1.  **SOCIAL_CONVERSATION**: Greetings, small talk, or simple feedback (e.g. "Okay", "Thanks").

            2.  **DEEP_DIVE**: **LONG-FORM ARCHITECTURE.** Select this if the user asks for a specific word count OR uses major drafting keywords.
                *   *Triggers:* "800 words", "1500 words", "Deep dive", "Draft an article", "Write a post", "Detailed draft", "Comprehensive article".
                *   *Priority:* If a word count is mentioned, ALWAYS pick this over TOPIC_CLUSTER.

            3.  **TOPIC_CLUSTER_PROPOSAL**: **SEMANTIC ARCHITECTURE.** The user specifically wants to generate a hierarchical map of keywords to establish topical authority. 
                *   Use this for: **Keyword fanning out** from a seed topic, mapping primary/secondary clusters.
                *   *CRITICAL:* If the user asks to "Write", "Draft", or "Expand" into a full piece, this is NOT a cluster.

            4.  **THEN_VS_NOW_PROPOSAL**: Specifically asking for a human-centric 'Then vs Now' structured comparison.

            5.  **PSEO_ARTICLE**: **STRICT CMS MODE (GHOST).** Only select this if the message EXPLICITLY mentions "**pSEO**", "**Ghost**", or "**Ghost CMS**". 

            6.  **SIMPLE_QUESTION**: For straightforward fact-checks, definitions, or simple inquiries (e.g., "What is...", "Is it true that...", "How many..."). 
                *   Use this when the user wants a direct, natural answer without structural headers or deep-dive analysis.

            7.  **DIRECT_ANSWER**: The "Collaborative Workspace" mode. Select this for EVERYTHING ELSE.
                *   Use this for: **Outlines**, **Strategies**, **Drafts**, **Lesson Plans**, **Research Queries**, and **Synthesis**.
                *   This is the high-quality synthesis mode for Slack interaction.

            CONVERSATION HISTORY:
            {history_text}

            USER REQUEST: "{original_topic}"

            CRITICAL: Respect the 'PSEO_ARTICLE' keyword rule. Handle TOPIC_CLUSTER as a technical semantic task. Respond with ONLY the category name.
            """
        intent = flash_model.generate_content(triage_prompt).text.strip()
        print(f"Smart Triage V5.4 classified intent as: {intent}")

        # Initialize variables for the response
        # FIX: Remove str()! Keep timestamp as a datetime object so it sorts correctly.
        new_events = [{"event_type": "user_request", "text": original_topic, "timestamp": datetime.datetime.now(datetime.timezone.utc)}] 
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
            reply_text = model.generate_content(social_prompt).text.strip()
            
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
            requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={"session_id": session_id, "type": "social", "message": reply_text, "channel_id": slack_context['channel'], "thread_ts": slack_context['ts']}, verify=True)
            return jsonify({"msg": "Social reply sent"}), 200


        # === PATH B: WORK/RESEARCH (The Heavy Lifting) ===
        # Only now do we pay the cost to distill and research.

        # 1. Research (Handle URLs or Keywords)
        urls = extract_urls_from_text(sanitized_topic)
        if urls:
            print(f"Detected {len(urls)} URLs. Processing sequentially (Browserless Tier Safety)...")
            combined_context = []
            for url in urls:
                article_text = fetch_article_content(url)
                combined_context.append(f"GROUNDING_CONTENT (Source: {url}):\n{article_text}")
            research_data = {"context": combined_context, "tool_logs": [], "research_intent": "URL_PROCESSING"}
        else:
            research_data = find_trending_keywords(sanitized_topic, history_context=history_text, session_id=session_id)
        
        # --- SAFETY KILL SWITCH (Hardening) ---
        intent_metadata = research_data.get("research_intent", "")
        is_blocked = False
        reply_text = "I'm sorry, I cannot fulfill this request due to safety guardrails."
        
        if isinstance(intent_metadata, str):
            meta_lower = intent_metadata.lower()
            # 1. Check for the new explicit VIOLATION intent
            if '"intent": "violation"' in meta_lower:
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
                "last_updated": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=30)
            })
            requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={
                "session_id": session_id, 
                "type": "social", 
                "message": reply_text, 
                "channel_id": slack_context['channel'], 
                "thread_ts": slack_context['ts']
            }, verify=True)
            return jsonify({"msg": "Safety block applied"}), 200

        if "tool_logs" in research_data: new_events.extend(research_data["tool_logs"])

        # CHANGE: Ensure downstream functions use the raw topic too, so they see the full request
        clean_topic = sanitized_topic

        # 3. Generate Output based on Intent
        if intent in ["DIRECT_ANSWER", "SIMPLE_QUESTION"]:
            answer_data = generate_comprehensive_answer(original_topic, research_data['context'], history=history_text, intent_hint=intent)
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

            session_ref.update({
                "status": "awaiting_feedback", 
                "type": "work_answer", 
                "topic": clean_topic,
                "slack_context": slack_context,
                "last_updated": expire_time
            })
            requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={
                "session_id": session_id, 
                "type": "social", 
                "message": answer_text, 
                "intent": research_intent,
                "directive": formatting_directive,
                "channel_id": slack_context['channel'], 
                "thread_ts": slack_context['ts']
            }, verify=True)
            return jsonify({"msg": "Answer sent"}), 200

        elif intent == "DEEP_DIVE":
            target_count = extract_target_word_count(original_topic)
            print(f"Executing Dynamic Deep-Dive Recursive Expansion (Target: {target_count})...")
            article_html = generate_deep_dive_article(original_topic, research_data['context'], history=history_text, target_length=target_count)
            
            # Use Claude for metadata even for deep dives
            seo_data = generate_seo_metadata(article_html, original_topic)
            
            new_events.append({"event_type": "agent_answer", "proposal_type": "deep_dive_article", "data": article_html})
            
            session_ref = db.collection('agent_sessions').document(session_id)
            events_ref = session_ref.collection('events')
            for event in new_events:
                if 'timestamp' not in event: event['timestamp'] = datetime.datetime.now(datetime.timezone.utc)
                events_ref.add(event)

            session_ref.update({
                "status": "awaiting_feedback", 
                "type": "work_proposal", 
                "topic": seo_data.get('title', clean_topic),
                "slack_context": slack_context,
                "last_updated": expire_time
            })
            requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={
                "session_id": session_id, 
                "type": "social", 
                "message": convert_html_to_markdown(article_html), 
                "channel_id": slack_context['channel'], 
                "thread_ts": slack_context['ts']
            }, verify=True)
            return jsonify({"msg": "Deep-Dive article sent"}), 200

        elif intent == "TOPIC_CLUSTER_PROPOSAL":
            cluster_data = generate_topic_cluster(clean_topic, research_data['context'], history=history_text)
            formatted_cluster = f"Here is the topic cluster you requested:\n```\n{json.dumps(cluster_data, indent=2)}\n```"
            new_events.append({"event_type": "agent_answer", "proposal_type": "topic_cluster", "data": cluster_data})
            
            # Writing to a Sub-collection
            session_ref = db.collection('agent_sessions').document(session_id)
            events_ref = session_ref.collection('events')

            for event in new_events:
                if 'timestamp' not in event: event['timestamp'] = datetime.datetime.now(datetime.timezone.utc)
                events_ref.add(event)

            session_ref.update({
                "status": "awaiting_feedback", 
                "type": "work_proposal", 
                "topic": clean_topic,
                "slack_context": slack_context,
                "last_updated": expire_time
            })
            requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={"session_id": session_id, "type": "social", "message": formatted_cluster, "channel_id": slack_context['channel'], "thread_ts": slack_context['ts']}, verify=True)
            return jsonify({"msg": "Topic cluster sent"}), 200

        elif intent == "THEN_VS_NOW_PROPOSAL":
            try:
                current_proposal = create_euphemistic_links(research_data)
                approval_id = f"approval_{uuid.uuid4().hex[:8]}"
                new_events.append({"event_type": "loop_draft", "iteration": 0, "proposal_data": current_proposal})

                loop_count = 0
                while loop_count < MAX_LOOP_ITERATIONS:
                    critique = critique_proposal(clean_topic, current_proposal)
                    if "APPROVED" in critique.upper(): break
                    try: current_proposal = refine_proposal(clean_topic, current_proposal, critique)
                    except Exception: break
                    loop_count += 1

                approval_id = f"approval_{uuid.uuid4().hex[:8]}"
                new_events.append({"event_type": "adk_request_confirmation", "approval_id": approval_id, "payload": current_proposal['interlinked_concepts']})
                
                # Writing to a Sub-collection
                session_ref = db.collection('agent_sessions').document(session_id)
                events_ref = session_ref.collection('events')

                for event in new_events:
                    if 'timestamp' not in event: event['timestamp'] = datetime.datetime.now(datetime.timezone.utc)
                    events_ref.add(event)

                session_ref.update({
                    "status": "awaiting_approval", 
                    "type": "work_proposal", 
                    "topic": clean_topic,          # <--- KEPT
                    "slack_context": slack_context, # <--- KEPT
                    "last_updated": expire_time
                })
                
                requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={"session_id": session_id, "approval_id": approval_id, "proposal": current_proposal['interlinked_concepts'], "thread_ts": slack_context['ts'], "channel_id": slack_context['channel'], "is_initial_post": True }, verify=True)
                return jsonify({"msg": "Proposal sent"}), 200

            except ValueError as e:
                # Fallback
                answer_text = generate_comprehensive_answer(original_topic, research_data['context'], history=history_text)
                new_events.append({"event_type": "agent_answer", "text": answer_text})
                
                session_ref = db.collection('agent_sessions').document(session_id)
                events_ref = session_ref.collection('events')

                for event in new_events:
                    if 'timestamp' not in event: event['timestamp'] = datetime.datetime.now(datetime.timezone.utc)
                    events_ref.add(event)

                session_ref.update({
                    "status": "awaiting_feedback", 
                    "type": "work_answer", 
                    "topic": clean_topic,
                    "slack_context": slack_context,
                    "last_updated": expire_time
                })

                requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={"session_id": session_id, "type": "social", "message": answer_text, "channel_id": slack_context['channel'], "thread_ts": slack_context['ts']}, verify=True)
                return jsonify({"msg": "Fallback answer sent"}), 200

        # === PATH C: PSEO ARTICLE GENERATION (Dual-Agent Path) ===
        elif intent == "PSEO_ARTICLE":
            # 1. AGENT A (Gemini): Generate the Body Content
            # We rely on Gemini's large context window for the deep research and writing.
            article_html = generate_pseo_article(original_topic, research_data['context'], history=history_text)
            
            # 2. AGENT B (Claude): Generate the Semantic Metadata
            # We feed the *drafted content* into Claude for high-precision SEO.
            seo_data = generate_seo_metadata(article_html, original_topic)
            
            # 3. Add to Memory
            new_events.append({"event_type": "agent_answer", "proposal_type": "pseo_article", "data": article_html})
            
            # 4. Write to Database
            session_ref = db.collection('agent_sessions').document(session_id)
            events_ref = session_ref.collection('events')
            for event in new_events:
                if 'timestamp' not in event: event['timestamp'] = datetime.datetime.now(datetime.timezone.utc)
                events_ref.add(event)

            session_ref.update({
                "status": "awaiting_feedback", 
                "type": "work_proposal", 
                "topic": seo_data.get('title', clean_topic), # Update topic with the better title
                "slack_context": slack_context,
                "last_updated": expire_time
            })
            
            # 5. Send STRUCTURED DATA to N8N
            # We merge the HTML from Agent A with the Metadata from Agent B
            requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={
                "session_id": session_id, 
                "type": "pseo_draft", 
                "payload": {
                    "title": seo_data.get("title"),
                    "html": article_html, # From Gemini
                    "tags": seo_data.get("tags", []),
                    
                    # NEW: High-Value Metadata from Anthropic
                    "custom_excerpt": seo_data.get("custom_excerpt"),
                    "meta_title": seo_data.get("meta_title"),
                    "meta_description": seo_data.get("meta_description"),
                    "featured_image_prompt": seo_data.get("featured_image_prompt"),
                    
                    "status": "draft"
                },
                "channel_id": slack_context['channel'], 
                "thread_ts": slack_context['ts']
            }, verify=certifi.where())

            return jsonify({"msg": "pSEO Draft (Enhanced) sent to N8N"}), 200

        else: 
            return jsonify({"error": f"Unknown intent: {intent}"}), 500

    except Exception as e:
        print(f"Worker Error: {e}")
        return jsonify({"error": str(e)}), 500

# --- FUNCTION: Ingest Knowledge ---
@functions_framework.http
def ingest_knowledge(request):
    global db, model
    if db is None: db = firestore.Client(project=PROJECT_ID)
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")

    req = request.get_json(silent=True)
    session_id = req.get('session_id')
    final_story = req.get('story')
    topic = req.get('topic') 
    
    if not final_story: return jsonify({"error": "Missing story"}), 400

    # Call the helper function (Global Scope)
    chunks = chunk_text(final_story)
    embeddings = embedding_model.get_embeddings([c for c in chunks])
    
    # Firestore Batch Write with Safety Check Limits
    batch = db.batch()
    count = 0
    
    # Iterate and add to batch
    for i, (text_segment, embedding_obj) in enumerate(zip(chunks, embeddings)):
        doc_ref = db.collection('knowledge_base').document(f"{session_id}_{i}")
        
        # 1. Set the data (Add to batch) - WITH PII SCRUBBING
        batch.set(doc_ref, {
            "content": scrub_pii(text_segment),
            "embedding": Vector(embedding_obj.values),
            "topic_trigger": scrub_pii(topic), 
            "source_session": session_id,
            "chunk_index": i,
            "created_at": datetime.datetime.now(datetime.timezone.utc)
        })
        
        count += 1
        
        # 2. The Safety Check
        if count >= 499:
            batch.commit()
            batch = db.batch() # Start fresh
            count = 0

    # 3. Commit leftovers (e.g., the last 12 items of 512)
    if count > 0: batch.commit()

    print(f"Ingested {len(chunks)} chunks.")
    return jsonify({"msg": "Knowledge ingested."}), 200