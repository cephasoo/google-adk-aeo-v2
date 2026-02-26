# --- /worker-story/main.py ---
import functions_framework
import html
from flask import jsonify
import vertexai
import litellm
from litellm import completion

import sys
import os
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
    get_mcp_client,
    get_system_instructions,
    extract_labeled_sources,
    get_stylistic_mentors,
    classify_topic_sector
)
from vertexai.generative_models import HarmCategory, HarmBlockThreshold
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
from google.cloud import firestore
from google.cloud.firestore_v1.vector import Vector
import concurrent.futures

# --- Logging Setup ---
logging_v_client = None
def get_logging_client():
    global logging_v_client
    if logging_v_client is None:
        try:
            logging_v_client = google.cloud.logging.Client()
            logging_v_client.setup_logging()
        except Exception:
            pass # Fallback to standard logging
    return logging_v_client

# --- SECRET MANAGER ---
secret_client = None

# --- Configuration ---
PROJECT_ID = os.environ.get("PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT")
if not PROJECT_ID:
    try:
        _, project_id_auth = google.auth.default()
        PROJECT_ID = project_id_auth
    except:
        PROJECT_ID = None

LOCATION = os.environ.get("LOCATION")
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-2.0-flash-001") 
MODEL_PROVIDER = os.environ.get("MODEL_PROVIDER", "vertex_ai")
FLASH_MODEL_NAME = os.environ.get("FLASH_MODEL_NAME", "gemini-2.0-flash-lite")
RESEARCH_MODEL_NAME = os.environ.get("RESEARCH_MODEL_NAME", "gemini-2.0-flash") # NEW: For High Context
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
N8N_PROPOSAL_WEBHOOK_URL = os.environ.get("N8N_PROPOSAL_WEBHOOK_URL") 
INGESTION_API_KEY = None
def get_ingestion_api_key():
    global INGESTION_API_KEY
    if not INGESTION_API_KEY:
        INGESTION_API_KEY = os.environ.get("INGESTION_API_KEY") or get_secret("ingestion-api-key")
    return INGESTION_API_KEY

KNOWLEDGE_INGESTION_API_KEY = None
def get_knowledge_ingestion_key():
    global KNOWLEDGE_INGESTION_API_KEY
    if not KNOWLEDGE_INGESTION_API_KEY:
        KNOWLEDGE_INGESTION_API_KEY = os.environ.get("KNOWLEDGE_INGESTION_API_KEY") or get_secret("knowledge-ingestion-api-key")
    return KNOWLEDGE_INGESTION_API_KEY

SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")  # Explicit fallback for code analysis
MAX_LOOP_ITERATIONS = 2
DEFAULT_GEO = os.environ.get("DEFAULT_GEO", "Global")
TRACKER_SERVER_URL = os.environ.get("TRACKER_SERVER_URL", "http://localhost:8081")
# (MCP_SERVER_URL removed, now centralized in shared.utils)

# --- Safety Configuration (ADK/RAI Compliant) ---
safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

# --- HELPER: Safe n8n Webhook Delivery ---


# --- Global Clients ---
unimodel = None
flash_model = None
research_model = None # NEW: The High-Context Specialist
specialist_model = None
search_api_key = None
db = None


# --- MODULAR SECTOR CLASSIFICATION ---

# --- MODULAR SECTOR CLASSIFICATION (Consolidated in shared.utils) ---
# (classify_topic_sector removed, now imported from shared.utils)

# --- MODULAR STYLE & SANITIZATION PROTOCOL (Consolidated in shared.utils) ---
# (PROTOCOL_* constants and get_system_instructions removed, now imported from shared.utils)

# --- MODULAR STYLE & SANITIZATION PROTOCOL (Consolidated in shared.utils) ---
# (Remaining PROTOCOL_* constants and get_system_instructions removed, now imported from shared.utils)

# --- HELPER: Dynamic Linguistic Palette ---
# (get_stylistic_mentors removed, now imported from shared.utils)

# --- HELPER: Citation Engine (Inline Anchors) ---
# (extract_labeled_sources removed, now imported from shared.utils)


# (RemoteTools and get_mcp_client removed, now centralized in shared.utils)

def sanitize_llm_html(raw_html):
    """
    Strips LLM preambles and postscripts, ensuring the output starts with 
    a valid Ghost-friendly HTML tag and ends at the last logical stop (tag or mermaid block).
    Standardized to protect <figure> tags and trailing code blocks.
    """
    if not raw_html: return ""
    
    # 1. Find the first structural tag OR markdown code fence
    # We include backticks in the start pattern to ensure we catch wrapped responses.
    start_pattern = r'(?:<(?:h1|h2|h3|section|article|p|figure|ul|ol|div|pre|code)|```|~~~)'
    start_match = re.search(start_pattern, raw_html, re.IGNORECASE)
    if not start_match:
        return raw_html.strip()
    
    start_idx = start_match.start()
    
    # 2. Find the last logical stop
    # This can be a closing tag OR a mermaid code fence
    end_pattern = r'(?:</(?:section|article|p|figure|ul|ol|div|h1|h2|h3|pre|code)>|```|~~~)'
    all_stops = list(re.finditer(end_pattern, raw_html, re.IGNORECASE))
    
    if not all_stops:
        # If no stops found, return from start_idx to end
        sanitized = raw_html[start_idx:].strip()
    else:
        last_stop = all_stops[-1]
        end_idx = last_stop.end()
        # Safety Check: If the last stop is very early, fallback to first-match pattern
        if end_idx <= start_idx:
            sanitized = raw_html[start_idx:].strip()
        else:
            sanitized = raw_html[start_idx:end_idx].strip()
            
    # 3. Post-clean: Strip wrapping code fences (e.g. ```html ... ```)
    # We use a non-greedy regex to strip if the content is entirely wrapped.
    fence_pattern = r'^(?:```(?:html|xml|text|ps1)?|~~~)\s*(.*?)\s*(?:```|~~~)$'
    fence_match = re.match(fence_pattern, sanitized, re.DOTALL | re.IGNORECASE)
    if fence_match:
        sanitized = fence_match.group(1).strip()

    # 4. Anti-Leakage: Strip accidental JSON blocks at the start (common in agent drift)
    # If the response starts with a code block contains "page_title" or "insight_type", it's leaked metadata
    json_leak_pattern = r'^\s*```json\s*\{[\s\S]*?\}\s*```\s*'
    if re.search(r'("page_title"|"insight_type"|"data_points")', sanitized[:500], re.IGNORECASE):
        sanitized = re.sub(json_leak_pattern, '', sanitized, count=1, flags=re.IGNORECASE)

    return sanitized.strip()

# --- UNIFIED MODEL ADAPTER (The Brain Switch) ---


# --- Safety Utils ---
# (scrub_pii removed - user requested removal of legacy PII logic)

# (detect_audience_context removed, now imported from shared.utils)

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
# (safe_generate_content removed, now imported from shared.utils)

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
    Synthesizes Tiered Discovery with Technical Discovery Protocol.
    Generates 1-3 prioritized queries in JSON format to mirror human research patterns.
    """
    global unimodel
    print(f"Distilling core topic from: '{user_prompt[:100]}...'")
    
    current_year = datetime.datetime.now().year
    extraction_prompt = f"""
    You are an expert Research Librarian specializing in authoritative discovery and high-precision information retrieval.
    Your goal is to convert a user request into a JSON list of 1-3 distinct, powerful Google Search queries.

    ### SEARCH OPERATOR RULES:
    1. **MAX 2 OPERATORS**: Never use more than 2 operators (like site: or "") in a single query.
    2. **TEMPORAL CONTEXT**: If the topic is time-sensitive (news, product releases, latest trends, current specs), include words like "latest", "{current_year}", "current", or "news".
    3. **QUOTES (VERY IMPORTANT)**: 
       - Use double quotes ONLY for the PRIMARY subject or specific error message.
       - **NEVER** use backslashes (`\`) for escaping.
       - **NEVER** wrap the entire query in an outer set of quotes if it already contains internal quotes.
    4. **NO COMMAS**: Use spaces only.
    5. **NO FLUFF**: Remove "I want to know", "Please tell me", "research for me".
    6. **CRITICAL: FILTER EDITORIAL INSTRUCTIONS**: 
       - REMOVE words like: "outline", "draft", "strategy", "blog post", "article", "word count", "Grade 8", "1500+ words", "logic flow".
       - PRESERVE discovery intent: "why", "how", "reasons", "causes", "factors", "impact", "trends".

    ### THE CORE PROBLEM:
    Do NOT generate "over-engineered" queries that include every technical or descriptive term. This creates a "technical vacuum" where no single page contains all terms, returning zero results.

    ### THE SOLUTION: TIERED DISCOVERY (Double-Tap)
    Generate a JSON list of 1-3 queries, moving from BROAD/AUTHORITATIVE to SPECIFIC/REFINED.

    **Tier 1: Broad Anchor (Authoritative Hubs)**
    Focus on the "Primary Entity" and the most authoritative domain for that subject.
    Patterns:
    - Official: site:domain.com "Subject Name"
    - Research: site:scholar.google.com OR site:arxiv.org
    - Public/News: site:nytimes.com OR site:reuters.com OR site:github.com
    Example: `site:developers.google.com "Google Merchant Center" documentation`

    **Tier 2: Discovery Pivot (Subject + Specific Intent)**
    Focus on the specialized intent or specific version/event.
    - News/Trends: "breaking", "update", "today", "report"
    - Products/Tech: "migration guide", "specs", "latest version", "manual"
    - Contextual: "comparison", "alternative", "reasons for"
    Example: `"Merchant API" product retrieval offerId`

    **Tier 3: Refined Detail (Nuance Only)**
    Only include implementation details or niche modifiers if primary intent fails.
    Example: `"Merchant API" register_gcp base64url`

    ### CONTEXT MAPPING:
    Use the HISTORY to resolve pronouns (he, it, that) to specific entities.

    HISTORY:
    {history[:5000]}

    USER REQUEST:
    "{user_prompt}"

    OUTPUT FORMAT (Raw JSON List Only):
    ["query 1", "query 2"]
    """
    
    raw_response = safe_generate_content(unimodel, extraction_prompt, generation_config={"temperature": 0.2})
    try:
        raw_queries = extract_json(raw_response)
        
        # Post-Processing Sanitize: Ensure syntax is clean for SerpAPI
        if isinstance(raw_queries, list):
            sanitized = []
            for q in raw_queries:
                if not isinstance(q, str): continue
                # 1. Replace escaped quotes with regular quotes
                q = q.replace('\\"', '"').replace("\\'", "'")
                # 2. Strip redundant outer quotes if internal quotes exist
                q = q.strip()
                if q.startswith('"') and q.endswith('"') and q.count('"') > 2:
                    q = q[1:-1].strip()
                sanitized.append(q)
            
            if sanitized:
                print(f"Distilled Tiers: {sanitized}")
                return sanitized
        
        # Fallback 1: Extract all quoted strings (handles broken lists)
        backup_queries = re.findall(r'"([^"]+)"', str(raw_response))
        if backup_queries:
            return backup_queries[:3]
            
        return [str(raw_response).strip().strip('"[]')]
    except Exception as e:
        print(f"FAILED TO DISTILL CORE TOPIC: {e}")
        return [str(raw_response).strip().strip('"[]')]

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


# --- HELPER: Market Data Extraction (pSEO) ---
def extract_market_data(context_text, history_text, topic_slug):
    """
    Extracts structured market data points from the research context AND conversation history.
    Ensures pSEO data cards reflect the full threaded context (User inputs + RAG findings).
    """
    extraction_prompt = f"""
    You are a Market Data Analyst. Extract specific economic metrics for "{topic_slug}".
    
    1. ANALYZE CONVERSATION HISTORY (User inputs & previous data):
    {history_text[:5000]}
    
    2. ANALYZE RESEARCH CONTEXT (Fresh Grounding):
    {context_text[:15000]} 
    
    Required Data Points:
    1. hub_capital: The commercial or consumption capital city.
    2. spend_trend: Specific consumption expenditure stats (e.g., "% of GDP", "Growth Rate").
    3. market_tier: Classify as "Emerging", "High Growth", "Maturing", or "Frontier".
    4. region: The specific sub-region (e.g., "West Africa", "East Africa").
    
    Output Format (JSON Only):
    {{
      "hub_capital": "City Name or 'Analyze'",
      "spend_trend": "Stat or 'Analyze'",
      "market_tier": "Classification",
      "region": "Sub-Region"
    }}
    
    Rules:
    - Prioritize recent Research Context, but fill gaps from History.
    - If data is missing, use "Analyze" or reasonable inference based on region.
    - Be concise.
    """
    
    try:
        # Use Specialist Model for precision
        raw_json = safe_generate_content(specialist_model or unimodel, extraction_prompt, generation_config={"temperature": 0.2})
        
        # Sanitize JSON
        if "```json" in raw_json:
            raw_json = raw_json.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_json:
            raw_json = raw_json.split("```")[1].split("```")[0].strip()
            
        data = json.loads(raw_json)
        return data
    except Exception as e:
        print(f"⚠️ Market Data Extraction Failed: {e}")
        return {
            "hub_capital": "Analyze",
            "spend_trend": "Data Pending",
            "market_tier": "Emerging",
            "region": "Sub-Saharan Africa"
        }


def extract_lp_data(context_text, history_text, topic_slug, topic):
    """
    Synthesises lp-data JSON for the codeinjection_head block of long-form Landing Pages.
    Parallel to extract_market_data() but designed for LP pages — produces the full
    lp-data object that N8N injects verbatim via JSON.stringify(data_points).
    """
    extraction_prompt = f"""
    You are an LP Metadata Extractor. Synthesise structured metadata for a content-heavy
    long-form landing page on: "{topic}"

    CONVERSATION CONTEXT:
    {history_text[:5000]}

    RESEARCH CONTEXT:
    {context_text[:10000]}

    Output ONLY this JSON (no markdown fences):
    {{
      "insight_type": "lp",
      "category_label": "<Short topic badge, e.g. 'AI Shopping Framework'>",
      "reading_time_label": "<Estimated read e.g. '12 min read'>",
      "last_updated": "<Month Year e.g. 'February 2025'>",
      "cluster_tag": "<Internal Ghost tag slug e.g. 'privacy-ai-cluster'>",
      "key_stats": [
        {{ "label": "<Label>", "value": "<Value>" }}
      ],
      "faq": [
        {{ "q": "<Question>", "a": "<Answer>" }}
      ]
    }}

    Rules:
    - insight_type: always "lp" (verbatim — used by the frontend router).
    - category_label: 2-4 words max. Topic badge shown in hero.
    - reading_time_label: estimate from 3,000+ words base. Use "X min read".
    - last_updated: current month + year (February 2025).
    - cluster_tag: lowercase, hyphenated, descriptive of this LP's content cluster.
    - key_stats: 1-3 items. MANDATORY: Search the HISTORY and RESEARCH for specific ROI metrics (e.g. Bloomreach 21.7% lift, Friendbuy 25% AOV). Do NOT hallucinate generic placeholders.
    - faq: 2-5 pairs. Use high-intent questions from the context. PRIORITIZE "People Also Ask" (PAA) queries found in the research context to align with real-world search intent.
    - Omit keys with empty arrays entirely.
    - NEGATIVE CONSTRAINT: Output ONLY JSON. Do not include any conversational preamble or metadata field names in the plain-text body.
    """

    try:
        raw_json = safe_generate_content(
            specialist_model or unimodel,
            extraction_prompt,
            generation_config={"temperature": 0.2}
        )

        # Sanitize JSON fences
        if "```json" in raw_json:
            raw_json = raw_json.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_json:
            raw_json = raw_json.split("```")[1].split("```")[0].strip()

        data = json.loads(raw_json)
        # Guarantee insight_type is always "lp" regardless of model output
        data["insight_type"] = "lp"
        return data

    except Exception as e:
        print(f"⚠️ LP Data Extraction Failed: {e}")
        return {
            "insight_type": "lp",
            "category_label": "Framework",
            "reading_time_label": "12 min read",
            "last_updated": "February 2025",
            "cluster_tag": f"{topic_slug}-cluster"
        }

def strip_html_tags(text):
    """Removes HTML tags for clean context stitching and plain-text views."""
    return re.sub(r'<[^>]+>', '', text).strip()

# (convert_html_to_markdown removed, now imported from shared.utils)

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

# (extract_json removed, now imported from shared.utils)

def extract_urls_from_text(text):
    """Extracts ALL URLs from a given string, stripping trailing punctuation."""
    # Matches http/https, allows typical URL chars, but stops before trailing punctuation like ), ], } or final dots/commas
    raw_urls = re.findall(r'(https?://[^\s<>|]+)', text)
    clean_urls = []
    for url in raw_urls:
        # Strip common trailing punctuation often caught by greedy regex
        clean_url = url.rstrip(').],;')
        clean_urls.append(clean_url)
    return clean_urls

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
    """
    Project-Wide Upgrade: Deep Research Protocol.
    Instead of just reading SERP Snippets, this function:
    1. Fetches the SERP (Headlines).
    2. Identifies Top URLs.
    3. Autonomously 'Clicks' (Scrapes) the most relevant page.
    4. Performs a 'Double-Tap' if the first page is thin or irrelevant.
    """
    print(f"TELEMETRY: Executing Deep Research Search for: '{query}'")
    
    # 1. Fetch Headlines (SERP)
    try:
        serp_results = get_mcp_client().call("google_web_search", {"query": query})
    except Exception as e:
        print(f"⚠️ SERP Fetch Failed: {e}")
        return "Search Failed."

    # 2. Extract Candidates
    urls = extract_urls_from_text(serp_results)
    unique_urls = []
    [unique_urls.append(x) for x in urls if x not in unique_urls]
    
    # 3. Deep Research Loop (The "Click")
    deep_context = [serp_results]
    
    if unique_urls:
        print(f"  -> Found {len(unique_urls)} candidates. Initiating Deep Content Retrieval...")
        
        # We try up to 2 URLs (The Double-Tap)
        attempts = 0
        max_attempts = 2 
        
        for url in unique_urls[:3]: # Look at top 3, scrape max 2
            if attempts >= max_attempts: break
            
            # Skip likely junk (navigation/policy pages)
            if any(x in url.lower() for x in ['privacy', 'terms', 'signup', 'login']): continue
            
            try:
                print(f"  -> Deep Scraping ({attempts+1}/{max_attempts}): {url}")
                content = fetch_article_content(url)
                
                if not content or len(content) < 500:
                    print(f"     ❌ Content too thin ({len(content) if content else 0} chars). Skipping.")
                    continue
                
                # 4. Semantic Validation (Relevance Check)
                # Does the content actually contain keywords from the user's query?
                # Simple heuristic: Split query into words (>3 chars), check existence in content.
                query_words = [w.lower() for w in query.split() if len(w) > 3]
                relevance_score = sum(1 for w in query_words if w in content.lower())
                
                # If we have 0 matches for query keywords, it's likely a navigation wrapper or captcha.
                if relevance_score == 0:
                    print(f"     ⚠️ Low Relevance (Score 0). Content might be garbage. Double-Tapping...")
                    # We still append it? Maybe not. Let's append but try next.
                    deep_context.append(f"\n[DEEP_CONTENT (Low Confidence) - {url}]:\n{content[:5000]}...")
                    attempts += 1
                    continue
                
                # High Relevance!
                print(f"     ✅ Content Valid (Score {relevance_score}). Appending.")
                deep_context.append(f"\n[DEEP_CONTENT - {url}]:\n{content[:25000]}") # 25k chars limit
                
                # If it's REALLY good (lots of keywords), we can stop early?
                # User asked for "thinks through another". So maybe we just do 1 good one?
                # Let's say: If score > 2, we are happy with 1.
                if relevance_score > 2:
                    break
                
                attempts += 1
                
            except Exception as e:
                print(f"     ⚠️ Scrape Error: {e}")
                
    return "\n".join(deep_context)

#5. The Specialist Google Trends Tool (MCP Refactored)
def search_google_trends(geo="US"):
    return get_mcp_client().call("google_trends", {"geo": geo})
    
# --- 6. ANALYSIS TOOL: Trend History (MCP Refactored) ---
def analyze_trend_history(query, geo="US", date="12 Months"):
    return get_mcp_client().call("trend_analysis", {"query": query, "geo": geo, "date": date})

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

# --- BING & YOUTUBE TOOLS (NEW) ---
def search_bing_web(query):
    return get_mcp_client().call("bing_search", {"query": query})

def search_bing_copilot(query):
    return get_mcp_client().call("bing_copilot", {"query": query})

def search_youtube_videos(query):
    return get_mcp_client().call("youtube_search", {"query": query})

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
    
    ### CONTEXTUAL ANCHORS (The Source of Truth):
    USER COMMAND: '{raw_topic}'
    MISSION SUBJECT: '{grounding_subject}'
    CURRENT INTENT: '{prev_intent}'
    CURRENT GEO: '{prev_geo}'
    
    ### KNOWLEDGE REPOSITORY:
    CONVERSATION HISTORY:
    {history_context[:8000]}
    ---
    RESEARCH/MEMORY SAMPLES: 
    {str(base_grounding)[:15000]}
    
    ### PROTOCOL & GUARDRAILS:
    1. **CONTEXTUAL ANCHORING**: Use the User Command or Provided Files (found in COLLECTED KNOWLEDGE) as your primary relevance filter. 
    2. **ERA ALIGNMENT (SUGGESTION)**: If the task involves modern tech (e.g., AP2, VCs, Web 3.0), prioritize research in that era. Avoid older patterns (e.g., OAuth2, JWT, Web 2.0) unless specifically requested.
    3. **TEMPORAL VERIFICATION PROTOCOL**: If the user's request involves ANY of the following, flag for verification in your rationale:
       - **Technical**: API versions, library versions, framework updates, deprecation status
       - **News/Events**: Current events, breaking news, recent announcements, policy changes
       - **Trends**: Market shifts, search volume data, emerging patterns, consumer behavior
       - **Statistics**: Current metrics, recent studies, updated benchmarks, market data
       - **Regulatory**: New laws, compliance updates, regulatory changes, legal requirements
       Example rationale: "API version verification required for Google Merchant Center" or "Current trend data needed for AI agent adoption"
    4. **MULTI-VARIATE RESEARCH**: Decide if the user needs Architectural, Contextual, or Comparative specs. 
    5. **VISUAL_INTENT**: Require seeing asset? [YES/NO]
    6. **GEO_PIVOT**: If the user refers to a specific country, you MUST return the **ISO-3166-1 alpha-2** code.
    7. **ADEQUACY_AUDIT**: Have enough info to solve the ARCHITECTURAL/CONTEXTUAL goal? [SUFFICIENT/INSUFFICIENT]
    8. **TOOL_RECRUITMENT**: List tools:
       - **WEB**: Technical documentation, news, and general grounding (Google).
       - **BING**: Cross-platform technical search and alternative perspectives.
       - **COPILOT**: Complex reasoning, technical synthesis, or summary-first answers.
       - **YOUTUBE**: Visual tutorials, demonstrations, and recollective grounding via video metadata.
       - **IMAGES**: Visual research and reference.
       - **VIDEOS**: General video search (Google).
       - **TRENDS**: (USE SPARINGLY) For broad regional trending topics only (e.g., "Top searches in US").
       - **ANALYSIS**: For topic-specific historical interest and volume (e.g., "Tesla Model 2 interest").
       - **COMPLIANCE**: Legal and regulatory documentation.
       - **USE_CONVERSATIONAL_CONTEXT**: If existing history is sufficient.
    
    OUTPUT FORMAT (Raw JSON Only):
    {{"visual_intent": "YES/NO", "new_geo": "...", "adequacy": "...", "selected_tools": [], "timeframe": "Today/7 Days/12 Months", "rationale": "..."}}
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
        requested_timeframe = router_data.get("timeframe", "12 Months")
        research_intent_raw = json.dumps({"intent": prev_intent, "rationale": router_data.get("rationale")})
        print(f"  -> Intelligence Router: {router_data.get('rationale')} | Target Geo: {detected_geo} | Timeframe: {requested_timeframe}")
    except Exception as e:
        print(f"⚠️ Router failed: {e}. Falling back.")
        is_recollective_visual_query = False
        detected_geo = prev_geo
        target_geos = [prev_geo] if prev_geo else ["NG"]
        adequacy_score = "INSUFFICIENT"
        selected_tools = ["WEB"]
        requested_timeframe = "12 Months"
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
    # ADK HARDENING: We FORCE research for high-fidelity intents (DEEP_DIVE, BLOG_OUTLINE)
    # if we have no unique grounding URLs in the context, even if the model thinks it's SUFFICIENT.
    # This ensures "Staff Engineer" over-confidence doesn't break the citation protocol.
    has_grounding_urls = len(re.findall(r'https?://[^\s<>"]+', str(context_snippets))) > 0
    is_high_fidelity = any(t in str(prev_intent) for t in ["DEEP_DIVE", "BLOG_OUTLINE", "AUTHOR"])
    
    if adequacy_score == "SUFFICIENT" and is_high_fidelity and not has_grounding_urls:
        print("SENSORY GATEKEEPER: High-fidelity intent detected with zero grounding URLs. Forcing search.")
        adequacy_score = "INSUFFICIENT"
        selected_tools.append("WEB")

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

    # Fast-Exit for SIMPLE_QUESTION: Skip expensive distillation for pulse checks
    has_target_urls = len(extract_urls_from_text(raw_topic)) > 0
    is_pulse_check = (prev_intent == "SIMPLE_QUESTION" and not has_target_urls and not images)
    
    if any(t in ["WEB", "BING", "COPILOT", "YOUTUBE", "IMAGES", "VIDEOS", "SCHOLAR", "SIMPLE"] for t in selected_tools):
        if is_pulse_check and any(word in raw_topic.lower() for word in ["trending", "latest", "today", "now"]):
            print("Sensory Router: [FAST-EXIT] Pulse check detected. Skipping distillation...")
            distilled_seo_queries = [raw_topic]
        else:
            print("Sensory Router: Category [SEO] active. Distilling high-precision query...")
            distilled_seo_queries = extract_core_topic(raw_topic, history=history_context)
        
        # Ensure it's a list for iteration
        if not isinstance(distilled_seo_queries, list):
            distilled_seo_queries = [str(distilled_seo_queries)]
            
        print(f"  -> SEO Queries: {distilled_seo_queries}")
        
        # VERIFICATION: Log query construction details for debugging
        is_technical_query = any(term in raw_topic.lower() for term in [
            "api", "current", "documentation", "library", "sdk", "paper", "thesis", 
            "research", "specifications", "specs", "technical", "standard", "rfc", "iso"
        ])
        if is_technical_query:
            print(f"  -> QUERY TYPE: Technical Discovery (will use fallback if needed)")
            print(f"  -> ORIGINAL REQUEST: '{raw_topic[:150]}...'")

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
    knowledge_tools = [t.split(':')[0].strip() for t in selected_tools if t.split(':')[0].strip() in ["WEB", "BING", "COPILOT", "YOUTUBE", "IMAGES", "VIDEOS", "SCHOLAR", "SIMPLE"]]
    
    for choice in knowledge_tools:
        print(f"Executing Global Sensory Tool: '{choice}'")
        research_context = None
        tool_name = "unknown"

        if choice == "WEB":
            # DOUBLE-TAP PROTOCOL: Iterate through tiers until results are found
            for i, query in enumerate(distilled_seo_queries[:3]):
                print(f"  -> [Attempt {i+1}/3] Search: '{query}'")
                research_context = search_google_web(query)
                tool_name = "serpapi_web_search_global"
                
                if research_context and "No significant results" not in str(research_context):
                    print(f"  ✅ Results found on attempt {i+1}.")
                    break
                else:
                    print(f"  ❌ No results for tier {i+1}. Pivoting...")

        elif choice == "IMAGES":
            research_context = search_google_images(distilled_seo_queries[0])
            tool_name = "serpapi_image_search"
        elif choice == "VIDEOS":
            research_context = search_google_videos(distilled_seo_queries[0])
            tool_name = "serpapi_video_search"
        elif choice == "BING":
            research_context = search_bing_web(distilled_seo_queries[0])
            tool_name = "serpapi_bing_web_search"
        elif choice == "COPILOT":
            research_context = search_bing_copilot(distilled_seo_queries[0])
            tool_name = "serpapi_bing_copilot_search"
        elif choice == "YOUTUBE":
            research_context = search_youtube_videos(distilled_seo_queries[0])
            tool_name = "serpapi_youtube_search"
        elif choice == "SCHOLAR":
            research_context = search_google_scholar(distilled_seo_queries[0])
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
                    print(f"  + Fetching Quantitative Trend Stats for '{search_query}' in '{geo_code}' (Timeframe: {requested_timeframe})...")
                    stats_context = analyze_trend_history(search_query, geo=geo_code, date=requested_timeframe)
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


# #11. The Topic Cluster Generator
def generate_topic_cluster(topic, context, history="", is_initial=True, session_id=None):
    global unimodel
    style_mentors = get_stylistic_mentors(session_id)
    
    state_context = "This is a NEW proposal for a new thread." if is_initial else "This is a REVISION or EXTENSION of a previous strategy in an existing thread."
    
    # ARCHITECTURAL FIX: Modular Instruction Assembly
    # Intent for Topic Clusters is specialized but shares technical/grounding rules.
    sys_instruction = get_system_instructions("TOPIC_CLUSTER_PROPOSAL", "MODERATOR_VIEW")
    sys_instruction += f"\n\n{style_mentors}"

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
    return extract_json(safe_generate_content(unimodel, prompt, system_instruction=sys_instruction))

# 12. The SEO Metadata Generator (Using Specialist Model)
def generate_seo_metadata(article_html, topic, session_id=None, intent=None):
    global unimodel
    style_mentors = get_stylistic_mentors(session_id)
    """
    Uses the UnifiedModel adapter to route this specific task to Anthropic (Claude).
    """
    print(f"Tool: Delegating SEO Metadata to Specialist Model (Anthropic)...")
    
    # ARCHITECTURAL FIX: Modular Instruction Assembly
    # Intent for Metadata is naturally specialized. Use BLOG_OUTLINE for high-fidelity rules.
    sys_instruction = get_system_instructions("BLOG_OUTLINE", "CMS_DRAFT")
    sys_instruction += f"\n\n{style_mentors}"

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
    Intent: "{intent or 'PSEO_ARTICLE'}"
    Topic: "{topic}"
    Title Hint (Priority): "{title_hint}"
    
    Article Content (HTML):
    {article_html[:15000]} # Truncate to avoid context limits
    
    TASK:
    Generate highly optimized, click-worthy metadata for this article or data page.
    
    REQUIREMENTS:
    1. **title**: The primary H1 title. 
       - If a clear, substantive title exists in the provided HTML (<h1>) or the 'Topic', PRESERVE IT EXACTLY. 
       - SPECIAL RULE FOR PSEO_PAGE: If no high-quality title exists, synthesize one using the format "[Entity] [Insight Name] Insight". Analyze the content to determine a relatable "Insight Name" (e.g., "Nigeria Market Penetration Insight").
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
        # Use the Global Specialist Brain (Anthropic) with system_instruction
        content = safe_generate_content(specialist_model, prompt, generation_config={"temperature": 0.7}, system_instruction=sys_instruction)
        
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
    3. TABLE COMPACTION: Strips blank lines within pipe-based tables.
    """
    if not text: return ""
    lines = text.split('\n')
    new_lines = []
    in_code_block = False
    
    # Pass 1: Compact Table Slags (Strip blank lines within |...|) + Strip Separators
    compacted_lines = []
    separator_pattern = re.compile(r'^\s*\|[\s\-:|\|]+\|\s*$')  # Matches |---|---| or |:::|:::| etc.
    
    for line in lines:
        # Skip table separator lines (e.g., |---|---|)
        if separator_pattern.match(line):
            continue
            
        if not line.strip() and compacted_lines and compacted_lines[-1].strip().startswith('|') and compacted_lines[-1].strip().endswith('|'):
            # Look ahead for a matching pipe to confirm it's a split table
            # However, for robustness, we just drop interior blanks if sandwiched by pipes
            continue 
        compacted_lines.append(line)
    
    lines = compacted_lines
    
    for i, line in enumerate(lines):
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            new_lines.append(line)
            continue
            
        if in_code_block:
            new_lines.append(line)
            continue
            
        is_table_row = line.strip().startswith('|') and line.strip().endswith('|')
        
        is_list_item = line.strip().startswith(('-', '*', '1.', '•'))
        is_header = line.strip().startswith('#')
        is_empty = not line.strip()
        
        processed_line = re.sub(r'\*\*(.+?)\*\*', r'*\1*', line)
        
        if is_header:
            # Convert markdown headers (# Header) to Slack bold (*Header*), allowing leading whitespace
            processed_line = re.sub(r'^\s*#+\s+(.+)$', r'*\1*', processed_line)
            
        new_lines.append(processed_line)
        
        if not is_empty and not is_list_item and not is_header and not is_table_row and i < len(lines) - 1:
            next_line = lines[i+1]
            if next_line.strip() and not next_line.strip().startswith(('-', '*', '1.', '•')) and not next_line.strip().startswith('|'):
                new_lines.append("")  

    return "\n".join(new_lines)





# 13a. The Natural Answer Generator (Conversational & Fluid)
def generate_natural_answer(topic, context, history="", session_id=None, output_target="MODERATOR_VIEW", intent_label="SIMPLE_QUESTION"):
    global unimodel, research_model
    style_mentors = get_stylistic_mentors(session_id)
    topic_sector = classify_topic_sector(topic, flash_model=flash_model)
    
    # ARCHITECTURAL FIX: Modular Instruction Assembly
    system_instruction = get_system_instructions(intent_label, output_target, topic_sector=topic_sector)
    system_instruction += f"\n\n{style_mentors}"

    print(f"DEBUG: generate_natural_answer (Native) starting... [Topic: {topic[:50]}]")
    
    # --- ROUTING LOGIC: Context Size Check ---
    total_context_size = len(str(context)) + len(str(history))
    
    has_code = "[CODE_ANALYSIS]" in str(context) or "[CODE_ANALYSIS]" in str(history)
    use_research_model = total_context_size > 30000 or has_code
    
    active_model = research_model if (use_research_model and research_model) else unimodel
    model_name = "Research Model (Flash)" if active_model == research_model else "Unimodel (GPT-5)"
    print(f"DEBUG: Routing Natural Answer to [{model_name}] | Context Size: {total_context_size} chars")

    prompt = f"""
    ### CONVERSATION HISTORY:
    {history}
    
    ### GROUNDING DATA:
    {context}
    
    ### USER INSTRUCTION:
    {topic}
    
    TASK:
    Answer naturally and conversationally using the GROUNDING DATA to ensure accuracy.
    """
    
    # Use Safe Gen Wrapper with system_instruction
    text = safe_generate_content(active_model, prompt, generation_config={"temperature": 0.4}, system_instruction=system_instruction)
    
    final_text = ensure_slack_compatibility(text.strip())
    intent = intent_label
    if "```" in final_text: intent = "TECHNICAL_EXPLANATION"
    
    return {
        "text": final_text,
        "intent": intent,
        "directive": ""
    }


# 13b. The Comprehensive Answer Generator (AEO-AWARE CONTENT STRATEGIST)
def generate_comprehensive_answer(topic, context, history="", intent_metadata=None, context_topic="", session_id=None, output_target="MODERATOR_VIEW", intent_label=None):
    global unimodel, research_model, specialist_model, flash_model
    style_mentors = get_stylistic_mentors(session_id)
    
    # 2. INTENT DETECTION (Early for instruction assembly)
    try:
        signal_data = json.loads(intent_metadata) if intent_metadata else {}
        research_intent = intent_label or signal_data.get("intent", "FORMAT_GENERAL")
        formatting_directive = signal_data.get("directive", "")
    except:
        research_intent = intent_label or "FORMAT_GENERAL"
        formatting_directive = ""

    # ARCHITECTURAL FIX: Modular Instruction Assembly
    topic_sector = classify_topic_sector(topic, flash_model=flash_model)
    system_instruction = get_system_instructions(research_intent, output_target, topic_sector=topic_sector)
    system_instruction += f"\n\n{style_mentors}"

    print(f"DEBUG: generate_comprehensive_answer starting... [Topic: {topic[:50]}]")
    
    # --- ROUTING LOGIC: Context Size Check ---
    total_context_size = len(str(context)) + len(str(history))
    
    has_code = "[CODE_ANALYSIS]" in str(context) or "[CODE_ANALYSIS]" in str(history)
    use_research_model = total_context_size > 30000 or has_code
    
    active_model = specialist_model if specialist_model else (research_model if (use_research_model and research_model) else unimodel)
    model_name = "Specialist Model (Claude 4.5)" if active_model == specialist_model else ("Research Model (Flash)" if active_model == research_model else "Unimodel (GPT-5)")
    print(f"DEBUG: Routing Comprehensive Answer to [{model_name}] | Context Size: {total_context_size} chars")
    
    context_str = str(context)
    is_grounded = "GROUNDING_CONTENT" in context_str or "IN-CONTEXT HISTORY" in context_str
    
    # 1. PERSONA & AUDIENCE
    audience_context = detect_audience_context(history)
    
    if topic_sector == "LIFESTYLE":
        persona_goal = "Generate a warm, descriptive lifestyle discovery or recommendation. Use a first-person curator voice. Avoid repetitive source name-dropping (e.g. TripAdvisor); weave facts into your own narrative."
        strategist_type = "Expert Travel & Lifestyle Journalist"
    elif topic_sector == "HUMANITIES":
        persona_goal = "Generate an objective sociopolitical research audit or overview. Use an authoritative analyst voice. Weave provenance into the narrative rather than repeating 'According to Bing/Search'."
        strategist_type = "Sociopolitical Analyst"
    else:
        persona_goal = "Generate a high-density technical research audit or implementation guide."
        strategist_type = "Senior Content Strategist"

    persona_instruction = f"""
    You are a {strategist_type}. 
    AUDIENCE: {audience_context}
    
    GOAL:
    {persona_goal}
    Avoid "Sports Blindness" or generic global signals. Look for the nuance in the grounding data.
    """

    # 3. FORMATTING SENTINEL (Hard-coded prioritization)
    intent_instruction = ""
    
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
    
    ### LATEST INSTRUCTION (PRIORITY):
    "{topic}"
    
    ### GROUNDING DATA:
    {context}
    
    ### CONVERSATION HISTORY:
    {history}
    
    ### FORMATTING GUIDANCE:
    {intent_instruction}
    {extract_labeled_sources(context)}
    """

    # Use Safe Gen Wrapper with system_instruction
    text = safe_generate_content(active_model, prompt, generation_config={"temperature": 0.3}, system_instruction=system_instruction)
    
    # INTEGRATE MERMAID: Convert any Mermaid blocks to images before final formatting
    processed_text = post_process_mermaid_to_images(text, output_target)
    
    # Enforce spacing and Slack compatibility
    final_text = ensure_slack_compatibility(processed_text.strip())
    
    return {
        "text": final_text,
        "intent": research_intent,
        "directive": formatting_directive
    }

# 13b. Mermaid Post-Processor (Ghost Compatibility Fix)
def post_process_mermaid_to_images(html_content, output_target="CMS_DRAFT"):
    """
    Detects Mermaid code blocks in HTML and converts them into visual images.
    Routes format to markdown if output_target is MODERATOR_VIEW.
    """
    if not html_content or 'mermaid' not in html_content.lower():
        return html_content

    print(f"  + [MCP-PROCESS] Converting Mermaid blocks via Sensory Hub...")
    mcp = get_mcp_client()
    
    def replacer(match):
        full_block = match.group(0)
        # Try to find a caption in a surrounding figure
        caption = "Mermaid Diagram"
        caption_match = re.search(r'<figcaption>(.*?)</figcaption>', full_block, re.IGNORECASE)
        if caption_match:
            caption = caption_match.group(1).strip()
            # Clean for ALT tag safety
            caption = re.sub(r'<[^>]+>', '', caption)
            
        safe_alt = html.escape(caption, quote=True)

        # Find the actual mermaid code - broadened class matching
        code_match = re.search(r'<code[^>]*class="[^"]*mermaid[^"]*"[^>]*>([\s\S]*?)</code>', full_block, re.IGNORECASE)
        if not code_match:
            return full_block
            
        mermaid_code = code_match.group(1).strip()
        
        try:
            # Call MCP with dynamic metadata based on target
            fmt = "markdown" if output_target == "MODERATOR_VIEW" else "html"
            rendered = mcp.call_tool("render_mermaid", {
                "mermaid_code": mermaid_code, 
                "format": fmt,
                "alt": safe_alt,
                "caption": caption
            })
            
            if output_target == "MODERATOR_VIEW":
                # For Slack, bypass the <figure> wrapper entirely
                return "\n" + rendered + "\n"
            
            # Replace the <pre><code> block or just <code> block with the <img> tag
            # Flexible recursive replacement
            inner_pattern = r'<pre\s*[^>]*><code\s+class="[^"]*mermaid[^"]*">[\s\S]*?</code></pre>|' \
                            r'<code\s+class="[^"]*mermaid[^"]*">[\s\S]*?</code>'
            
            return re.sub(inner_pattern, lambda m: rendered, full_block, flags=re.IGNORECASE)
            
        except Exception as e:
            print(f"⚠️ Mermaid MCP Error: {e}")
            return full_block


    # 1. Figure Pass: Capture metadata (broadened class)
    figure_pattern = r'<figure[^>]*>(?:(?!</figure>)[\s\S])*?<code[^>]*class="[^"]*mermaid[^"]*"[^>]*>(?:(?!</figure>)[\s\S])*?</code>(?:(?!</figure>)[\s\S])*?</figure>'
    result = re.sub(figure_pattern, replacer, html_content, flags=re.IGNORECASE)
    
    # 2. Markdown Pass: Convert ```mermaid blocks into images inside figures
    def markdown_replacer(match):
        code = match.group(1).strip()
        try:
            fmt = "markdown" if output_target == "MODERATOR_VIEW" else "html"
            rendered = mcp.call_tool("render_mermaid", {"mermaid_code": code, "format": fmt})
            if output_target == "MODERATOR_VIEW":
                return "\n" + rendered + "\n"
            return f'<figure class="kg-card kg-image-card kg-width-wide">{rendered}</figure>'
        except Exception as e:
            print(f"⚠️ Mermaid Markdown Error: {e}")
            return match.group(0)

    markdown_pattern = r'```mermaid\s*([\s\S]*?)\s*```'
    result = re.sub(markdown_pattern, markdown_replacer, result, flags=re.IGNORECASE)

    # 3. Orphan Pass: Fallback for orphaned HTML code blocks
    orphan_pattern = r'<pre\s*[^>]*><code\s+class="[^"]*mermaid[^"]*">([\s\S]*?)</code></pre>|' \
                     r'<code\s+class="[^"]*mermaid[^"]*">([\s\S]*?)</code>'

    def orphan_replacer(match):
        code = (match.group(1) or match.group(2) or "").strip()
        if not code: return match.group(0)
        try:
            fmt = "markdown" if output_target == "MODERATOR_VIEW" else "html"
            # Wrap orphan images in a Ghost-friendly container by default
            rendered = mcp.call_tool("render_mermaid", {"mermaid_code": code, "format": fmt})
            if output_target == "MODERATOR_VIEW":
                return "\n" + rendered + "\n"
            return f'<figure class="kg-card kg-image-card kg-width-wide">{rendered}</figure>'
        except Exception as e:
            print(f"⚠️ Mermaid Orphan Error: {e}")
            return match.group(0)

    return re.sub(orphan_pattern, orphan_replacer, result, flags=re.IGNORECASE)

# 13c. The Chunked Refactor Helper (ADK Performance Fix)
def chunked_refactor_article(source_text, audience_context, specialist_model, style_mentors, output_target="CMS_DRAFT"):
    """
    Chunks a large article and refactors it segment-by-segment to prevent summarization drift.
    Ensures that a 3000+ word draft remains a 3000+ word article in Ghost HTML.
    """
    
    # ARCHITECTURAL FIX: Modular Instruction Assembly
    # Refactoring requires high-fidelity literary AND technical rules.
    chunk_sys = get_system_instructions("PSEO_ARTICLE", output_target)
    chunk_sys += f"\n\n{style_mentors}"

    target_instruction = ""
    if output_target == "MODERATOR_VIEW":
        target_instruction = "IMPORTANT: You are talking to a MODERATOR on Slack. Use Markdown Pipes (|) for any tables."
    else:
        target_instruction = "IMPORTANT: You are drafting for Ghost CMS. Use strictly semantic HTML <table> tags."

    print(f"  + Initiating Chunked Refactor for {len(source_text)} characters...")
    
    # Split by H2 headers, keeping the headers with the chunks
    # We use a lookahead to split and keep the delimiter
    chunks = re.split(r'(?=\n## )', source_text)
    chunks = [c.strip() for c in chunks if c.strip()]
    
    if len(chunks) <= 1 and len(source_text) > 8000:
        # Emergency Paragraph-Based Chunking for Large Text without Headers
        print("  ⚠️ No H2 headers found in large text. Using paragraph-based chunking.")
        raw_paras = source_text.split('\n\n')
        chunks = []
        buffer = []
        buffer_size = 0
        for p in raw_paras:
            buffer.append(p)
            buffer_size += len(p)
            if buffer_size > 4000:
                chunks.append('\n\n'.join(buffer))
                buffer = []
                buffer_size = 0
        if buffer: chunks.append('\n\n'.join(buffer))

    results = [None] * len(chunks)
    
    def process_chunk(idx, chunk_text):
        is_first = (idx == 0)
        chunk_refactor_prompt = f"""
        You are a Content Refactor Engine specializing in high-fidelity Ghost CMS delivery.
        
        SEGMENT: {idx+1}/{len(chunks)}
        TARGET AUDIENCE: {audience_context}
        
        TASK: Convert this 'SOURCE SEGMENT' into target GHOST-FRIENDLY HTML format.
        
        STRUCTURE REQUIREMENTS (Strict HTML):
        1.  **Semantic Structuring**: Use <section> wrappers with appropriate classes (e.g., class="intro", class="body", class="deep-dive", class="methodology").
        2.  **Semantic Headers**: Use <h1> for the main title, <h2> for major sections, and <h3> for sub-sections.
        3.  **NO NUMBERED HEADINGS**: Do NOT include numbers (e.g., "1. ", "2. ") in your <h2> or <h3> headers.
        4.  **Code Blocks**: Use semantic HTML: `<pre><code class="language-...">...</code></pre>`.
        5.  **Mermaid Diagrams**: MUST be preserved. Wrap Mermaid code in a `<figure class="kg-card kg-image-card kg-width-wide">` block. Inside, include the `<pre><code class="language-mermaid">...</code></pre>` and a context-aware `<figcaption>...</figcaption>`.
        6.  **Lists**: Use semantic `<ul><li>` or `<ol><li>`.
        7.  **Paragraphs**: Wrap all text in <p> tags.
        
        CRITICAL CONTENT RULES:
        - **Verbatim Integrity (ADK PRESERVATION)**: DO NOT summarize. Refactor the content EXACTLY as it appears in the source segment. If there are 5 paragraphs, output 5 paragraphs.
        - **Acronym Protocol**: Define all acronyms in parentheses on the first use.
        
        CRITICAL FORMATTING RULES:
        -   **NO MARKDOWN**: Absolutely NO markdown backticks (```), asterisks for bold (**), or hyphens for lists (- or *).
        -   **CLEAN CONTENT**: Remove any internal instructions or metalabels.
        -   **NO PREAMBLE**: Start directly with the first HTML tag.
        
        {target_instruction}
        
        SOURCE SEGMENT:
        {chunk_text}
        
        {extract_labeled_sources(audience_context)}
        
        CRITICAL CITATION RULE:
        - **Inline Anchored Links**: If the source text contains facts supported by Grounding Sources, you MUST use semantic HTML anchored links: `<a href="URL">Anchor Text</a>`.
        - **No Link Dumps**: Do NOT append a "Sources" list at the end.
        """
        # Lower temp for maximum literal adherence
        raw_out = safe_generate_content(specialist_model, chunk_refactor_prompt, system_instruction=chunk_sys, generation_config={"temperature": 0.1})
        return idx, sanitize_llm_html(raw_out)

    print(f"  + Launching Parallel Refactor Crew ({len(chunks)} chunks)...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_idx = {executor.submit(process_chunk, i, chunk): i for i, chunk in enumerate(chunks)}
        for future in concurrent.futures.as_completed(future_to_idx):
            idx, html_segment = future.result()
            results[idx] = html_segment
            print(f"  + [CHUNK {idx+1}] Refactor Complete.")
            
    final_html = "\n\n".join(results)
    return post_process_mermaid_to_images(final_html, output_target)

# 14. The Dedicated pSEO Article Generator
def generate_pseo_article(topic, context, history="", history_events=None, is_initial_post=True, session_id=None, output_target="CMS_DRAFT"):
    global unimodel
    style_mentors = get_stylistic_mentors(session_id)
    topic_sector = classify_topic_sector(topic, flash_model=flash_model)
    
    # --- STRATEGIC CONTEXT SANITIZATION: Strip internal meta-talk from history ---
    if output_target == "CMS_DRAFT":
        internal_keywords = [
            "competitor gap", "audit scores", "ranking analysis", "aeo strategy", 
            "moat factor", "technical density score", "turn 1", "turn 2", "turn 3", 
            "turn 4", "internal blueprint", "vetted prompt", "technical vacuum"
        ]
        for kw in internal_keywords:
            history = re.sub(rf"(?i){kw}.*?\n?", "[STRATEGIC_CONTEXT_OMITTED] ", history)
    
    target_instruction = ""
    if output_target == "MODERATOR_VIEW":
        target_instruction = "TARGET: MODERATOR_VIEW (Slack). Use Markdown Pipes (|) for tables. Use standard Slack formatting."
    else:
        target_instruction = "TARGET: CMS_DRAFT (Ghost). Use Semantic HTML <table> tags. PROHIBIT Markdown."

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
    is_repurpose_cmd = any(kw in topic_lower for kw in ["dump", "repurpose", "refactor", "publish the", "provision the", "for the draft", "finalize the"])
    
    # Path B Signals (Creative Iteration)
    is_expand_cmd = any(kw in topic_lower for kw in ["expand", "flesh out", "write from", "based on outline", "refine", "polish", "rewrite"])
    
    has_outline_structure = "## " in str(history) and len(str(history)) < 3000 # Rough heuristic: Outlines are shorter
    has_full_draft_structure = "## " in str(history) and len(str(history)) > 3000
    
    # FUTURE-PROOFING: If follow-up, favor REFACTOR/EXPAND mode if content exists.
    # Only force AUTHOR mode if no pseo/article context is found in the current turn OR history.
    if not is_initial_post and not has_pseo_keyword and not has_full_draft_structure and not has_outline_structure:
        print("  + Follow-up turn without 'pSEO' keyword or existing structure. Defaulting to AUTHOR mode for safety.")
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
            # HARDENING: Skip direct provisioning if the user explicitly asked to "refactor"
            proposal_data = event.get('proposal_data')
            if isinstance(proposal_data, dict) and proposal_data.get('article_html') and "refactor" not in topic_lower:
                print(f"  + FAST-TRACK: Found clean HTML in proposal_data. Provisioning directly.")
                return post_process_mermaid_to_images(proposal_data['article_html'], output_target)
            
            # 2. Look for existing text (Markdown) to REFACTOR
            content = event.get('text') or event.get('content')
            if content and isinstance(content, str) and len(content) > 500:
                # HARDENING: Reject Fast-Track if it contains raw markdown artifacts
                has_markdown_artifacts = "```" in content or "\n- " in content or "\n* " in content
                
                if "<section" in content and "</section>" in content and not has_markdown_artifacts and "refactor" not in topic_lower:
                    print(f"  + FAST-TRACK: Found existing high-fidelity HTML draft. Provisioning directly.")
                    return post_process_mermaid_to_images(content, output_target)
            
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
        
        # PERFORMANCE FIX (ADK-CHUNK): Handle massive drafts (>8k chars) via segmentation
        if len(source_to_refactor) > 8000:
            return chunked_refactor_article(source_to_refactor, audience_context, specialist_model, style_mentors, output_target=output_target)

        # ARCHITECTURAL FIX: Modular Instruction Assembly
        sys_instruction = get_system_instructions("PSEO_ARTICLE", output_target)
        sys_instruction += f"\n\n{style_mentors}"

        print("  + Using Single-Pass Refactor (Small Article)")
        refactor_prompt = f"""
        You are a Content Refactor Engine.
        TASK: Convert the 'SOURCE TEXT' into target GHOST-FRIENDLY HTML format.

        TARGET AUDIENCE: {audience_context}
        
        STRATEGIC REQUIREMENTS:
        - Preservation: Do NOT summarize. Refactor exactly as it appears.
        - Acronym Protocol: Define acronyms on first use.

        SOURCE TEXT:
        {source_to_refactor}
        """
        # Use Specialist (Claude Sonnet 4.5) for high-fidelity refactoring with system_instruction
        raw_html = safe_generate_content(specialist_model, refactor_prompt, system_instruction=sys_instruction, generation_config={"temperature": 0.2})
        return post_process_mermaid_to_images(sanitize_llm_html(raw_html), output_target)

    # --- PATH B & C: EXPAND / AUTHOR (Creative Synthesis) ---
    elif start_mode == "EXPAND":
        # EXPANSION PROMPT: Use history as a skeleton.
        print(f"  + Executing PATH B: EXPAND ({topic_sector}) with Unimodel")
        
        if topic_sector == "LIFESTYLE":
            persona = "Expert Travel & Lifestyle Journalist"
        elif topic_sector == "HUMANITIES":
            persona = "Sociopolitical Analyst and Researcher"
        else:
            persona = "Specialized Technical Writer"

        system_instruction = f"""
        You are an {persona}.
        TASK: Write a comprehensive pSEO article based on the provided OUTLINE/BLUEPRINT.
        
        RULES:
        1.  **FOLLOW STRUCTURE**: Strictly follow the headers defined in the 'BLUEPRINT'.
        2.  **SYNTHESIZE CONTENT**: Flesh out each bullet point into full, detailed paragraphs.
        3.  **USE RESEARCH**: Use the provided Research Context to fill in facts/data.
        
        BLUEPRINT/OUTLINE:
        {history}
        """
        tone_instruction = "Tone: High-authority, engaging, and detailed. Match the depth implied by the blueprint."
        context_block = f"Research Context:\n{context}"
        
    else: # AUTHOR MODE
        print(f"  + Executing PATH C: AUTHOR ({topic_sector}) with Unimodel")
        
        if topic_sector == "LIFESTYLE":
            persona = "Expert Travel & Lifestyle Journalist"
            goal = "Clear, warm, and inviting prose."
        elif topic_sector == "HUMANITIES":
            persona = "Sociopolitical Analyst and Researcher"
            goal = "Objective, authoritative, and nuanced prose."
        else:
            persona = "Specialized Technical Writer"
            goal = "High-authority technical article."

        if is_grounded:
            system_instruction = f"You are an {persona}. Base the article PRIMARILY on the provided 'Research Context'."
        else:
            system_instruction = f"You are an {persona}. Use the provided context to write a {goal}"
        
        tone_instruction = f"Tone: {goal} Use analogies to explain complex ideas."
        context_block = f"Research Context:\n{context}\n\nConversation History:\n{history}"

    # ARCHITECTURAL FIX: Modular Instruction Assembly
    sys_instruction = get_system_instructions("PSEO_ARTICLE", output_target, topic_sector=topic_sector)
    sys_instruction += f"\n\n{style_mentors}"

    # Shared Prompt for B & C (Creative Paths)
    prompt = f"""
    {system_instruction}
    
    AUDIENCE: {audience_context}
    TONE & STYLE: {tone_instruction}
    
    CONTEXTUAL DATA:
    Current Topic: "{topic}"
    
    {context_block}
    
    {extract_labeled_sources(context_block)}
    """
    
    raw_response = safe_generate_content(unimodel, prompt, system_instruction=sys_instruction, generation_config={"temperature": 0.4})
    return post_process_mermaid_to_images(sanitize_llm_html(raw_response), output_target)

def generate_pseo_page(topic, context, history="", history_events=None, is_initial_post=True, session_id=None, output_target="CMS_DRAFT"):
    """
    Generates a data-rich, non-narrative pSEO page for a specific entity/location.
    Uses reverse-engineered logic from generate_pseo_article for robust AUTHOR/REFACTOR paths.
    """
    global unimodel, specialist_model
    
    topic_sector = classify_topic_sector(topic, flash_model=flash_model)
    topic_lower = topic.lower()
    style_mentors = get_stylistic_mentors(session_id)
    audience_context = detect_audience_context(history)
    is_grounded = any("GROUNDING_CONTENT" in str(c) for c in context)

    # 1. Mode Detection (Reverse-engineered from generate_pseo_article)
    # Tri-State Logic: REFACTOR vs EXPAND vs AUTHOR
    start_mode = "AUTHOR"
    
    # Path A Signals (Verbatim Finalization)
    is_repurpose_cmd = any(kw in topic_lower for kw in ["dump", "repurpose", "re-purpose", "refactor", "correct", "fix", "update", "refine", "publish", "provision", "finalize"])
    
    # Path B Signals (Creative Iteration)
    is_expand_cmd = any(kw in topic_lower for kw in ["expand", "flesh out", "write from", "based on outline", "polish", "rewrite"])
    
    has_outline_structure = "## " in str(history) and len(str(history)) < 3500
    has_full_draft_structure = ("## " in str(history) or "<table>" in str(history)) and len(str(history)) > 3500

    if not is_initial_post and not is_repurpose_cmd and not is_expand_cmd and not has_full_draft_structure and not has_outline_structure:
        start_mode = "AUTHOR"
    elif is_repurpose_cmd and (has_full_draft_structure or not is_initial_post):
        start_mode = "REFACTOR"
    elif (is_expand_cmd and "## " in str(history)) or (has_outline_structure and not is_repurpose_cmd):
        start_mode = "EXPAND"
    elif is_repurpose_cmd and has_outline_structure:
        start_mode = "EXPAND" 
    elif not is_initial_post:
        start_mode = "REFACTOR"
        
    print(f"  + pSEO Page Generation Mode Selected: {start_mode}")

    # 2. De-copy-cat: Disable Article-style "Fast-Track" for Pages
    # Unlike Articles, Pages only deliver synthesis/insights to Slack, not raw HTML drafts.
    # We must ALWAYS refactor based on the holistic thread history.
    best_source_text = None
    if history:
         best_source_text = history
         print(f"  + Holistic Grounding: Using full thread history ({len(history)} chars) for refactor/update.")

    # 3. Instruction Assembly
    sys_instruction = get_system_instructions("PSEO_PAGE", output_target, topic_sector=topic_sector)
    sys_instruction += f"\n\n{style_mentors}"

    # 4. Prompt Engineering
    if start_mode == "REFACTOR":
        print("  + Executing pSEO Page REFACTOR with Specialist Model")
        source_to_refactor = best_source_text or history
        
        prompt = f"""
        You are a Data Page Refactor Engine.
        TASK: Convert/Update the 'SOURCE TEXT' into a high-fidelity pSEO Data Page.
        
        AUDIENCE: {audience_context}
        
        SOURCE TEXT:
        {source_to_refactor}
        """
        # Use Specialist for updates
        raw_response = safe_generate_content(specialist_model, prompt, system_instruction=sys_instruction, generation_config={"temperature": 0.2})
    elif start_mode == "EXPAND":
        print("  + Executing pSEO Page EXPAND with Unimodel")
        # EXPANSION PROMPT: Use history as a skeleton.
        prompt = f"""
        You are a Specialized Data Page Writer.
        TASK: Write a comprehensive pSEO entity page based on the provided OUTLINE/STRUCTURE.
        
        AUDIENCE: {audience_context}
        
        STRUCTURE/OUTLINE:
        {history}
        
        REQUIRED METRICS:
        You MUST include/address the following 4 core market metrics in your expansion:
        1. hub_capital
        2. spend_trend
        3. market_tier
        4. region
        
        Research Context:
        {context}
        """
        raw_response = safe_generate_content(unimodel, prompt, system_instruction=sys_instruction, generation_config={"temperature": 0.2})
    else:
        print("  + Executing pSEO Page AUTHOR with Unimodel")
        context_block = f"Research Context:\n{context}" if is_grounded else "Use internal knowledge."
        
        prompt = f"""
        TOPIC: {topic}
        TASK: Generate a NEW data-rich pSEO page.
        
        AUDIENCE: {audience_context}
        
        {context_block}
        """
        raw_response = safe_generate_content(unimodel, prompt, system_instruction=sys_instruction, generation_config={"temperature": 0.2})

    return post_process_mermaid_to_images(sanitize_llm_html(raw_response), output_target)


# 14.2 The LP Page Generator (Long-Form Pillar Page)
def generate_lp_page(topic, context, history="", history_events=None, is_initial_post=True, session_id=None, output_target="CMS_DRAFT"):
    """
    Generates a long-form Landing Page (LP) for a specific framework, concept, or strategy.
    Uses the same REFACTOR / EXPAND / AUTHOR tri-state as generate_pseo_page(), but prompts
    target 3,000+ word depth and never reference market geo fields.
    """
    global unimodel, specialist_model

    topic_sector = classify_topic_sector(topic, flash_model=flash_model)
    topic_lower = topic.lower()
    style_mentors = get_stylistic_mentors(session_id)
    audience_context = detect_audience_context(history)
    is_grounded = any("GROUNDING_CONTENT" in str(c) for c in context)

    # 1. Tri-State Mode Detection (mirrors generate_pseo_page logic)
    start_mode = "AUTHOR"
    is_repurpose_cmd = any(kw in topic_lower for kw in ["dump", "repurpose", "re-purpose", "refactor", "correct", "fix", "update", "refine", "publish", "provision", "finalize"])
    is_expand_cmd = any(kw in topic_lower for kw in ["expand", "flesh out", "write from", "based on outline", "polish", "rewrite"])
    has_outline_structure = "## " in str(history) and len(str(history)) < 3500
    has_full_draft_structure = ("## " in str(history) or "<table>" in str(history)) and len(str(history)) > 3500

    if not is_initial_post and not is_repurpose_cmd and not is_expand_cmd and not has_full_draft_structure and not has_outline_structure:
        start_mode = "AUTHOR"
    elif is_repurpose_cmd and (has_full_draft_structure or not is_initial_post):
        start_mode = "REFACTOR"
    elif (is_expand_cmd and "## " in str(history)) or (has_outline_structure and not is_repurpose_cmd):
        start_mode = "EXPAND"
    elif is_repurpose_cmd and has_outline_structure:
        start_mode = "EXPAND"
    elif not is_initial_post:
        start_mode = "REFACTOR"

    print(f"  + LP Page Generation Mode Selected: {start_mode}")

    # Always refactor based on holistic thread history for LPs
    best_source_text = None
    if history:
        best_source_text = history
        print(f"  + Holistic Grounding: Using full thread history ({len(history)} chars) for refactor/update.")

    # 2. Instruction Assembly
    # Port from PSEO_PAGE to PSEO_ARTICLE intent to ensure narrative depth/high-fidelity persona
    sys_instruction = get_system_instructions("PSEO_ARTICLE", output_target, topic_sector=topic_sector)
    sys_instruction += f"\n\n{style_mentors}"

    # 3. Sector-Aware Persona Determination (Mirroring Article Logic)
    if topic_sector == "LIFESTYLE":
        persona = "Expert Travel & Lifestyle Journalist"
        goal = "Clear, warm, and inviting prose."
    elif topic_sector == "HUMANITIES":
        persona = "Sociopolitical Analyst and Researcher"
        goal = "Objective, authoritative, and nuanced prose."
    else:
        persona = "Pillar Page Strategist"
        goal = "High-authority, 3,000+ word technical framework landing page."

    context_block = f"Research Context for Depth and Citations:\n{context}" if is_grounded else "Use internal knowledge and expert synthesis."
    
    # 4. Prompt Engineering
    if start_mode == "REFACTOR":
        print("  + Executing LP Page REFACTOR with Specialist Model")
        source_to_refactor = best_source_text or history
        labeled_sources = extract_labeled_sources(str(context) + "\n" + str(source_to_refactor))
        prompt = f"""
        You are a Pillar Page Architect.
        TASK: Refactor/Upgrade the 'SOURCE TEXT' into a definitive 3,000+ word strategy pillar.

        AUDIENCE: {audience_context}
        PERSONA: {persona} ({goal})

        STRATEGIC REQUIREMENTS:
        - Target: 3,000+ words minimum
        - Focus on narrative authority and structural clarity.
        - Ensure every H2 section is substantive and contains unique technical depth.

        SOURCE TEXT:
        {source_to_refactor}

        {context_block}

        {labeled_sources}
        """
        raw_response = safe_generate_content(specialist_model, prompt, system_instruction=sys_instruction, generation_config={"temperature": 0.2})

    elif start_mode == "EXPAND":
        print("  + Executing LP Page EXPAND with Unimodel")
        labeled_sources = extract_labeled_sources(str(context) + "\n" + str(history))
        prompt = f"""
        You are an {persona}.
        TASK: Write a comprehensive long-form landing page based on the OUTLINE/STRUCTURE below.

        AUDIENCE: {audience_context}
        GOAL: {goal}

        STRUCTURE/OUTLINE:
        {history}

        STRATEGIC REQUIREMENTS:
        - Target: 3,000+ words minimum.
        - Every H2 section must be substantive — no single-paragraph stubs.
        - Focus on narrative framework and persuasive storytelling.

        {context_block}

        {labeled_sources}
        """
        raw_response = safe_generate_content(unimodel, prompt, system_instruction=sys_instruction, generation_config={"temperature": 0.3})

    else:
        print("  + Executing LP Page AUTHOR with Unimodel")
        prompt = f"""
        You are an {persona}.
        TOPIC: {topic}
        TASK: Write a NEW long-form, high-authority landing page on this topic.

        AUDIENCE: {audience_context}
        GOAL: {goal}

        STRATEGIC REQUIREMENTS:
        - Minimum 3,000 words.
        - Every H2 section must be substantive and authoritative.

        {context_block}

        {extract_labeled_sources(context_block)}
        """
        raw_response = safe_generate_content(unimodel, prompt, system_instruction=sys_instruction, generation_config={"temperature": 0.4})

    return post_process_mermaid_to_images(sanitize_llm_html(raw_response), output_target)

# 14.5 The Recursive Deep-Dive Generator (Dynamic Room-by-Room Construction)
def generate_deep_dive_article(topic, context, history="", history_events=None, target_length=1500, target_geo="Global", session_id=None, output_target="MODERATOR_VIEW"):
    global unimodel
    style_mentors = get_stylistic_mentors(session_id)
    topic_sector = classify_topic_sector(topic, flash_model=flash_model)
    
    target_instruction = ""
    if output_target == "MODERATOR_VIEW":
        target_instruction = "TARGET: MODERATOR_VIEW (Slack). Use Markdown Pipes (|) for tables. Use standard Slack formatting."
    else:
        target_instruction = "TARGET: CMS_DRAFT (Ghost). Use Semantic HTML <table> tags. PROHIBIT Markdown."
    print(f"Tool: Initiating Recursive Deep-Dive for '{topic}' (Region: {target_geo}, Target: {target_length} words)")
    
    # CLAMP: Prevent unbounded generation that crashes SSL/Webhooks
    target_length = min(3500, max(500, target_length))
    
    is_grounded = any("GROUNDING_CONTENT" in str(c) for c in context)
    audience_context = detect_audience_context(history)
    
    # ARCHITECTURAL FIX: Modular Instruction Assembly
    blueprint_sys = get_system_instructions("BLOG_OUTLINE", output_target, topic_sector=topic_sector)
    blueprint_sys += f"\n\n{style_mentors}"

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
    
    ### PROTOCOL:
    - **HUMAN FINGERPRINT**: Mandate punctuation diversity (semicolons/parentheses).
    - **ANTI-WATERMARK**: Ban buzzwords (delve, tapestry, robust).
    - **GRITTY REALISM**: Mandate the mention of "operational friction" or "localized hurdles."
    
    ### STEP 3: BLUEPRINT GENERATION
    Design the optimal structure.
    - **Sections**: Create between 3 and 8 H2 sections depending on the complexity of the topic.
    - **NO REDUNDANCY**: Do NOT include a "Lede", "Introduction", or "Conclusion" section in your list. These parts are handled by special drafting phases. Start directly with the core problem or first technical insight.
    - **WORD COUNT BUDGETING**: You MUST assign a `target_word_count` to each section. The SUM of all these counts MUST be approximately {target_length - 300}. 
    - **Strategic Allocation**: Allocate more words to complex case studies/ROOT CAUSE analysis and fewer to simple definitions/hooks.
    
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
        blueprint_raw = safe_generate_content(unimodel, blueprint_prompt, system_instruction=blueprint_sys)
        if "```json" in blueprint_raw: blueprint_raw = blueprint_raw.split("```json")[1].split("```")[0].strip()
        blueprint = json.loads(blueprint_raw)
        title = blueprint.get("title", f"The Analysis of {topic}")
        sections = blueprint.get("sections", [])
    except Exception as e:
        print(f"Blueprint Error: {e}")
        # Fallback to pSEO if blueprinting fails
        return generate_pseo_article(topic, context, history, history_events=history_events, is_initial_post=True, session_id=session_id, output_target=output_target)

    # PHASE 2: Recursive Room Building (The Writer - PARALLELIZED)
    article_parts = [f"<h1>{title}</h1>"]
    
    # Intro (Sequential - Sets the stage)
    # ARCHITECTURAL FIX: Modular Instruction Assembly
    intro_sys = get_system_instructions("DEEP_DIVE", output_target, topic_sector=topic_sector)
    intro_sys += f"\n\n{style_mentors}"

    intro_prompt = f"""
    Write a compelling {intro_words}-word intro for: '{title}'.
    REGION: {target_geo}
    
    AUDIENCE: {audience_context}
    GOAL: {topic}
    
    {extract_labeled_sources(context if is_grounded else "")}
    """
    intro_raw = safe_generate_content(unimodel, intro_prompt, system_instruction=intro_sys)
    article_parts.append(f'<section class="intro">\n{sanitize_llm_html(intro_raw)}\n</section>')
    
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
            
        # ARCHITECTURAL FIX: Modular Instruction Assembly
        section_sys = get_system_instructions("DEEP_DIVE", output_target, topic_sector=topic_sector)
        section_sys += f"\n\n{style_mentors}"

        # NOTE: In parallel mode, we rely on the Blueprint logic rather than the literal previous text.
        room_prompt = f"""
        Write a {section_words}-word section for: "{title}".
        REGION: {target_geo}
        
        CHAPTER {index+1}/{total_sections}: {section['title']}
        INSTRUCTION: {instruction}
        
        GROUNDING DATA: {context if is_grounded else "Internal Knowledge base"}
        {extract_labeled_sources(context if is_grounded else "")}
        
        ### CRITICAL CITATION RULE:
        - **Inline Anchored Links**: When referencing facts or data points from the GROUNDING SOURCES, you MUST use semantic HTML anchored links: `<a href="URL">Anchor Text</a>`.
        
        OUTPUT: Start with <h2>{section['title']}</h2> then the content in semantic HTML (no markdown). Wrap paragraphs in <p>.
        """
        content = safe_generate_content(unimodel, room_prompt, system_instruction=section_sys)
        return index, f'<section class="body-part">\n{sanitize_llm_html(content)}\n</section>'

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
    # ARCHITECTURAL FIX: Modular Instruction Assembly
    conc_sys = get_system_instructions("DEEP_DIVE", output_target)
    conc_sys += f"\n\n{style_mentors}"

    conc_prompt = f"""
    Write a {conc_words}-word concluding thought for '{title}'. 
    Audience: {audience_context}. 
    Summary of key impact.
    
    OUTPUT: Return ONLY the content in semantic HTML (no markdown). Wrap paragraphs in <p>. Do NOT include a header.
    """
    conc_raw = safe_generate_content(unimodel, conc_prompt, system_instruction=conc_sys)
    article_parts.append(f'<section class="conclusion">\n<h2>Final Reflection</h2>\n{sanitize_llm_html(conc_raw)}\n</section>')

    # PHASE 3: Methodology
    print("  + Finalizing Methodology & Transparency...")
    # Calculate interim word count for the footer
    current_content = " ".join(article_parts)
    est_words = len(current_content.split())

    methodology_text = f"""
    <section class="methodology">
    <h2>Methodology & Transparency</h2>
    <p>This {est_words}-word technical analysis was recursively architected using a multi-agent orchestration framework. Concepts were synthesized through a combination of contextual grounding and forensic technical auditing to ensure architectural accuracy.</p>
    </section>
    """
    article_parts.append(methodology_text)

    full_html = "\n\n".join(article_parts)
    print(f"  -> Deep-Dive Complete. Est words: {len(full_html.split())}")
    return post_process_mermaid_to_images(full_html, output_target)

#15. The Euphemistic 'Then vs Now' Linker
def create_euphemistic_links(keyword_context, is_initial=True, session_id=None):
    global unimodel
    style_mentors = get_stylistic_mentors(session_id)
    
    state_context = "This is a NEW proposal for a new thread." if is_initial else "This is a REVISION or EXTENSION of a previous strategy in an existing thread."
    
    # ARCHITECTURAL FIX: Modular Instruction Assembly
    # Intent for Then/Now is naturally technical but formatted for Slack
    sys_instruction = get_system_instructions("SIMPLE_QUESTION", "MODERATOR_VIEW")
    sys_instruction += f"\n\n{style_mentors}"
    
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
    
    # Use system_instruction layer
    response = safe_generate_content(unimodel, prompt, system_instruction=sys_instruction)
    return extract_json(response)

#15. The Euphemistic 'Then vs Now' Linker
def tell_then_and_now_story(interlinked_concepts, tool_confirmation=None, session_id=None, output_target="MODERATOR_VIEW"):
    if not tool_confirmation or not tool_confirmation.get("confirmed"): raise PermissionError("Wait for approval.")
    style_mentors = get_stylistic_mentors(session_id)
    
    # ARCHITECTURAL FIX: Modular Instruction Assembly
    sys_instruction = get_system_instructions("SIMPLE_QUESTION", output_target)
    sys_instruction += f"\n\n{style_mentors}"

    prompt = f"TASK: Tell a 'Then and Now' story using these concepts: {interlinked_concepts}"
    return safe_generate_content(unimodel, prompt, system_instruction=sys_instruction)

#16. The Proposal Critic and Refiner
def critique_proposal(topic, current_proposal):
    global unimodel
    prompt = f"Review proposal for '{topic}': {json.dumps(current_proposal, indent=2)}. If excellent, respond: APPROVED. Else, provide concise feedback to improve 'Then vs Now' contrast."
    return safe_generate_content(unimodel, prompt)

#17. The Proposal Refiner
def refine_proposal(topic, current_proposal, critique, session_id=None):
    global unimodel
    style_mentors = get_stylistic_mentors(session_id)
    
    # ARCHITECTURAL FIX: Modular Instruction Assembly
    sys_instruction = get_system_instructions("SIMPLE_QUESTION", "MODERATOR_VIEW")
    sys_instruction += f"\n\n{style_mentors}"

    prompt = f"""
    REWRITE proposal. Topic: {topic}. Draft: {json.dumps(current_proposal)}. Critique: {critique}. 
    Preserve keys: 'then_concept', 'now_concept', 'link'. Ensure JSON format.
    """
    return extract_json(safe_generate_content(unimodel, prompt, system_instruction=sys_instruction))

# 18. Phase 1: Sales-to-Content Pipeline
def process_sales_transcript(transcript_text, output_target="MODERATOR_VIEW"):
    """
    Extracts customer objections and generates solution brief.
    """
    global specialist_model, db
    
    style_mentors = get_stylistic_mentors()
    
    # ARCHITECTURAL FIX: Modular Instruction Assembly
    # Extraction phase uses lightweight reasoning.
    extraction_sys = get_system_instructions("SIMPLE_QUESTION", "MODERATOR_VIEW")
    extraction_sys += f"\n\n{style_mentors}"

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
    
    CRITICAL: Valid JSON ONLY. If a category has no data, use an empty array [] or "Not discussed".
    """
    
    try:
        # Use Specialist Model for high-fidelity extraction
        objections_raw = safe_generate_content(specialist_model, objection_prompt, system_instruction=extraction_sys)
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
    
    # ARCHITECTURAL FIX: Modular Instruction Assembly
    # Briefing phase uses high-fidelity rules.
    brief_sys = get_system_instructions("TECHNICAL_EXPLANATION", output_target)
    brief_sys += f"\n\n{style_mentors}"

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
    
    ### FORMATTING RULES (PRIORITY):
    1. Use **bold** for emphasis.
    2. Tables: IF MODERATOR_VIEW, use Markdown Pipes (|). IF CMS_DRAFT, use HTML <table>.
    3. **Code Blocks**: If including technical/code examples, use markdown fenced code blocks:
       - Use triple backticks with language identifier: ```language
    """
    
    brief_raw = safe_generate_content(specialist_model, brief_prompt, system_instruction=brief_sys)
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
    try:
        return _process_story_logic_inner(request)
    except Exception as e:
        print(f"❌ FATAL ERROR CAUGHT AT TOP LEVEL: {e}")
        from flask import jsonify
        return jsonify({"error": "Fatal process error", "details": str(e)}), 200

@functions_framework.http
def _process_story_logic_inner(request):
    global unimodel, flash_model, research_model, specialist_model, db

    # 0. ENTRY TELEMETRY (Observability)
    now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[MISSION START] {now_str} | Function: process_story_logic")

    # 0.5 CORE SERVICES (Pre-flight init)
    if any(m is None for m in [unimodel, flash_model, research_model, specialist_model]):
        vertexai.init(project=PROJECT_ID, location=LOCATION)
    # Vertex AI gRPC Mitigations (Pool Reset)
    vertexai.init(project=PROJECT_ID, location=LOCATION)

    if unimodel is None:
        unimodel = UnifiedModel(MODEL_PROVIDER, MODEL_NAME)
    
    if flash_model is None:
        # Standardize: Use UnifiedModel wrapper instead of raw GenerativeModel
        flash_model = UnifiedModel("vertex_ai", FLASH_MODEL_NAME)

    if research_model is None:
        research_model = UnifiedModel("vertex_ai", RESEARCH_MODEL_NAME)
        
    if specialist_model is None:
        # STRICT: User enforced Claude 3.5 Sonnet (using 4.5 alias as requested)
        specialist_model = UnifiedModel(provider="anthropic", model_name="claude-sonnet-4-5")
        
    if db is None:
        db = firestore.Client(project=PROJECT_ID)

    # Late-Init for MCP (Shared Resilient Implementation) 🛡️🏗️🛡️
    get_mcp_client()

    req = request.get_json(silent=True)
    if not req:
        print("🛑 ERROR: Story Worker received empty or malformed payload.")
        return jsonify({"error": "Missing payload"}), 400

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
    # NORMALIZE: Ensure Slack Context is usable
    slack_context = req.get('slack_context', {}).copy()
    if not slack_context.get('channel'):
        slack_context['channel'] = req.get('slack_channel') or req.get('channel')
    if not slack_context.get('ts'):
        slack_context['ts'] = req.get('slack_ts') or req.get('ts')

    unique_event_id = slack_context.get('ts') or req.get('client_msg_id')
    
    if unique_event_id:
        # Check if we've already seen this event ID in the last 30 minutes
        dedup_ref = db.collection('processed_events').document(str(unique_event_id))
        
        # NATIVE RESCUE: Try/Except wrap the synchronous read to catch TSI_DATA_CORRUPTED
        try:
            doc = _firestore_call_with_timeout(lambda: dedup_ref.get())
        except Exception as e:
            print(f"⚠️ FIRESTORE READ ERROR ({e}). Attempting channel resuscitation...")
            # Forcing a fresh client instantiation specifically for this container runtime
            db = firestore.Client(project=PROJECT_ID)
            dedup_ref = db.collection('processed_events').document(str(unique_event_id))
            doc = _firestore_call_with_timeout(lambda: dedup_ref.get())
            print("✅ Channel resuscitation successful!")
            
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
            _firestore_call_with_timeout(lambda: dedup_ref.set({
                'timestamp': datetime.datetime.now(datetime.timezone.utc),
                'status': 'processing',
                'session_id': req.get('session_id')
            }))
        except Exception as e:
            print(f"⚠️ Warning: Failed to set idempotency lock: {e}")
            
    session_id, original_topic = req['session_id'], req.get('topic', "")
    feedback_text = req.get('feedback_text') # Captured from worker-feedback
    images = req.get('images', [])
    code_files = req.get('code_files', []) # ADD: Extract code_files
    
    # Use the Slack-mimicking logic: if there's no thread_ts OR thread_ts == ts, AND no feedback_text, it's the root message (initial).
    is_initial_post = (not slack_context.get('thread_ts') or slack_context.get('thread_ts') == slack_context.get('ts')) and not feedback_text
    print(f"DEBUG: Thread State -> is_initial_post: {is_initial_post} (ts: {slack_context.get('ts')}, thread_ts: {slack_context.get('thread_ts')})")
    
    # --- STEP 1: PERCEPTION (Sanitize Input) ---
    sanitized_topic = sanitize_input_text(feedback_text or original_topic)
    clean_topic = sanitized_topic # Consistency alias 🛡️🩹🛡️
    expire_time = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=30)
    
    # --- STEP 2: LOAD SHORT-TERM MEMORY ---
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
    if session_doc.exists:
        session_data = session_doc.to_dict()
        
    # LIGHT ARCHITECTURE: Context Inheritance (Dispatcher-Anchored)
    # Hardened merge logic: Prevent nulls in trigger 'req' from overriding valid session data.
    session_slack = session_data.get('slack_context', {})
    req_slack = req.get('slack_context', {})
    slack_context = session_slack.copy()
    for k, v in req_slack.items():
        if v is not None:
            slack_context[k] = v
            
    # CRITICAL: If the trigger itself has channel/ts, they take precedence
    if not slack_context.get('channel'):
        slack_context['channel'] = req.get('slack_channel') or req.get('channel') or req.get('event', {}).get('channel')
    if not slack_context.get('ts'):
        slack_context['ts'] = req.get('slack_ts') or req.get('ts') or req.get('event', {}).get('ts')

    # --- ANCHOR SESSION (Early Initialization) ---
    # We commit the root document immediately to prevent "headless" sessions 
    # if a crash occurs during the research/generation phase.
    if not session_doc.exists:
        print(f"❌ SESSION NOT FOUND: {session_id}")
        return jsonify({"error": "Session not found"}), 404

    history_events = []
    history_text = ""
    if session_doc.exists:
        # MEMORY EXPANSION: Removed .limit(20) to ensure deep context recall
        events_ref = doc_ref.collection('events')
        
        # Chronological order (Old -> New)
        query = events_ref.order_by('timestamp', direction=firestore.Query.ASCENDING)
        
        # CLAMPING: Limit history to last 15 turns to prevent context bloat
        # 30 matches (15 turns of User/Agent pairs)
        all_events = _firestore_call_with_timeout(lambda: [doc.to_dict() for doc in query.stream()], timeout_secs=20)
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
            Classify the PRIMARY GOAL into one of the categories below.
            
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

            5.  **PSEO_ARTICLE**: **Ghost CMS Content Hook (Narrative).** Select this whenever the user includes the super-keywords "**pSEO**", "**pSEO Ghost**", or "**pSEO Ghost CMS**" in their request.
                *   *Super-Keyword Rule:* If the prompt contains "pSEO", "pSEO Ghost", or "pSEO Ghost CMS" (WITHOUT the word "Page"), this is MANDATORY.
                *   *Dichotomy Rule:* Use this for remote publication/updates of narrative articles.
                *   *Triggers:* "Update the pSEO draft", "Push this pSEO article to Ghost".

            6.  **PSEO_PAGE**: **Data-Rich Page Generation (Non-Narrative).** Select this if the user includes the super-keywords "**pSEO Ghost Page**", "**pSEO Ghost CMS Page**".
                *   *Super-Keyword Rule:* If the prompt contains "pSEO Ghost Page" or "pSEO Ghost CMS Page", this is MANDATORY.
                *   *Triggers:* "Create a pSEO Ghost Page for NYC", "Generate the entity page on pSEO Ghost CMS", "Build the location specific page on pSEO Ghost CMS".

            7.  **PSEO_LP**: **Long-Form Landing Page (Pillar/Hub Page).** Select this when the user wants a permanent, high-authority, long-form page (3,000+ words) on a framework, concept, or strategy — NOT a market-data entity page.
                *   *Super-Keyword Rule:* If the prompt contains "**Ghost LP**", "**LP Ghost Page**", "**Ghost Landing Page**", or "**LP Ghost CMS**", this is MANDATORY.
                *   *Triggers:* "Create a Ghost LP for the Privacy Framework", "Build the landing page on Ghost CMS", "Write a pillar page for Ghost LP".
                *   *Dichotomy Rule:* Use PSEO_PAGE for geo/entity data cards. Use PSEO_LP for evergreen framework/strategy pages.

            8.  **SALES_TRANSCRIPT**: **Sales Analysis.** Select this if the user provides a transcript or asks to analyze a sales call for objections, solution briefs, or deal coaching.
                *   *Triggers:* "Analyze this call", "Create a solution brief", "Extract objections", "What were the pain points?".
                *   *Rule:* If the user requests a "Solution Brief" from text, select this.

            8.  **SIMPLE_QUESTION**: **The Factual Scout.** For fact-checks, definitions, or retrieving specific data points that require external grounding or RAG search.
                *   *Triggers:* "What is...", "Explain the concept of...", "Find me the rules for...", "Check the status of...".
                *   *Rule:* Use this for initial questions about the world. If the user asks for a STRUCTURE (Outline/Brief), pick **OPERATIONAL_REFORMAT** instead.

            9.  **DIRECT_ANSWER**: **The Research Architect.** Select this ONLY for **NOVEL SYNTHESIS** or **TECHNICAL AUDITS** that require combining multiple external data sources into a new expert opinion. 
                *   *Example:* "Analyze the security layer of AP2 vs standard PCI-DSS." 

            10. **OPERATIONAL_REFORMAT**: **The Efficiency Engine.** Select this for any structural command whose COMPLETE context is already available in the history.
                *   *Triggers:* "Summarize our progress", "Reformat these points as bullets", "Make it more professional", "Convert this into a brief".
                *   *CRITICAL:* If the request is purely about structural formatting of existing knowledge, select this.
                *   *EXCEPTION:* If the user says "Repurpose... for pSEO" or "Create a pSEO article", select **PSEO_ARTICLE**. 

            11. **BLOG_OUTLINE**: **The Content Blueprint.** Select this for requests to create, repurpose, or draft a Blog Outline, Content Structure, or Strategy Document.
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

            CRITICAL: Respect the "Action Priority" rule. Update and Refactor requests for Ghost/pSEO content must stay in the pSEO pipeline. Select OPERATIONAL_REFORMAT for Outlines, checklists, or formatting requests using existing data. Select DIRECT_ANSWER ONLY for new, complex synthesis of external data. Respond with ONLY the category name.

            """
        # --- EXECUTION: Triage Model Selection ---
        # Triage on the Command (sanitized_topic) but with history context
        # Use Flash Model for Speed/Cost Efficiency (Gemini 2.0 Flash)
        intent = safe_generate_content(flash_model, triage_prompt)
        
        # ADK HARDENING: Deterministic Triage Override (Keyword Overlord)
        # This prevents the LLM from misclassifying Ghost/pSEO requests as Operational Reformats.
        keyword_payload = sanitized_topic.lower()

        # PSEO_LP: Force intent for Landing Page requests (Highest priority — checked first)
        if any(kw in keyword_payload for kw in ["ghost lp", "lp ghost", "ghost landing page", "landing page cms", "lp ghost cms", "ghost lp page"]):
            if "PSEO_LP" not in intent:
                print(f"🎯 TRIAGE OVERRIDE: Forcing PSEO_LP due to keywords in '{sanitized_topic[:30]}'")
                intent = "PSEO_LP"

        # PSEO_PAGE: Force intent for explicit entity-level page requests
        elif any(kw in keyword_payload for kw in ["pseo ghost page", "pseo ghost cms page"]):
            if "PSEO_PAGE" not in intent:
                print(f"🎯 TRIAGE OVERRIDE: Forcing PSEO_PAGE due to keywords in '{sanitized_topic[:30]}'")
                intent = "PSEO_PAGE"

        # PSEO_ARTICLE: Preserved logic (Noun base is sufficient as per user yardstick)
        elif any(kw in keyword_payload for kw in ["pseo", "pseo ghost", "pseo ghost cms"]):
            if "PSEO_ARTICLE" not in intent and "PSEO_PAGE" not in intent and "PSEO_LP" not in intent:
                print(f"🎯 TRIAGE OVERRIDE: Forcing PSEO_ARTICLE due to keywords in '{sanitized_topic[:30]}'")
                intent = "PSEO_ARTICLE"

        print(f"TELEMETRY: Triage V6.0 -> Intent classified as: [{intent}] for command: '{sanitized_topic[:50]}...'")

        # Initialize variables for the response
        new_events = [{"event_type": "user_request", "text": sanitized_topic, "slack_context": slack_context}]
        
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
        
        # Determine global operation type for all N8N payloads
        ghost_post_id = session_data.get('ghost_post_id') if isinstance(session_data, dict) else None
        global_operation_type = get_n8n_operation_type(intent, original_topic, sanitized_topic, ghost_post_id)
        
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
                _firestore_call_with_timeout(lambda: events_ref.add(event))

            # 2. Update Parent Metadata (Preserving Context)
            _firestore_call_with_timeout(lambda: session_ref.update({
                "status": "completed",
                "type": "social",
                "last_updated": expire_time
            }))
            target = get_output_target(intent)
            safe_n8n_delivery({
                "session_id": session_id, 
                "type": global_operation_type, 
                "message": reply_text,
                "query": sanitized_topic,
                "output_target": target,
                "channel_id": slack_context.get('channel') or slack_context.get('channel_id') or req.get('channel') or req.get('slack_channel') or req.get('event', {}).get('channel'), 
                "thread_ts": slack_context.get('ts') or slack_context.get('thread_ts'), 
                "is_initial_post": is_initial_post
            })
            return jsonify({"msg": "Social reply sent"}), 200

        # === PATH B: OPERATIONAL REFORMAT / FAST FOLLOW-UP (Fast Exit) ===
        elif intent == "OPERATIONAL_REFORMAT":
            # GROUNDING GUARD: We only "Fast-Exit" if we actually have history to reformat.
            # If this is the first turn and the topic is new, we MUST research.
            has_substantive_history = len(history_events) > 0 or len(sanitized_topic) > 100 
            
            if has_substantive_history:
                print(f"Executing Operational Reformat (Fast Exit) for command: {sanitized_topic[:50]}")
                # Determined Output Target
                target = get_output_target(intent)
                
                # RESTORED: Use simpler natural answer generator to avoid persona-induced bloat
                answer_data = generate_natural_answer(sanitized_topic, "COMPLETE_CONTEXT_IN_HISTORY", history=history_text, session_id=session_id, output_target=target, intent_label=intent)
                answer_text = answer_data['text']
                research_intent = answer_data.get('intent', "OPERATIONAL_REFORMAT")
                
                new_events.append({"event_type": "agent_answer", "text": answer_text, "intent": "OPERATIONAL_REFORMAT"})
                
                # Writing to Session Memory
                session_ref = db.collection('agent_sessions').document(session_id)
                events_ref = session_ref.collection('events')
                for event in new_events:
                    if 'timestamp' not in event: event['timestamp'] = datetime.datetime.now(datetime.timezone.utc)
                    _firestore_call_with_timeout(lambda: events_ref.add(event))

                _firestore_call_with_timeout(lambda: session_ref.update({
                    "status": "completed",
                    "type": "operational_answer",
                    "last_updated": expire_time
                }))
                
                safe_n8n_delivery({
                    "session_id": session_id, 
                    "type": global_operation_type, 
                    "message": answer_text, 
                    "query": sanitized_topic,
                    "intent": "OPERATIONAL_REFORMAT",
                    "output_target": target,
                    "channel_id": slack_context.get('channel') or req.get('channel') or req.get('slack_channel') or req.get('event', {}).get('channel'), 
                    "thread_ts": slack_context.get('thread_ts') or slack_context.get('ts'),
                    "is_initial_post": is_initial_post
                })
                return jsonify({"msg": "Operational reformat sent"}), 200
            else:
                print(f"OPERATIONAL_REFORMAT fallback: Insufficient grounding. Routing to Research Path.")
                # Fall through to Path D (Work/Research)
                pass

        # === PATH CX: BLOG OUTLINE (High-Fidelity Structure) ===
        elif intent == "BLOG_OUTLINE":
            print(f"Executing Blog Outline generation for command: {sanitized_topic[:50]}")
            # Determined Output Target
            target = get_output_target(intent)
            
            # Use Comprehensive Content Strategist for outlines
            answer_data = generate_comprehensive_answer(sanitized_topic, "BLOG_OUTLINE_MODE", history=history_text, context_topic=original_topic, session_id=session_id, output_target=target, intent_label=intent)
            answer_text = answer_data['text']
            
            new_events.append({"event_type": "agent_answer", "text": answer_text, "intent": "BLOG_OUTLINE"})
            
            session_ref = db.collection('agent_sessions').document(session_id)
            events_ref = session_ref.collection('events')
            for event in new_events:
                if 'timestamp' not in event: event['timestamp'] = datetime.datetime.now(datetime.timezone.utc)
                _firestore_call_with_timeout(lambda: events_ref.add(event))

            _firestore_call_with_timeout(lambda: session_ref.update({
                "status": "completed",
                "type": "work_answer",
                "last_updated": expire_time
            }))
            
            safe_n8n_delivery({
                "session_id": session_id, 
                "type": global_operation_type, 
                "message": answer_text, 
                "query": sanitized_topic,
                "intent": "BLOG_OUTLINE",
                "output_target": target,
                "channel_id": slack_context.get('channel') or req.get('channel') or req.get('slack_channel') or req.get('event', {}).get('channel'), 
                "thread_ts": slack_context.get('thread_ts') or slack_context.get('ts'),
                "is_initial_post": is_initial_post
            })
            return jsonify({"msg": "Blog outline sent"}), 200

        # === PATH C: SALES TRANSCRIPT (Fast Exit) ===
        elif intent == "SALES_TRANSCRIPT" or req.get('type') == 'sales_transcript':
            print("Executing Sales Transcript Processing (Fast Exit)")
            transcript_text = req.get('transcript', sanitized_topic)
            target = get_output_target(intent)
            result = process_sales_transcript(transcript_text, output_target=target)
            
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
                _firestore_call_with_timeout(lambda: events_ref.add(event))

            # Update session status
            _firestore_call_with_timeout(lambda: session_ref.update({
                "status": "awaiting_feedback",
                "type": "solution_brief",
                "last_updated": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=30)
            }))

            # Send to N8N for distribution
            safe_n8n_delivery({
                "session_id": session_id,
                "type": global_operation_type,
                "query": sanitized_topic,
                "objections": result['objections'],
                "brief": result['brief'],
                "output_target": target,
                "channel_id": slack_context.get('channel') or req.get('channel') or req.get('slack_channel') or req.get('event', {}).get('channel'),
                "thread_ts": slack_context.get('thread_ts') or slack_context.get('ts'),
                "is_initial_post": is_initial_post
            })
            
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
            
            # --- STRUCTURAL SAFETY AUDIT ---
            # Try to parse the metadata to isolate the AI's "rationale" (which should be safe)
            try:
                meta_obj = json.loads(intent_metadata)
                intent_val = str(meta_obj.get("intent", "")).lower()
                rationale_val = str(meta_obj.get("rationale", "")).lower()
                
                # 1. Check for explicit signals in the intent field
                if "signal_block" in intent_val or "violation" in intent_val:
                    is_blocked = True
                
                # 2. Check for keywords in the structure EXCEPT for the rationale
                # We build a 'vetting_string' that includes everything but the AI's thinking
                vetting_parts = [intent_val]
                for k, v in meta_obj.items():
                    if k != "rationale" and k != "intent":
                        vetting_parts.append(str(v).lower())
                
                vetting_string = " ".join(vetting_parts)
                refusal_keywords = [r"\bharm\b", r"\brefuse\b", r"\billegal\b", r"\bviolence\b", r"\bsensitive\b", r"\bprohibited\b", r"\bviolate\b"]
                if any(re.search(kw, vetting_string) for kw in refusal_keywords):
                    is_blocked = True
                    
            except Exception:
                # Fallback to legacy string-match if JSON is malformed (raw refusal)
                if '"intent": "signal_block"' in meta_lower or '"intent": "violation"' in meta_lower:
                    is_blocked = True
                
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
                _firestore_call_with_timeout(lambda: events_ref.add(event))

            _firestore_call_with_timeout(lambda: session_ref.update({
                "status": "blocked",
                "type": "safety_violation",
                "detected_geo": final_geo,
                "intent": final_intent_key,
                "last_updated": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=30)
            }))
            target = get_output_target(intent)
            safe_n8n_delivery({
                "session_id": session_id, 
                "type": global_operation_type, 
                "message": reply_text, 
                "query": sanitized_topic,
                "output_target": target,
                "channel_id": slack_context.get('channel') or req.get('channel') or req.get('slack_channel') or req.get('event', {}).get('channel'), 
                "thread_ts": slack_context.get('thread_ts') or slack_context.get('ts')
            })
            return jsonify({"msg": "Safety block applied"}), 200

        # --- MONITORING: Signal Propagation ---
        if "tool_logs" in research_data: new_events.extend(research_data["tool_logs"])
        print(f"TELEMETRY: Research Phase concluded with {len(research_data.get('context', []))} context snippets.")

        # CHANGE: Ensure downstream functions use the raw topic too, so they see the full request
        clean_topic = sanitized_topic

        # 3. Generate Output based on Intent
        # UPGRADE: Handle structural intents (FORMAT_*) and provide a robust fallback
        if intent in ["DIRECT_ANSWER", "BLOG_OUTLINE"] or str(intent).startswith("FORMAT_"):
            if intent == "BLOG_OUTLINE": print(f"Executing Blog Outline generation for command: {clean_topic[:50]}")
            
            # Determined Output Target
            target = get_output_target(intent)
            
            # HIGH-FIDELITY ROUTING: Use Comprehensive Content Strategist for Audits/Formats
            answer_data = generate_comprehensive_answer(clean_topic, research_data['context'], history=history_text, context_topic=original_topic, session_id=session_id, output_target=target, intent_label=intent)
            answer_text = answer_data['text']
            research_intent = answer_data.get('intent', intent)
            
            new_events.append({"event_type": "agent_answer", "text": answer_text, "intent": research_intent})
            
            # Writing to a Sub-collection
            session_ref = db.collection('agent_sessions').document(session_id)
            events_ref = session_ref.collection('events')

            for event in new_events:
                if 'timestamp' not in event: event['timestamp'] = datetime.datetime.now(datetime.timezone.utc)
                _firestore_call_with_timeout(lambda: events_ref.add(event))

            update_data = {
                "status": "awaiting_feedback", 
                "type": "work_answer", 
                "slack_context": slack_context,
                "detected_geo": final_geo,
                "intent": research_intent,
                "last_updated": expire_time
            }
            if is_initial_post: update_data["topic"] = clean_topic
            _firestore_call_with_timeout(lambda: session_ref.update(update_data))

            safe_n8n_delivery({
                "session_id": session_id, 
                "type": global_operation_type, 
                "message": convert_html_to_markdown(answer_text), 
                "query": sanitized_topic,
                "intent": research_intent,
                "output_target": target,
                "channel_id": slack_context.get('channel') or req.get('channel') or req.get('slack_channel') or req.get('event', {}).get('channel'), 
                "thread_ts": slack_context.get('thread_ts') or slack_context.get('ts'),
                "is_initial_post": is_initial_post
            })
            return jsonify({"msg": "High-fidelity answer sent"}), 200

        elif intent == "SIMPLE_QUESTION":
            # Determined Output Target
            target = get_output_target(intent)
            # LIGHTWEIGHT ROUTING: Use Natural Answer Engine
            answer_data = generate_natural_answer(clean_topic, research_data['context'], history=history_text, session_id=session_id, output_target=target, intent_label=intent)
            answer_text = answer_data['text']
            research_intent = answer_data.get('intent', intent)
            formatting_directive = answer_data.get('directive', "")
            
            new_events.append({"event_type": "agent_answer", "text": answer_text, "intent": research_intent})
            
            session_ref = db.collection('agent_sessions').document(session_id)
            events_ref = session_ref.collection('events')
            for event in new_events:
                if 'timestamp' not in event: event['timestamp'] = datetime.datetime.now(datetime.timezone.utc)
                _firestore_call_with_timeout(lambda: events_ref.add(event))

            update_data = {
                "status": "awaiting_feedback", 
                "type": "work_answer", 
                "slack_context": slack_context,
                "detected_geo": final_geo,
                "intent": research_intent,
                "last_updated": expire_time
            }
            if is_initial_post: update_data["topic"] = clean_topic
            _firestore_call_with_timeout(lambda: session_ref.update(update_data))

            safe_n8n_delivery({
                "session_id": session_id, 
                "type": global_operation_type, 
                "message": convert_html_to_markdown(answer_text), 
                "query": sanitized_topic,
                "intent": research_intent,
                "directive": formatting_directive,
                "output_target": target,
                "channel_id": slack_context.get('channel') or req.get('channel') or req.get('slack_channel') or req.get('event', {}).get('channel'), 
                "thread_ts": slack_context.get('thread_ts') or slack_context.get('ts'),
                "is_initial_post": is_initial_post
            })
            return jsonify({"msg": "Natural answer sent"}), 200

        elif intent == "DEEP_DIVE":
            target_count = extract_target_word_count(sanitized_topic, history=history_text)
            print(f"TELEMETRY: Executing Dynamic Deep-Dive Recursive Expansion (Target: {target_count})...")
            # Determined Output Target
            target = get_output_target(intent)
            # PASS THE sanitzied_topic as the DIRECTIVE to prioritize feedback
            article_html = generate_deep_dive_article(sanitized_topic, research_data['context'], history=history_text, history_events=history_events, target_length=target_count, target_geo=final_geo, session_id=session_id, output_target=target)
            
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
                _firestore_call_with_timeout(lambda: events_ref.add(event))

            update_data = {
                "status": "awaiting_feedback", 
                "type": "work_answer", 
                "slack_context": slack_context,
                "detected_geo": final_geo,
                "intent": "DEEP_DIVE",
                "last_updated": expire_time
            }
            _firestore_call_with_timeout(lambda: session_ref.update(update_data))

            safe_n8n_delivery({
                "session_id": session_id, 
                "type": global_operation_type, 
                "message": convert_html_to_markdown(article_html), 
                "query": sanitized_topic,
                "output_target": target,
                "channel_id": slack_context.get('channel') or req.get('channel') or req.get('slack_channel') or req.get('event', {}).get('channel'), 
                "thread_ts": slack_context.get('thread_ts') or slack_context.get('ts'),
                "is_initial_post": is_initial_post
            })
            return jsonify({"msg": "Deep-Dive article sent"}), 200

        elif intent == "TOPIC_CLUSTER_PROPOSAL":
            try:
                # FIX: Pass original_topic (Mission) for tool consistency
                cluster_data = generate_topic_cluster(sanitized_topic, research_data['context'], history=history_text, is_initial=is_initial_post, session_id=session_id)
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
                    _firestore_call_with_timeout(lambda: events_ref.add(event))

                update_data = {
                    "status": "awaiting_feedback", 
                    "type": "work_proposal", 
                    "slack_context": slack_context,
                    "detected_geo": final_geo,
                    "intent": final_intent_key,
                    "last_updated": expire_time
                }
                if is_initial_post: update_data["topic"] = clean_topic
                _firestore_call_with_timeout(lambda: session_ref.update(update_data))
                
                # Align with N8N Parser: Wrap JSON in markdown for regex extraction
                cluster_msg = f"```json\n{json.dumps(cluster_data, indent=2)}\n```"
                safe_n8n_delivery({
                    "session_id": session_id, 
                    "type": global_operation_type, 
                    "message": cluster_msg,
                    "intent": "TOPIC_CLUSTER_PROPOSAL",
                    "channel_id": slack_context.get('channel'), 
                    "thread_ts": slack_context.get('thread_ts') or slack_context.get('ts'),
                    "is_initial_post": is_initial_post
                })
                return jsonify({"msg": "Topic cluster sent"}), 200

            except Exception as e:
                print(f"TELEMETRY: ⚠️ TOPIC_CLUSTER Fallback: {e}")
                target = get_output_target(intent)
                answer_data = generate_comprehensive_answer(clean_topic, research_data['context'], history=history_text, context_topic=original_topic, output_target=target, intent_label=intent)
                answer_text = answer_data['text']
                research_intent = answer_data.get('intent', intent)
                
                new_events.append({"event_type": "agent_answer", "text": answer_text, "intent": research_intent})
                
                session_ref = db.collection('agent_sessions').document(session_id)
                events_ref = session_ref.collection('events')
                for event in new_events:
                    if 'timestamp' not in event: event['timestamp'] = datetime.datetime.now(datetime.timezone.utc)
                    _firestore_call_with_timeout(lambda: events_ref.add(event))

                update_data = {
                    "status": "awaiting_feedback", 
                    "type": "work_answer", 
                    "slack_context": slack_context,
                    "detected_geo": final_geo,
                    "intent": final_intent_key,
                    "last_updated": expire_time
                }
                if is_initial_post: update_data["topic"] = clean_topic
                _firestore_call_with_timeout(lambda: session_ref.update(update_data))
                safe_n8n_delivery({"session_id": session_id, "type": global_operation_type, "message": answer_text, "intent": research_intent, "channel_id": slack_context['channel'], "thread_ts": slack_context['ts'], "is_initial_post": is_initial_post})
                return jsonify({"msg": "Cluster Fallback Answer sent"}), 200
        elif intent == "THEN_VS_NOW_PROPOSAL":
            try:
                # FIX: Pass original_topic (Mission) for tool consistency
                current_proposal = create_euphemistic_links({**research_data, "clean_topic": original_topic}, is_initial=is_initial_post, session_id=session_id)
                if not current_proposal or "interlinked_concepts" not in current_proposal:
                     raise ValueError("Failed to generate Then-vs-Now proposal.")
                
                approval_id = f"approval_{uuid.uuid4().hex[:8]}"
                new_events.append({"event_type": "loop_draft", "iteration": 0, "proposal_data": current_proposal})

                loop_count = 0
                while loop_count < MAX_LOOP_ITERATIONS:
                    critique = critique_proposal(original_topic, current_proposal)
                    if "APPROVED" in critique.upper(): break
                    try: 
                         refined = refine_proposal(original_topic, current_proposal, critique, session_id=session_id)
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
                    _firestore_call_with_timeout(lambda: events_ref.add(event))

                update_data = {
                    "status": "awaiting_feedback", 
                    "type": "work_proposal", 
                    "slack_context": slack_context,
                    "detected_geo": final_geo,
                    "intent": final_intent_key,
                    "last_updated": expire_time
                }
                if is_initial_post: update_data["topic"] = clean_topic
                _firestore_call_with_timeout(lambda: session_ref.update(update_data))

                safe_n8n_delivery({
                    "session_id": session_id, 
                    "type": "answer_proposal", 
                    "proposal": current_proposal['interlinked_concepts'], 
                    "approval_id": approval_id, 
                    "channel_id": slack_context.get('channel'), 
                    "thread_ts": slack_context.get('thread_ts') or slack_context.get('ts'),
                    "is_initial_post": is_initial_post
                })
                return jsonify({"msg": "Then-vs-Now proposal sent"}), 200

            except ValueError as e:
                # Fallback
                target = get_output_target(intent)
                answer_data = generate_comprehensive_answer(clean_topic, research_data['context'], history=history_text, context_topic=original_topic, output_target=target, intent_label=intent)
                answer_text = answer_data['text']
                research_intent = answer_data.get('intent', intent)
                
                new_events.append({"event_type": "agent_answer", "text": answer_text, "intent": research_intent})
                
                session_ref = db.collection('agent_sessions').document(session_id)
                events_ref = session_ref.collection('events')

                for event in new_events:
                    if 'timestamp' not in event: event['timestamp'] = datetime.datetime.now(datetime.timezone.utc)
                    _firestore_call_with_timeout(lambda: events_ref.add(event))

                update_data = {
                    "status": "awaiting_feedback", 
                    "type": "work_answer", 
                    "slack_context": slack_context,
                    "detected_geo": final_geo,
                    "intent": final_intent_key,
                    "last_updated": expire_time
                }
                if is_initial_post: update_data["topic"] = clean_topic
                _firestore_call_with_timeout(lambda: session_ref.update(update_data))
                safe_n8n_delivery({"session_id": session_id, "type": global_operation_type, "message": answer_text, "intent": research_intent, "channel_id": slack_context['channel'], "thread_ts": slack_context['ts'], "is_initial_post": is_initial_post})
                return jsonify({"msg": "Fallback answer sent"}), 200

        # === PATH C: PSEO ARTICLE GENERATION (Dual-Agent Path) ===
        elif intent in ["PSEO_ARTICLE", "PSEO_PAGE"]:
            try:
                # 1. AGENT A (Gemini): Generate the Body Content (Prioritizing Latest Feedback)
                target = get_output_target(intent)
                
                if intent == "PSEO_PAGE":
                     article_html = generate_pseo_page(sanitized_topic, research_data['context'], history=history_text, history_events=history_events, is_initial_post=is_initial_post, output_target=target)
                else:
                     article_html = generate_pseo_article(sanitized_topic, research_data['context'], history=history_text, history_events=history_events, is_initial_post=is_initial_post, output_target=target)
                
                # 2. AGENT B (Claude): Generate the Semantic Metadata
                seo_data = generate_seo_metadata(article_html, original_topic, session_id=session_id, intent=intent)
                
                # 3. Get session reference (used throughout this handler)
                session_ref = db.collection('agent_sessions').document(session_id)
                
                # 4. Check for existing Ghost post ID (Prevent Duplicates)
                session_data = session_ref.get().to_dict()
                ghost_post_id = session_data.get('ghost_post_id') if session_data else None
                
                # 5. Determine operation type (ADK Interoperability: Support Pages of Posts)
                is_page_target = (intent == "PSEO_PAGE") or any(kw in original_topic.lower() or kw in sanitized_topic.lower() for kw in ["pseo page", "collection page", "page template", "ghost page", "page slug"])
                
                # 6. Build Payload (Schema-Aware for Ghost or Next.js)
                delivery_payload = {
                    "title": seo_data.get("title", "Untitled Draft"),
                    "html": article_html,
                    "tags": seo_data.get("tags", []),
                    "meta_title": seo_data.get("meta_title"),
                    "meta_description": seo_data.get("meta_description"),
                    "custom_excerpt": seo_data.get("custom_excerpt")
                }
                
                # Page-Specific Metadata
                if is_page_target:
                    # Extract slug from topic (e.g., "Nigeria Consumption Insights" -> "nigeria")
                    slug = re.sub(r'[^a-zA-Z0-9-]', '', original_topic.lower().replace(' ', '-'))
                    
                    # ADK FIX: Support Manual Slug Override via Prompt
                    request_text = (feedback_text or original_topic).lower()
                    slug_match = re.search(r'(?:slug|link)\s+(?:to|be|is|=)\s*["\']?([a-zA-Z0-9-]+)["\']?', request_text)
                    if slug_match:
                        manual_slug = slug_match.group(1).strip()
                        print(f"🎯 MANUAL SLUG OVERRIDE: '{manual_slug}'")
                        slug = manual_slug
                        
                    delivery_payload["slug"] = slug
                    # DYNAMIC EXTRACTION: Use the helper to get real data
                    print(f"🔍 Extracting Market Data for slug: {slug}...")
                    market_data = extract_market_data(research_data.get('context', ''), history_text, slug)

                    delivery_payload["data_points"] = market_data
                    delivery_payload["script_id"] = "market-insight-data"  # N8N generic node uses this for <script id=...>
                
                # Set status to 'draft' ONLY for CREATE operations
                # For UPDATE operations, omit status to preserve current status (draft or published)
                if not ghost_post_id:
                    delivery_payload["status"] = "draft"
                
                # Add ghost_post_id for updates
                if ghost_post_id:
                    delivery_payload["ghost_post_id"] = ghost_post_id
                
                # Remove None values to avoid N8N errors
                delivery_payload = {k: v for k, v in delivery_payload.items() if v is not None}
                
                # 7. SUCCESS: Log the event to subcollection
                new_events.append({
                    "event_type": "agent_proposal", 
                    "proposal_type": "pseo_article", 
                    "text": convert_html_to_markdown(article_html),
                    "proposal_data": {"article_html": article_html}
                })
                
                # 8. Write events to Database
                events_ref = session_ref.collection('events')
                for event in new_events:
                    if 'timestamp' not in event: event['timestamp'] = datetime.datetime.now(datetime.timezone.utc)
                    _firestore_call_with_timeout(lambda: events_ref.add(event))

                update_data = {
                    "status": "awaiting_feedback", 
                    "type": "work_article", 
                    "slack_context": slack_context,
                    "detected_geo": final_geo,
                    "intent": intent,
                    "last_updated": expire_time
                }
                if is_initial_post: update_data["topic"] = clean_topic
                _firestore_call_with_timeout(lambda: session_ref.update(update_data))
                
                # 9. Send STRUCTURED DATA to N8N (Restored Wrapper for GhostNode Harmony)
                safe_n8n_delivery({
                    "session_id": session_id, 
                    "type": global_operation_type,
                    "payload": delivery_payload,
                    "query": sanitized_topic,
                    "output_target": target,
                    "channel_id": slack_context.get('channel') or req.get('channel') or req.get('slack_channel') or req.get('event', {}).get('channel'), 
                    "thread_ts": slack_context.get('thread_ts') or slack_context.get('ts'),
                    "is_initial_post": is_initial_post
                })
                
                # 10. FUTURE: Generate and send featured image to Slack
                # featured_image_prompt = seo_data.get("featured_image_prompt")
                # if featured_image_prompt:
                #     image_base64 = generate_featured_image(featured_image_prompt)
                #     send_slack_image(image_base64, slack_context)

                return jsonify({"msg": "pSEO Draft sent"}), 200

            except Exception as e:
                print(f"⚠️ PSEO_ARTICLE Fallback: {e}")
                target = get_output_target(intent)
                answer_data = generate_comprehensive_answer(clean_topic, research_data['context'], history=history_text, context_topic=original_topic, output_target=target, intent_label=intent)
                answer_text = answer_data['text']
                research_intent = answer_data.get('intent', intent)
                
                new_events.append({"event_type": "agent_answer", "text": answer_text, "intent": research_intent})
                
                session_ref = db.collection('agent_sessions').document(session_id)
                events_ref = session_ref.collection('events')
                for event in new_events:
                    if 'timestamp' not in event: event['timestamp'] = datetime.datetime.now(datetime.timezone.utc)
                    _firestore_call_with_timeout(lambda: events_ref.add(event))

                update_data = {
                    "status": "awaiting_feedback", 
                    "type": "work_answer", 
                    "slack_context": slack_context,
                    "detected_geo": final_geo,
                    "intent": research_intent,
                    "last_updated": expire_time
                }
                if is_initial_post: update_data["topic"] = clean_topic
                _firestore_call_with_timeout(lambda: session_ref.update(update_data))
                if intent == "PSEO_ARTICLE":
                    safe_n8n_delivery({
                        "session_id": session_id, 
                        "type": global_operation_type, 
                        "query": sanitized_topic,
                        "payload": {
                            "title": clean_topic,
                            "html": answer_text,
                            "status": "draft_fallback"
                        },
                        "output_target": target,
                        "channel_id": slack_context.get('channel') or req.get('channel') or req.get('slack_channel') or req.get('event', {}).get('channel'), 
                        "thread_ts": slack_context.get('thread_ts') or slack_context.get('ts'),
                        "is_initial_post": is_initial_post
                    })
                else:
                    safe_n8n_delivery({
                        "session_id": session_id, 
                        "type": global_operation_type, 
                        "message": answer_text, 
                        "query": sanitized_topic,
                        "intent": research_intent, 
                        "output_target": target,
                        "channel_id": slack_context.get('channel') or req.get('channel') or req.get('slack_channel') or req.get('event', {}).get('channel'), 
                        "thread_ts": slack_context.get('thread_ts') or slack_context.get('ts'), 
                        "is_initial_post": is_initial_post
                    })
                return jsonify({"msg": "pSEO Fallback Answer sent"}), 200

        # === PATH C2: PSEO_LP GENERATION (Long-Form Landing Page) ===
        elif intent == "PSEO_LP":
            try:
                target = get_output_target(intent)
                article_html = generate_lp_page(
                    sanitized_topic, research_data['context'],
                    history=history_text, history_events=history_events,
                    is_initial_post=is_initial_post, output_target=target
                )

                seo_data = generate_seo_metadata(article_html, original_topic, session_id=session_id, intent=intent)

                session_ref = db.collection('agent_sessions').document(session_id)

                # 4. Check for existing Ghost post ID (Prevent Duplicates)
                session_data = session_ref.get().to_dict()
                ghost_post_id = session_data.get('ghost_post_id') if session_data else None

                # Slug generation: Check for explicit override, then fall back to generated
                request_text = (feedback_text or original_topic).lower()
                slug = seo_data.get("slug")
                slug_match = re.search(r'(?:slug|link)\s+(?:to|be|is|=)\s*["\']?([a-zA-Z0-9-]+)["\']?', request_text)
                if slug_match:
                    slug = slug_match.group(1).strip()
                    print(f"🎯 LP MANUAL SLUG OVERRIDE: '{slug}'")
                
                if not slug:
                    fallback_text = seo_data.get("title") or sanitized_topic
                    slug = re.sub(r'[^a-zA-Z0-9-]', '', fallback_text.lower().replace(' ', '-'))
                # Prevent runaway slugs
                slug = slug[:60].strip('-')

                print(f"🔍 Extracting LP Data for slug: {slug}...")
                lp_data = extract_lp_data(research_data.get('context', ''), history_text, slug, original_topic)

                lp_tags = seo_data.get("tags", [])
                if lp_data and lp_data.get("cluster_tag"):
                    cluster_tag = f"#{lp_data.get('cluster_tag').lstrip('#')}"
                    lp_tags = [cluster_tag]

                delivery_payload = {
                    "title": seo_data.get("title", "Untitled LP"),
                    "html": article_html,
                    "slug": slug,
                    "tags": lp_tags,
                    "meta_title": seo_data.get("meta_title"),
                    "meta_description": seo_data.get("meta_description"),
                    "custom_excerpt": seo_data.get("custom_excerpt"),
                    "data_points": lp_data,
                    "script_id": "lp-data",   # N8N generic node uses this for <script id="lp-data">
                }

                if not ghost_post_id:
                    delivery_payload["status"] = "draft"
                if ghost_post_id:
                    delivery_payload["ghost_post_id"] = ghost_post_id

                delivery_payload = {k: v for k, v in delivery_payload.items() if v is not None}

                new_events.append({
                    "event_type": "agent_proposal",
                    "proposal_type": "pseo_lp",
                    "text": convert_html_to_markdown(article_html),
                    "proposal_data": {"article_html": article_html}
                })

                events_ref = session_ref.collection('events')
                for event in new_events:
                    if 'timestamp' not in event: event['timestamp'] = datetime.datetime.now(datetime.timezone.utc)
                    _firestore_call_with_timeout(lambda: events_ref.add(event))

                update_data = {
                    "status": "awaiting_feedback",
                    "type": "work_article",
                    "slack_context": slack_context,
                    "detected_geo": final_geo,
                    "intent": intent,
                    "last_updated": expire_time
                }
                if is_initial_post: update_data["topic"] = clean_topic
                _firestore_call_with_timeout(lambda: session_ref.update(update_data))

                safe_n8n_delivery({
                    "session_id": session_id,
                    "type": global_operation_type,
                    "payload": delivery_payload,
                    "query": sanitized_topic,
                    "output_target": target,
                    "channel_id": slack_context.get('channel') or req.get('channel') or req.get('slack_channel') or req.get('event', {}).get('channel'),
                    "thread_ts": slack_context.get('thread_ts') or slack_context.get('ts'),
                    "is_initial_post": is_initial_post
                })
                return jsonify({"msg": "LP Draft sent"}), 200

            except Exception as e:
                print(f"⚠️ PSEO_LP Fallback: {e}")
                target = get_output_target(intent)
                answer_data = generate_comprehensive_answer(clean_topic, research_data['context'], history=history_text, context_topic=original_topic, output_target=target, intent_label=intent)
                answer_text = answer_data['text']
                research_intent = answer_data.get('intent', intent)
                new_events.append({"event_type": "agent_answer", "text": answer_text, "intent": research_intent})
                session_ref = db.collection('agent_sessions').document(session_id)
                events_ref = session_ref.collection('events')
                for event in new_events:
                    if 'timestamp' not in event: event['timestamp'] = datetime.datetime.now(datetime.timezone.utc)
                    _firestore_call_with_timeout(lambda: events_ref.add(event))
                update_data = {"status": "awaiting_feedback", "type": "work_answer", "slack_context": slack_context, "detected_geo": final_geo, "intent": research_intent, "last_updated": expire_time}
                if is_initial_post: update_data["topic"] = clean_topic
                _firestore_call_with_timeout(lambda: session_ref.update(update_data))
                safe_n8n_delivery({"session_id": session_id, "type": global_operation_type, "message": answer_text, "intent": research_intent, "output_target": target, "channel_id": slack_context.get('channel') or req.get('channel') or req.get('slack_channel'), "thread_ts": slack_context.get('thread_ts') or slack_context.get('ts'), "is_initial_post": is_initial_post})
                return jsonify({"msg": "LP Fallback Answer sent"}), 200

        else: 
            # ROBUST FALLBACK: If intent is unknown, default to a natural answer rather than crashing with 500
            print(f"TELEMETRY: ⚠️ Unknown Intent Detected: '{intent}'. Defaulting to Natural Answer fallback.")
            target = get_output_target("FALLBACK") # Defaults to MODERATOR_VIEW
            answer_data = generate_natural_answer(clean_topic, research_data['context'], history=history_text, output_target=target, intent_label=intent)
            answer_text = answer_data['text']
            research_intent = answer_data['intent']
            formatting_directive = answer_data['directive']
            
            new_events.append({"event_type": "agent_answer", "text": answer_text, "intent": research_intent, "status": "intent_fallback"})
            
            session_ref = db.collection('agent_sessions').document(session_id)
            events_ref = session_ref.collection('events')
            for event in new_events:
                if 'timestamp' not in event: event['timestamp'] = datetime.datetime.now(datetime.timezone.utc)
                _firestore_call_with_timeout(lambda: events_ref.add(event))

            update_data = {
                "status": "awaiting_feedback", 
                "type": "work_answer", 
                "slack_context": slack_context,
                "detected_geo": final_geo,
                "intent": research_intent,
                "last_updated": expire_time
            }
            if is_initial_post: update_data["topic"] = clean_topic
            _firestore_call_with_timeout(lambda: session_ref.update(update_data))

            safe_n8n_delivery({
                "session_id": session_id, 
                "type": global_operation_type, 
                "message": answer_text, 
                "intent": research_intent,
                "directive": formatting_directive,
                "query": sanitized_topic,
                "output_target": target,
                "channel_id": slack_context.get('channel') or slack_context.get('channel_id') or req.get('channel') or req.get('slack_channel') or req.get('event', {}).get('channel'), 
                "thread_ts": slack_context.get('thread_ts') or slack_context.get('ts'),
                "is_initial_post": is_initial_post
            })
            
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
    # COMPATIBILITY: Allow either Knowledge Key or original Ingestion Key (for legacy n8n flows)
    if provided_key != KNOWLEDGE_INGESTION_API_KEY and provided_key != INGESTION_API_KEY:
        print(f"⚠️ Unauthorized knowledge ingestion attempt from IP: {request.remote_addr}")
        return jsonify({"error": "Invalid API key"}), 403

    global db, unimodel
    if db is None: db = firestore.Client(project=PROJECT_ID)
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")

    req = request.get_json(silent=True) or {}
    session_id = req.get('session_id', 'unknown')
    topic = req.get('topic', 'unknown')
    
    # COMPATIBILITY: Support legacy 'text' or 'content' fields alongside new 'story' field
    final_story = req.get('story') or req.get('text') or req.get('content', '')
    
    # Unified Ingestion Router (Determines destination collection)
    ingest_type = req.get('type', 'knowledge') # 'knowledge' or 'solution_brief'
    
    if ingest_type == 'solution_brief':
        # ROUTING: Solution Briefs -> 'solution_briefs' collection
        # NOTE: This path BYPASSES the vector embedding pipeline. 
        # Briefs are structured session memories, not RAG knowledge chunks.
        db.collection('solution_briefs').add({
            'session_id': session_id,
            'topic': topic,
            'objections': req.get('objections'),
            'brief': req.get('story'),
            'created_at': datetime.datetime.now(datetime.timezone.utc)
        })
        return jsonify({"msg": "Solution brief ingested."}), 200

    elif ingest_type in ['knowledge', 'grounding_data']:
        # Knowledge Base Path: Chunk and Embed
        # SUPPORT: 'grounding_data' (Raw CSV) and 'knowledge' (Approved Synthesis)
        chunks = chunk_text(final_story)
        embeddings = embedding_model.get_embeddings([c for c in chunks])
        
        batch = db.batch()
        count = 0
        for i, (text_segment, embedding_obj) in enumerate(zip(chunks, embeddings)):
            doc_ref = db.collection('knowledge_base').document(f"{session_id}_{i}")
            batch.set(doc_ref, {
                "content": text_segment, # BYPASS_PII: scrub_pii(text_segment) to prevent [PHONE_MASKED] corruption on years/prices
                "embedding": Vector(embedding_obj.values),
                "topic_trigger": topic, # BYPASS_PII: scrub_pii(topic) 
                "source_session": session_id,
                "knowledge_type": ingest_type, # NEW: Explicitly track type (grounding_data vs knowledge)
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

    else:
        print(f"⚠️ Unknown ingestion type: {ingest_type}")
        return jsonify({"error": f"Unknown type: {ingest_type}"}), 400



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
        print(f"⚠️ Unauthorized compliance ingestion attempt from IP: {request.remote_addr}")
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
            "content": chunk,
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
