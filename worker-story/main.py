# --- /worker-story/main.py ---
import functions_framework
from flask import jsonify
import vertexai
from vertexai.generative_models import GenerativeModel
from litellm import completion
from vertexai.language_models import TextEmbeddingModel
import os
import json
import re
import uuid
import certifi
import requests
from bs4 import BeautifulSoup
from google.cloud import secretmanager, firestore
from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from googleapiclient.discovery import build
from serpapi import GoogleSearch
import datetime

# --- Configuration ---
PROJECT_ID = os.environ.get("PROJECT_ID")
LOCATION = os.environ.get("LOCATION")
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-2.5-pro") 
MODEL_PROVIDER = os.environ.get("MODEL_PROVIDER", "vertex_ai")
FLASH_MODEL_NAME = os.environ.get("FLASH_MODEL_NAME", "gemini-2.5-flash-lite")
SEARCH_ENGINE_ID = os.environ.get("SEARCH_ENGINE_ID")
BROWSERLESS_API_KEY = os.environ.get("BROWSERLESS_API_KEY")
SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
N8N_PROPOSAL_WEBHOOK_URL = os.environ.get("N8N_PROPOSAL_WEBHOOK_URL") 
MAX_LOOP_ITERATIONS = 2

# --- Global Clients ---
model = None
flash_model = None
search_api_key = None
db = None

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
            self._native_model = GenerativeModel(model_name)
            print(f"✅ Loaded Native Vertex Model: {model_name}", flush=True)

    def generate_content(self, prompt, generation_config=None):
        """
        Universal generation function.
        """
        # PATH A: Native Vertex AI
        if self.provider == "vertex_ai":
            return self._native_model.generate_content(prompt, generation_config=generation_config)

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
                    timeout=30
                )
                
                # Create a Fake Vertex Response Object
                class MockResponse:
                    def __init__(self, content):
                        self.text = content
                
                return MockResponse(response.choices[0].message.content)
                
            except Exception as e:
                print(f"❌ LiteLLM Error: {e}", flush=True)
                raise e

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
        * *Bad:* OpenAI OR Gemini news
        * *Good:* (OpenAI OR Gemini) news
    2.  **Alternatives:** Use uppercase `OR` between entities
        * *User Intent:* Find a recipe for a dessert using either strawberries or raspberries
        * *Bad Query:* strawberry raspberry dessert recipe (Implies you want both berries).
        * *Good Query:* (strawberry OR raspberry) dessert recipe
    3.  **Exact Phrases:** Use quotes `""` for specific multi-word concepts.
        * *User Intent:* Find the origin of the specific phrase "the early bird catches the worm".
        * *Bad Query:* early bird catches worm origin (Finds pages with these words scattered anywhere).
        **Good Query:* "the early bird catches the worm" origin
    4.  **Exclusion:** Use `-` to remove noise. Example: `jobs -entry_level`
    5.  **No Commas:** Do not use commas to separate terms; they are ignored
    6.  **Freshness:** If the user asks for news, include terms like "news", "updates", "latest"
    7.  **Remove Fluff and stop words:** Remove "I want to know about", "Please tell me", etc
    8.  **Searching Within a Specified Site:** Use the site: operator along with precise keywords. e.g. "site:nytimes.com climate change impacts"
    9.  **Searching a Specified File Type:** Use the filetype: operator to find specific document types. e.g. "artificial intelligence filetype:pdf"
    10. **Combining Operators:** Combine multiple operators for refined searches. e.g. "(AI OR artificial intelligence) site:edu filetype:pdf -outdated"
    11. **Avoid Ambiguity:** Ensure the query is specific enough to avoid broad or vague results
    12.  **Limit Length:** Keep the query concise, ideally under 10 words, while retaining precision
    13.  **Use AND Implicitly:** Remember that Google uses AND by default between terms
    14.  **Prioritize Relevance:** Focus on the most relevant keywords that capture the essence of the user's request
    15.  **Avoid Redundancy:** Do not repeat terms or concepts unnecessarily

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

def extract_url_from_text(text):
    url_pattern = r'(https?://[^\s<>|]+)'
    match = re.search(url_pattern, text)
    if match: return match.group(1)
    return None

def chunk_text(text, chunk_size=1500, overlap=300):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start = end - overlap 
    return [c for c in chunks if c]

# --- Tools ---
#1. The Specialist Web Scraper Tool
def fetch_article_content(url):
    print(f"Tool: Reading URL via Browserless: {url}")
    if not BROWSERLESS_API_KEY: return "Error: Browserless API Key is missing."
    endpoint = f"https://production-sfo.browserless.io/content?token={BROWSERLESS_API_KEY}&stealth=true"
    payload = {"url": url, "rejectResourceTypes": ["image", "media", "font"], "gotoOptions": {"timeout": 15000, "waitUntil": "networkidle2"}}
    headers = {"Cache-Control": "no-cache", "Content-Type": "application/json"}
    try:
        response = requests.post(endpoint, json=payload, headers=headers, timeout=20)
        if response.status_code != 200: return f"Failed to read content. (Status: {response.status_code})"
        soup = BeautifulSoup(response.content, 'html.parser')
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "iframe", "ad"]): tag.decompose()
        text = soup.get_text(separator='\n\n')
        clean_text = re.sub(r'\n\s*\n', '\n\n', text).strip()
        if len(clean_text) < 200: return "Content appears to be behind a login wall or empty."
        return clean_text[:20000]
    except Exception as e:
        print(f"Scraper Exception: {e}")
        return "Error: Could not read page."
    
#2. The Internal Knowledge Retrieval Tool
def search_long_term_memory(query):
    global db
    if db is None: db = firestore.Client()
    
    print(f"Tool: Accessing Hippocampus (Vector Search) for: '{query}'")
    
    # 1. Embed
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
    # FIX: Enforce Token Limit (Approx 1500 words / 6000 chars is safe)
    safe_query = query[:6000] 
    query_embedding = embedding_model.get_embeddings([safe_query])[0].values
    
    # 2. Search
    collection = db.collection('knowledge_base')
    
    # Prepare the query (This is the instruction)
    vector_query = collection.find_nearest(
        vector_field="embedding",
        query_vector=Vector(query_embedding),
        distance_measure=DistanceMeasure.COSINE, # <--- Use Enum, not string
        limit=3
    )
    
    # --- THE FIX: EXECUTE THE QUERY ---
    try:
        results = vector_query.get() # <--- This pulls the trigger
    except Exception as e:
        print(f"Vector Search failed (Index might be building or collection empty): {e}")
        return []

    # 3. Format
    memories = []
    for res in results:
        data = res.to_dict()
        # Safety check in case data is partial
        if data and 'content' in data:
            trigger = data.get('topic_trigger', 'Unknown Topic')
            memories.append(f"PAST SUCCESSFUL OUTPUT (Trigger: '{trigger}'):\n{data.get('content')}")
        
    return memories

# --- 3. FIXED PARSER (Includes Organic Results Always) ---
def _parse_serp_features(results):
    extracted_features = []
    # 1. AI Overview
    if "ai_overview" in results:
        aio = results["ai_overview"]
        if "text_blocks" in aio:
            aio_parts = ["**Google AI Overview:**"]
            for block in aio["text_blocks"]:
                snippet = block.get("snippet", "")
                if block.get("type") == "heading": aio_parts.append(f"*{snippet}*")
                elif block.get("type") == "paragraph": aio_parts.append(snippet)
                elif block.get("type") == "list" and "list" in block:
                    aio_parts.append("\n".join([f" - {i.get('snippet')}" for i in block["list"]]))
            if len(aio_parts) > 1: extracted_features.append("\n\n".join(aio_parts))

    # 2. Knowledge Graph
    if "knowledge_graph" in results:
        kg = results["knowledge_graph"]
        title = kg.get("title", "Knowledge Graph")
        desc = kg.get("description", "")
        if not desc and "see_results_about" in kg:
             item = kg["see_results_about"][0]
             desc = f"{item.get('name')} - {', '.join(item.get('extensions', []))}"
        extracted_features.append(f"**Knowledge Graph ({title}):**\n{desc}")

    # 3. Top Stories
    if "top_stories" in results:
        stories = results["top_stories"]
        story_list = [f"- {s.get('title', 'N/A')} ({s.get('source', 'N/A')})" for s in stories[:5]]
        if story_list: extracted_features.append("**Top Stories:**\n" + "\n".join(story_list))

    # 4. Related Questions
    if "related_questions" in results:
        questions = results["related_questions"]
        qa_list = [f"- Q: {q.get('question')} \n  A: {q.get('snippet', '')}" for q in questions[:3]]
        if qa_list: extracted_features.append("**People Also Ask:**\n" + "\n".join(qa_list))

    # 5. Organic Results (Always included now)
    if "organic_results" in results:
        organic = results["organic_results"][:5]
        organic_list = [f"- {res.get('title', 'Untitled')}: {res.get('snippet', '')} ({res.get('link')})" for res in organic]
        if organic_list:
            extracted_features.append("**Web Results:**\n" + "\n".join(organic_list))

    # Add Grounding Tag to fallback results too
    if extracted_features:
        return "[GROUNDING_CONTENT]\n" + "\n\n---\n\n".join(extracted_features)
    return None

#4. The Specialist Google Web Search Tool (Stability + Full Parsing + Grounded)
def search_google_web(query):
    """
    Specialist tool for deep analysis of the main web SERP.
    Includes robust logic to 'hydrate' lazy-loaded AI Overviews.
    """
    print(f"Tool: Executing WEB search for: '{query}'")
    try:
        
        
        # 1. Primary Search (The Standard 'google' Engine)
        params = {"api_key": SERPAPI_API_KEY, "engine": "google", "q": query, "no_cache": True}
        search = GoogleSearch(params)
        results = search.get_dict()

        # 2. AI Overview Hydration Logic (The Corrected Logic)
        ai_overview = results.get("ai_overview", {})
        
        # Condition: We have a token (indicating lazy loading) but no text blocks
        if ai_overview and "page_token" in ai_overview and "text_blocks" not in ai_overview:
            print("Tool: AI Overview is lazy-loaded. Fetching full content via secondary call...")
            try:
                token = ai_overview['page_token']
                aio_params = {
                    "api_key": SERPAPI_API_KEY,
                    "engine": "google_ai_overview",
                    "page_token": token,
                    #  "q": query 
                }
                
                aio_search = GoogleSearch(aio_params)
                aio_results = aio_search.get_dict()
                
                # --- THE FIX: Robustly checking for data ---
                if "ai_overview" in aio_results and aio_results["ai_overview"]:
                     # Case A: It's nested under 'ai_overview'
                    results["ai_overview"] = aio_results["ai_overview"]
                    print("Tool: AI Overview successfully hydrated (Nested).")
                elif "text_blocks" in aio_results:
                     # Case B: The root object IS the overview (common for this engine)
                    results["ai_overview"] = aio_results
                    print("Tool: AI Overview successfully hydrated (Root).")
                else:
                    # Case C: Google returned no data for the token (common anti-bot measure)
                    print("Tool: Secondary AI Overview call returned no content. Proceeding with partial data.")
                    # We do NOT return None. We proceed with the other rich results (Top Stories, etc.)
                    
            except Exception as e:
                print(f"Tool: Error fetching expanded AI Overview: {e}")
                # We do not crash; we just proceed with the other rich results
        
        # 3. Parse and Return
        # Even if AI Overview failed, we still want the Top Stories and PAA!
        # The previous code might have been returning None if ANY part failed.
        # _parse_serp_features is robust enough to handle missing keys.
        return _parse_serp_features(results) 

    except Exception as e:
        print(f"SerpApi Web Search failed: {e}")
        return None

#5. The Specialist Google Trends Tool (Stability + Full Parsing + Grounded + Breakdown) ---
def search_google_trends(geo="US"):
    """
    Specialist tool for fetching 'Trending Now' searches via SerpApi.
    Refined for production: Clean logs, additive context, and strict parameters.
    """
    print(f"Tool: Executing TRENDS search for geo: '{geo}'")
    try:
        
        
        # 1. Configuration (Hours as string, No Cache for freshness)
        params = {
            "api_key": SERPAPI_API_KEY,
            "engine": "google_trends_trending_now",
            "geo": geo,
            "hours": "24", 
            "no_cache": "true"
        }
        
        search = GoogleSearch(params)
        results = search.get_dict()
        
        trending_list = []

        # 2. Greedy Parsing (Check all potential keys)
        if "trending_searches" in results:
            trending_list = results.get("trending_searches", [])
        elif "daily_searches" in results:
            daily = results.get("daily_searches", [])
            if daily: trending_list = daily[0].get("searches", [])
        elif "realtime_searches" in results:
            trending_list = results.get("realtime_searches", [])
            
        if not trending_list:
            print(f"Tool: No trending data found for {geo}. Keys: {list(results.keys())}")
            return None 

        # 3. Formatting with [GROUNDING_CONTENT]
        formatted_trends = [f"[GROUNDING_CONTENT]\n**Google Trends (Viral Now in {geo}):**"]
        
        for item in trending_list[:15]: 
            # A. Identity
            query = item.get("query") or item.get("title") or "Unknown Topic"
            
            # B. Metadata (Categories)
            categories = item.get("categories", [])
            cat_label = ""
            if categories:
                cat_names = [c.get("name") for c in categories if c.get("name")]
                if cat_names: cat_label = f" *({', '.join(cat_names)})*"

            # C. Metrics (Traffic + Velocity)
            traffic = item.get("formatted_traffic") or item.get("search_volume")
            growth = item.get("increase_percentage")
            
            metrics_str = ""
            if traffic:
                metrics_str = f" [{traffic} searches"
                if growth: metrics_str += f", +{growth}% growth"
                metrics_str += "]"

            # D. Additive Context (News + Related Topics)
            context_snippet = ""
            
            # Try Articles
            articles = item.get("articles", [])
            if articles:
                first = articles[0]
                title = first.get("title") or first.get("article_title") or "News"
                source = first.get("source") or "Source"
                context_snippet += f"\n  - *News:* {title} ({source})"
            
            # Try Breakdown (Add this EVEN if we have articles, for maximum context)
            if "trend_breakdown" in item:
                breakdown = item.get("trend_breakdown", [])
                if breakdown:
                    related = ", ".join(breakdown[:4])
                    context_snippet += f"\n  - *Related:* {related}"
            
            formatted_trends.append(f"- **{query}**{cat_label}{metrics_str}{context_snippet}")
            
        return "\n".join(formatted_trends)

    except Exception as e:
        print(f"SerpApi Trends Search failed: {e}")
        return None
    
# --- 6. ANALYSIS TOOL: Trend History (For enhancing SEO keywords and semantic landscape) ---
def analyze_trend_history(query, geo="US"):
    """
    Analysis Tool: Fetches 'Interest Over Time' for a specific topic.
    Returns: 12-month trajectory, Peak Date, and Current Status (Rising/Falling).
    """
    print(f"Tool: Executing TREND HISTORY analysis for: '{query}' in '{geo}'")
    try:  
        
        params = {
            "api_key": SERPAPI_API_KEY,
            "engine": "google_trends",
            "q": query,
            "geo": geo,
            "data_type": "TIMESERIES",
            "date": "today 12-m", # Standard 1-year lookback
            "no_cache": "true"
        }
        
        search = GoogleSearch(params)
        results = search.get_dict()
        
        interest_over_time = results.get("interest_over_time", {})
        timeline_data = interest_over_time.get("timeline_data", [])
        
        if not timeline_data:
            return f"No historical trend data found for '{query}' in {geo}."

        # --- SMART ANALYSIS ---
        # 1. Extract values
        values = [int(point.get("values", [{}])[0].get("value", 0)) for point in timeline_data]
        dates = [point.get("date", "") for point in timeline_data]
        
        if not values: return "Data found but contained no values."

        # 2. Calculate Key Metrics
        peak_value = max(values)
        peak_index = values.index(peak_value)
        peak_date = dates[peak_index]
        
        current_value = values[-1]
        start_value = values[0]
        
        # 3. Determine Trajectory
        trend_status = "Stable"
        if current_value > start_value * 1.5: trend_status = "Growing significantly 📈"
        elif current_value > start_value * 1.1: trend_status = "Rising slightly ↗️"
        elif current_value < start_value * 0.5: trend_status = "Declining sharply 📉"
        elif current_value < start_value * 0.9: trend_status = "Falling slightly ↘️"
        
        # 4. Generate Report
        report = [
            f"[GROUNDING_CONTENT]",
            f"**Trend Analysis for '{query}' ({geo} - Last 12 Months):**",
            f"- **Status:** {trend_status}",
            f"- **Current Interest:** {current_value}/100",
            f"- **Peak Popularity:** {peak_value}/100 on {peak_date}",
            f"- **Trajectory:** Started at {start_value}, currently at {current_value}."
        ]
        
        return "\n".join(report)

    except Exception as e:
        print(f"Trend Analysis failed: {e}")
        return f"Error analyzing trend history: {e}"

#7. The Specialist Image Search Tool
def search_google_images(query, num_results=5):
    """Specialist tool for analyzing image search results."""
    print(f"Tool: Executing IMAGE search for: '{query}'")
    try:
        
        params = {"api_key": SERPAPI_API_KEY, "engine": "google_images", "q": query}
        results = GoogleSearch(params).get_dict().get("images_results", [])[:num_results]
        if not results: return None
        image_descriptions = [f"- Title: {img.get('title', 'N/A')}, Source: {img.get('source', 'N/A')}" for img in results]
        return "**Image Search Results (Visual Trends):**\n" + "\n".join(image_descriptions)
    except Exception as e:
        print(f"SerpApi Image Search failed: {e}")
        return None
    
#8. The Specialist Video Search Tool
def search_google_videos(query, num_results=5):
    """Specialist tool for analyzing video search results."""
    print(f"Tool: Executing VIDEO search for: '{query}'")
    try:
        
        params = {"api_key": SERPAPI_API_KEY, "engine": "google_videos", "q": query}
        results = GoogleSearch(params).get_dict().get("video_results", [])[:num_results]
        if not results: return None
        video_descriptions = [f"- Title: {vid.get('title', 'N/A')}, Source: {vid.get('source', 'N/A')}, Length: {vid.get('duration', 'N/A')}" for vid in results]
        return "**Video Search Results (Top Videos):**\n" + "\n".join(video_descriptions)
    except Exception as e:
        print(f"SerpApi Video Search failed: {e}")
        return None

#9. The Specialist Scholar Search Tool
def search_google_scholar(query, num_results=3):
    """Specialist tool for finding academic papers and research."""
    print(f"Tool: Executing SCHOLAR search for: '{query}'")
    try:
        
        params = {"api_key": SERPAPI_API_KEY, "engine": "google_scholar", "q": query}
        results = GoogleSearch(params).get_dict().get("organic_results", [])[:num_results]
        if not results: return None
        scholar_summaries = [f"- Title: {res.get('title')}\n  - Publication: {res.get('publication_info', {}).get('summary')}\n  - Snippet: {res.get('snippet', 'N/A')}" for res in results]
        return "**Scholarly Articles (Academic Research):**\n" + "\n".join(scholar_summaries)
    except Exception as e:
        print(f"SerpApi Scholar Search failed: {e}")
        return None

# 10. The Router (Updated with "Double-Tap" Analysis)
def find_trending_keywords(raw_topic, history_context=""):
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
    internal_context = search_long_term_memory(raw_topic)

    # Prepare RAG text for the prompt
    rag_text = "No relevant long-term memories found."
    if internal_context:
        # Flatten the list of strings into a single block of text
        rag_content_str = "\n".join([str(item) for item in internal_context])
        rag_text = rag_content_str
        
        context_snippets.append(internal_context)
        tool_logs.append({"event_type": "tool_call", "tool_name": "internal_knowledge_retrieval", "status": "success"})

    # 2. Sensory Array Router (Decides based on RAW input)
    tool_choice_prompt = f"""
    Analyze the user's query to select the single best research tool.

    CONVERSATION HISTORY:
    {history_context}

    LONG-TERM MEMORY (RAG Results):
    {rag_text}

    USER'S CORE QUERY: '{raw_topic}'

    ### DECISION PROTOCOL (CHECK IN ORDER):
    1. **CHECK HISTORY (Priority #1):**
        - If the user is asking for information that CAN BE GATHERED from the "LONG-TERM MEMORY" or "CONVERSATION HISTORY", select **NONE**.
        - **CRITICAL EXCEPTION:** If the user asks for **REASONS**, **DRIVERS**, **CAUSES**, or **"WHY"** something happened (and the text history contains only numbers/stats but not the *explanation*), **DO NOT** select NONE. Proceed to Priority #2.

    2. **CHECK CREATIVE INTENT (Priority #2):**
        - If the user asks you to **DRAFT**, **CREATE**, or **WRITE** content for an entity that is clearly **INVENTED** in the prompt (e.g., phrases like "a company called X", "a new brand named Y", "a character named Z"), select **NONE**.
        - **CRITICAL EXCEPTION:** If the entity is a **REAL-WORLD ENTITY** (e.g., "Google", "Nike", "The Beatles"), DO NOT select NONE. Select **WEB** or **SIMPLE** to ensure the draft is grounded in reality.

    3. **NEW RESEARCH (Priority #3):**
        - If the user wants you to gather NEW information not in history, use a tool below.
        - If the user wants you to explain a trend (and the history contains only numbers/stats but not the *explanation*), DO NOT select NONE. Select a tool below.

    ### AVAILABLE TOOLS:
    - WEB: For deep analysis of web results, knowledge graphs, AI Overviews, and top stories related to the topic.
        Also, select this if the user asks about:
        1. RECENT events, news, or stories (e.g. "top stories", "today", "current").
        2. Specific FACTS that might be outdated in my training data.
        3. Entities I might not know (e.g. specific companies, people).
    - IMAGES: If the user explicitly asks for images, pictures, or visual inspiration.
    - VIDEOS: If the user explicitly asks for videos, clips, or tutorials.
    - SCHOLAR: If the user is asking for academic papers, research, studies, or highly credible sources.
    - TRENDS: Use this ONLY for general "Trending Now" or "Viral Now" searches for a region (e.g., "What's trending in US?"). 
        * WARNING: This tool ignores the user's specific topic and returns general viral topics. 
        * Format: "TRENDS:CODE" (e.g., "TRENDS:NG" for Nigeria). 
        * Default: "TRENDS:US".
    - ANALYSIS: For topic-specific "interest over time", "growth", "history", or "how has X trended?". 
        * Use this when the user asks about a SPECIFIC topic's popularity curve or historical trend.
        * Format: "ANALYSIS:CODE" (e.g., "ANALYSIS:US"). 
    - SIMPLE: For all other general web searches.
    - NONE: Select this if:
        1. The answer is a CORRECTION or REFINEMENT (e.g., "I said strategy not cluster").
        2. The answer can be derived from CONVERSATION HISTORY or LONG-TERM MEMORY.
        3. The user has PROVIDED the source material (e.g., "Summarize this text", "Create a strategy from these points") and just wants you to process/organize it.
        4. The user asks for an OPINION, EVALUATION, or LOGICAL DISPUTE based on known facts (e.g., "Do you agree?", "Critique this", "Validate my point").
        5. The user asks for **CREATIVE WRITING** regarding a hypothetical or user-defined entity (e.g., "Draft a short story", "Create a brand voice").

    Which tools are most appropriate? You may select multiple tools if the query has multiple distinct parts. 
    Respond ONLY with the tool selection(s), comma-separated (e.g., "WEB, ANALYSIS:US" or "IMAGES" or "NONE").
    
    """
    # 3. Robust Parsing Logic
    try:
        raw_response = flash_model.generate_content(tool_choice_prompt).text.strip().upper()
        clean_response = raw_response.replace("*", "").replace("`", "").replace("'", "").replace('"', "").rstrip(".")
        
        # Support for multi-tool selection (comma-separated)
        selected_tools = [t.strip() for t in clean_response.split(',')]
        print(f"Sensory Array decided on tools: {selected_tools}")
    except Exception as e:
        print(f"Router Parse Error: {e}. Defaulting to SIMPLE.")
        selected_tools = ["SIMPLE"]

    if any("NONE" in choice for choice in selected_tools):
        return {"context": context_snippets, "tool_logs": tool_logs}

    # Iterate through each selected tool
    for tool_selection in selected_tools:
        try:
            parts = tool_selection.split(':')
            choice = parts[0].strip()       
            geo_code = parts[1].strip() if len(parts) > 1 else "US" 
        except Exception:
            choice = "SIMPLE"
            geo_code = "US"

        print(f"Executing Sensory Tool: '{choice}' (Geo: '{geo_code}')")

        search_query = raw_topic 
        research_context = None
        tool_name = "unknown"

        # --- 1. VIRAL DISCOVERY (No specific query needed) ---
        if "TRENDS" in choice:
            print(f"Entering TRENDS logic for {geo_code}...") 
            research_context = search_google_trends(geo=geo_code)
            tool_name = "serpapi_trends_search"

        # --- 2. TREND ANALYSIS (The "Double-Tap" Upgrade) ---
        elif "ANALYSIS" in choice:
            # A. Extract the specific entity (e.g., "Tinubu")
            search_query = extract_trend_term(raw_topic, history=history_context)
            print(f"Entering TREND ANALYSIS logic for '{search_query}' in {geo_code}...")
            
            # B. TAP 1: Quantitative Data (The Graph/Stats)
            stats_context = analyze_trend_history(search_query, geo=geo_code)
            
            # C. TAP 2: Qualitative Data (The News - NEW FIX)
            try:
                print(f"  + Fetching Qualitative News Context for '{search_query}'...")
                news_params = {
                    "engine": "google",
                    "q": search_query,
                    "gl": geo_code,
                    "tbm": "nws", # Force News Tab
                    "num": 3,     # Top 3 stories are enough for context
                    "api_key": SERPAPI_API_KEY
                }
                news_search = GoogleSearch(news_params) 
                news_results = news_search.get_dict().get("news_results", [])
                
                news_text = ""
                if news_results:
                    news_text = "\n[NEWS CONTEXT - DRIVERS OF TREND]:\n"
                    for n in news_results:
                        news_text += f"- {n.get('title')} (Source: {n.get('source')})\n"
                
                # Merge Stats + News
                research_context = f"{stats_context}\n{news_text}"
                
            except Exception as e:
                print(f"  ! News fetch failed (Non-Critical): {e}")
                research_context = stats_context # Fallback to just stats if news fails

            tool_name = "serpapi_trend_analysis_with_news"

        # --- 3. STANDARD SEARCH TOOLS (Use complex SEO extractor) ---
        elif choice in ["WEB", "IMAGES", "VIDEOS", "SCHOLAR", "SIMPLE"]:
            search_query = extract_core_topic(raw_topic, history=history_context)
            if choice == "WEB":
                research_context = search_google_web(search_query)
                tool_name = "serpapi_web_search"
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
        if not research_context:
            print(f"Primary tool failed or SIMPLE selected. Initiating fallback search...")
            
            # 1. Remove strict operators that restrict results
            relaxed_query = search_query.replace('"', '').replace('(', '').replace(')', '')
            
            # 2. Remove Exclusions (e.g., -guide) as they often cause 0 results
            relaxed_query = re.sub(r'\s-\S+', '', relaxed_query)
            
            # 3. Clean up the boolean words left behind
            relaxed_query = re.sub(r'\b(OR|AND)\b', '', relaxed_query)
            
            # 4. Collapse whitespace
            fallback_query = " ".join(relaxed_query.split())

            print(f"Relaxed Query: '{search_query}' \n       -> '{fallback_query}'")
            
            try:
                # 1. Powerful Web Search (With Relaxed Query)
                research_context = search_google_web(fallback_query)
                if research_context:
                    tool_name = "serpapi_web_search"
                
                # 2. Simple Search Safety Net
                if not research_context:
                    api_key = get_search_api_key()
                    service = build("customsearch", "v1", developerKey=api_key)
                    res = service.cse().list(q=search_query, cx=SEARCH_ENGINE_ID, num=5).execute()
                    google_snippets = [item.get('snippet', '') for item in res.get('items', [])]
                    if google_snippets:
                        research_context = "[FALLBACK CONTEXT]:\n" + "\n".join(google_snippets)
                        tool_name = "google_simple_search"
                        # tool_logs.append({"event_type": "tool_call", "tool_name": "google_simple_search", "input": fallback_query})

            except Exception as e:
                print(f"Fallback search failed: {e}")

        if research_context:
            context_snippets.append(research_context)
            tool_logs.append({"event_type": "tool_call", "tool_name": tool_name, "input": search_query, "status": "success", "content": research_context})

    return { "context": context_snippets, "tool_logs": tool_logs }


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

#13. The Comprehensive Answer Generator
def generate_comprehensive_answer(topic, context, history=""):
    global model
    is_grounded = any("GROUNDING_CONTENT" in str(c) for c in context)
    if is_grounded:
        temperature = 0.0
        system_instruction = "CRITICAL INSTRUCTION: You are in READING MODE. Base answer PRIMARILY on 'GROUNDING_CONTENT'."
    else:
        temperature = 0.7
        system_instruction = "You are an expert AI assistant. Use the research context to provide accurate answers."
    prompt = f"""
        {system_instruction}
        
        Conversation History:
        {history}
        
        Current Request: "{topic}"
        
        Research Context:
        {context}
        
        Answer:
        """
    return model.generate_content(prompt, generation_config={"temperature": temperature}).text.strip()

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
        
        # 2. Initialize Vertex AI explicitly for the Flash Model (Triage)
        # We do this separately because Flash is likely still a Gemini model
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        flash_model = GenerativeModel(FLASH_MODEL_NAME)
        db = firestore.Client()

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
        # NEW FIX: Query the subcollection instead of the 'event_log' array
        # We limit to the last 20 turns to keep the context window focused and cheap
        events_ref = doc_ref.collection('events')
        
        # 1. Sort DESCENDING (Newest first) and get the top 20
        query = events_ref.order_by('timestamp', direction=firestore.Query.DESCENDING).limit(20)
        
        # Stream, convert to dict, and REVERSE to get chronological order (Old -> New)
        history_events = [doc.to_dict() for doc in query.stream()][::-1]


    # START: MEMORY EXPANSION & PHOTOGRAPHIC RECALL ---
    formatted_history = []
    for i, e in enumerate(history_events[-10:]):
        if e.get('event_type') == 'tool_call':
            # FIX: Explicitly read the 'content' field to remember previous search results
            # Cap at 5000 chars to save tokens while keeping rich context
            entry = f"Turn {i+1} (System Search Results): {e.get('content', '')[:5000]}" 
        else:
            # FIX: Increased limit from 200 to 5000 chars so clusters/long prompts aren't lobotomized
            entry = f"Turn {i+1} ({e.get('event_type')}): {e.get('text', '')[:5000]}"
        formatted_history.append(entry)

    history_text = "\n".join(formatted_history)
    
    print(f"Worker loaded context with {len(history_events)} past events.")

    try:
        # --- STEP 3: TRIAGE (Now includes SOCIAL category) ---
        # Note: We triage BEFORE distilling or researching to save cost/latency.
        
        # --- KEYWORD SAFEGUARD (Force DIRECT_ANSWER for non-article requests) ---
        direct_convo_keywords = ["lesson plan", "plan", "outline", "summarize", "research", "strategy", "brief"]
        if any(kw in sanitized_topic.lower() for kw in direct_convo_keywords):
            print(f"Keyword Safeguard: Direct conversation intent detected. Forcing DIRECT_ANSWER.")
            intent = "DIRECT_ANSWER"
        else:
            triage_prompt = f"""
            Analyze the user's latest message in the context of the conversation history.
            Classify the PRIMARY GOAL into one of five categories:

            1.  **SOCIAL_CONVERSATION**: The user is engaging in "obvious" small talks, greetings, expressing gratitude, or meta-comments, or comments that DOES NOT require external information or research.
                *Examples:* "Thanks!", "Great job.", "I have to go now.", "You are funny.", "Okay.", "Good Morning."

            2.  **TOPIC_CLUSTER_PROPOSAL**: The user specifically wants you to GENERATE a NEW list of topics, topic clusters, pillar page and/or supporting pages outline, or a hierarchical content structure.
                *   **CRITICAL EXCLUSION:** If the user asks for a "Strategy", "Plan", or "Brief" regarding *existing or user-provided* clusters, select DIRECT_ANSWER.

            3.  **THEN_VS_NOW_PROPOSAL**: The user EXPLICITLY asks for a "Then vs Now" story, draft, or structured comparison.

            4.  **PSEO_ARTICLE**: The user intent is to PRODUCE a COMPLETE, PUBLICATION-READY ASSET (Full Article, Guide, or Blog Post) intended for publication in a CMS.
                * **CRITICAL EXCLUSIONS (Force DIRECT_ANSWER):** 
                    - If the user asks for an **Outline**, **Summary**, **Research**, **Strategy**, **Brief**, or **Lesson Plan**.
                    - If the user asks for **planning or structuring frameworks** (e.g., SEO Brief, Content Strategy).
                * **Semantics:** The user is asking for the *final end-product* (The Blog Post itself) to be written.
                * **Constraint:** Only select this if the user wants the ACTUAL FINAL DRAFT created in the CMS.

            5.  **DIRECT_ANSWER**: This is the default for ALL research, planning, reasoning, logic, and CREATIVITY tasks.
                * **Select this for:** **Outlines**, **Summaries**, **Research Answers**, **Planningh**, or general questions about news, facts, or strategies.
                * **Even if:** The user is polite or casual (e.g., "I was just wondering...", "Can you check...", "Do you know...").

            CONVERSATION HISTORY:
            {history_text}

            USER REQUEST: "{sanitized_topic}"

            THOUGHT PROCESS:
            1. Is this a request for an outline, summary, research, or lesson plan? (Yes -> DIRECT_ANSWER)
            2. Is this just a strategy or planning task? (Yes -> DIRECT_ANSWER)
            3. Is it a request for a final, published blog post in the CMS? (Yes -> PSEO_ARTICLE)

            CRITICAL: Respond with ONLY the category name.
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

        # 1. Research (Handle URL or Keywords)
        url = extract_url_from_text(sanitized_topic)
        if url:
            article_text = fetch_article_content(url)
            research_data = {"context": [f"GROUNDING_CONTENT:\n{article_text}"], "tool_logs": []}
        else:
            research_data = find_trending_keywords(sanitized_topic, history_context=history_text)
        
        if "tool_logs" in research_data: new_events.extend(research_data["tool_logs"])

        # CHANGE: Ensure downstream functions use the raw topic too, so they see the full request
        clean_topic = sanitized_topic

        # 3. Generate Output based on Intent
        if intent == "DIRECT_ANSWER":
            answer_text = generate_comprehensive_answer(original_topic, research_data['context'], history=history_text)
            new_events.append({"event_type": "agent_answer", "text": answer_text})
            
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
            requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={"session_id": session_id, "type": "social", "message": answer_text, "channel_id": slack_context['channel'], "thread_ts": slack_context['ts']}, verify=True)
            return jsonify({"msg": "Answer sent"}), 200

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
            }, verify=True)

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
    if db is None: db = firestore.Client()
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
        
        # 1. Set the data (Add to batch)
        batch.set(doc_ref, {
            "content": text_segment,
            "embedding": Vector(embedding_obj.values),
            "topic_trigger": topic, 
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