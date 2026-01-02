import os
import re
import json
import requests
import certifi
import vertexai
import uvicorn
from fastapi import FastAPI, Request
from bs4 import BeautifulSoup
from google.cloud import secretmanager, firestore
from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel
from serpapi import GoogleSearch
from googleapiclient.discovery import build

# --- Configuration ---
PROJECT_ID = os.environ.get("PROJECT_ID")
LOCATION = os.environ.get("LOCATION", "us-central1")
SEARCH_ENGINE_ID = os.environ.get("SEARCH_ENGINE_ID")
SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY")
BROWSERLESS_API_KEY = os.environ.get("BROWSERLESS_API_KEY")
FLASH_MODEL_NAME = os.environ.get("FLASH_MODEL_NAME", "gemini-2.0-flash-exp")
DEFAULT_GEO = os.environ.get("DEFAULT_GEO", "NG") # Default to Nigeria as primary operating region, but configurable

# --- Initialize FastAPI ---
app = FastAPI(title="Sensory-Tools-Server")

# --- Global Clients ---
db = None
secret_client = None
flash_model = None

def get_db():
    global db
    if db is None:
        db = firestore.Client(project=PROJECT_ID)
    return db

def get_secret_client():
    global secret_client
    if secret_client is None:
        secret_client = secretmanager.SecretManagerServiceClient()
    return secret_client

def get_search_api_key():
    client = get_secret_client()
    response = client.access_secret_version(name=f"projects/{PROJECT_ID}/secrets/google-search-api-key/versions/latest")
    return response.payload.data.decode("UTF-8")

# --- Initialize Vertex AI ---
vertexai.init(project=PROJECT_ID, location=LOCATION)

def get_flash_model():
    global flash_model
    if flash_model is None:
        flash_model = GenerativeModel(FLASH_MODEL_NAME)
    return flash_model

# --- Shared Helper: SERP Parser ---
def _parse_serp_features(results):
    extracted_features = []
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

    if "knowledge_graph" in results:
        kg = results["knowledge_graph"]
        title = kg.get("title", "Knowledge Graph")
        desc = kg.get("description", "")
        if not desc and "see_results_about" in kg:
             item = kg["see_results_about"][0]
             desc = f"{item.get('name')} - {', '.join(item.get('extensions', []))}"
        extracted_features.append(f"**Knowledge Graph ({title}):**\n{desc}")

    if "top_stories" in results:
        stories = results["top_stories"]
        story_list = [f"- {s.get('title', 'N/A')} ({s.get('source', 'N/A')})" for s in stories[:5]]
        if story_list: extracted_features.append("**Top Stories:**\n" + "\n".join(story_list))

    if "related_questions" in results:
        questions = results["related_questions"]
        qa_list = [f"- Q: {q.get('question')} \n  A: {q.get('snippet', '')}" for q in questions[:3]]
        if qa_list: extracted_features.append("**People Also Ask:**\n" + "\n".join(qa_list))

    if "organic_results" in results:
        organic = results["organic_results"][:5]
        organic_list = [f"- {res.get('title', 'Untitled')}: {res.get('snippet', '')} ({res.get('link')})" for res in organic]
        if organic_list:
            extracted_features.append("**Web Results:**\n" + "\n".join(organic_list))

    if extracted_features:
        return "[GROUNDING_CONTENT]\n" + "\n\n---\n\n".join(extracted_features)
    return "No significant results found."

# --- Unified Tool Routing ---
def handle_tool_call(name, arguments):
    """
    Central dispatcher for all sensory tools.
    Logs every call for real-time debugging.
    """
    print(f"MCP Hub: Handling tool call '{name}' with args: {arguments}")
    
    if name == "google_web_search":
        params = {"api_key": SERPAPI_API_KEY, "engine": "google", "q": arguments.get("query"), "no_cache": True}
        results = GoogleSearch(params).get_dict()
        return _parse_serp_features(results)

    elif name == "scrape_article":
        url = arguments.get("url")
        endpoint = f"https://production-sfo.browserless.io/content?token={BROWSERLESS_API_KEY}&stealth=true"
        payload = {"url": url, "rejectResourceTypes": ["image", "media", "font"], "gotoOptions": {"timeout": 15000, "waitUntil": "networkidle2"}}
        response = requests.post(endpoint, json=payload, timeout=20, verify=certifi.where())
        soup = BeautifulSoup(response.content, 'html.parser')
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "iframe", "ad"]): tag.decompose()
        text = soup.get_text(separator='\n\n')
        clean_text = re.sub(r'\n\s*\n', '\n\n', text).strip()
        return clean_text[:20000]

    elif name == "rag_search":
        query = arguments.get("query")
        db = get_db()
        embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
        query_embedding = embedding_model.get_embeddings([query[:6000]])[0].values
        collection = db.collection('knowledge_base')
        results = collection.find_nearest(
            vector_field="embedding",
            query_vector=Vector(query_embedding),
            distance_measure=DistanceMeasure.COSINE,
            limit=3
        ).get()
        memories = []
        for res in results:
            data = res.to_dict()
            if data and 'content' in data:
                trigger = data.get('topic_trigger', 'Unknown Topic')
                memories.append(f"PAST SUCCESSFUL OUTPUT (Trigger: '{trigger}'):\n{data.get('content')}")
        return "\n".join(memories) if memories else "No relevant memories found."

    elif name == "google_trends":
        geo = arguments.get("geo", "US")
        params = {"api_key": SERPAPI_API_KEY, "engine": "google_trends_trending_now", "geo": geo, "hours": "24"}
        results = GoogleSearch(params).get_dict()
        return json.dumps(results, indent=2)[:5000]

    elif name == "trend_analysis":
        query = arguments.get("query")
        geo = arguments.get("geo", "US")
        params = {"api_key": SERPAPI_API_KEY, "engine": "google_trends", "q": query, "geo": geo, "data_type": "TIMESERIES", "date": "today 12-m"}
        results = GoogleSearch(params).get_dict()
        return json.dumps(results, indent=2)[:5000]

    elif name == "google_images_search":
        query = arguments.get("query")
        params = {"api_key": SERPAPI_API_KEY, "engine": "google_images", "q": query}
        results = GoogleSearch(params).get_dict().get("images_results", [])[:5]
        image_descriptions = [f"- Title: {img.get('title', 'N/A')}, Source: {img.get('source', 'N/A')}" for img in results]
        return "**Image Search Results:**\n" + "\n".join(image_descriptions)

    elif name == "google_videos_search":
        query = arguments.get("query")
        params = {"api_key": SERPAPI_API_KEY, "engine": "google_videos", "q": query}
        results = GoogleSearch(params).get_dict().get("video_results", [])[:5]
        video_descriptions = [f"- Title: {vid.get('title', 'N/A')}, Source: {vid.get('source', 'N/A')}, Length: {vid.get('duration', 'N/A')}" for vid in results]
        return "**Video Search Results:**\n" + "\n".join(video_descriptions)

    elif name == "google_scholar_search":
        query = arguments.get("query")
        params = {"api_key": SERPAPI_API_KEY, "engine": "google_scholar", "q": query}
        results = GoogleSearch(params).get_dict().get("organic_results", [])[:3]
        scholar_summaries = [f"- Title: {res.get('title')}\n  - Publication: {res.get('publication_info', {}).get('summary')}\n  - Snippet: {res.get('snippet', 'N/A')}" for res in results]
        return "**Scholarly Articles:**\n" + "\n".join(scholar_summaries)

    elif name == "google_news_search":
        query = arguments.get("query")
        geo = arguments.get("geo", "US")
        params = {"engine": "google", "q": query, "gl": geo, "tbm": "nws", "num": 3, "api_key": SERPAPI_API_KEY}
        results = GoogleSearch(params).get_dict().get("news_results", [])
        news_text = "\n[NEWS CONTEXT]:\n" + "\n".join([f"- {n.get('title')}" for n in results]) if results else "No news found."
        return news_text

    elif name == "google_simple_search":
        query = arguments.get("query")
        api_key = get_search_api_key()
        service = build("customsearch", "v1", developerKey=api_key)
        res = service.cse().list(q=query, cx=SEARCH_ENGINE_ID, num=5).execute()
        google_snippets = [item.get('snippet', '') for item in res.get('items', [])]
        return "[GROUNDING_CONTENT (Fallback)]:\n" + "\n".join(google_snippets) if google_snippets else "No fallback results found."

    elif name == "detect_geo":
        query = arguments.get("query")
        history = arguments.get("history", "")
        model = get_flash_model()
        prompt = f"""
        Identify the ISO 3166-1 alpha-2 country code(s) for the geographic focus of the user's request.
        
        RULES:
        1. If the user mentions one or more countries (e.g., Nigeria, USA, UK), return their 2-letter codes (NG, US, GB).
        2. If the user mentions cities (e.g., Lagos, London), return the codes for their respective countries.
        3. If no geographic focus is found, return '{DEFAULT_GEO}' (The primary operating region).
        4. If multiple locations are mentioned, return them as a comma-separated list (e.g., "NG, KE").
        5. Return ONLY the 2-letter code(s).
        
        HISTORY: {history}
        USER INPUT: "{query}"
        
        COUNTRY CODE(S):
        """
        try:
            # We remove spaces and non-alphanumeric characters (except commas) to ensure clean codes
            raw_geo = model.generate_content(prompt).text.strip().upper()
            clean_geo = re.sub(r'[^A-Z,]', '', raw_geo)
            return clean_geo if clean_geo else DEFAULT_GEO
        except Exception:
            return DEFAULT_GEO

    elif name == "detect_intent":
        query = arguments.get("query")
        model = get_flash_model()
        prompt = f"""
        Analyze the user's request to identify the specific deliverable or research depth requested.
        
        CATEGORIES:
        - TIMELINE: Requesting historical dates/progression.
        - TABLE: Requesting structured comparison or lists.
        - CSV: Explicit mention of spreadsheet format.
        - NONE: General conversation or drafting without specific structural requirements.
        
        USER INPUT: "{query}"
        
        CATEGORY:
        """
        try:
            intent = model.generate_content(prompt).text.strip().upper()
            return intent if intent in ["TIMELINE", "TABLE", "CSV"] else "NONE"
        except Exception:
            return "NONE"

    return f"Error: Tool '{name}' not found."

# --- The Lightweight MCP Messenger ---
@app.post("/messages")
async def messages(request: Request):
    """
    Direct POST endpoint for the Worker's legacy MCP client.
    Supports statutory tools/call logic without forcing SSE session handshakes.
    
    This is the lightweight 'Hub' architecture that ensures low-latency sensory retrieval.
    """
    body = await request.json()
    method = body.get("method")
    
    # Log incoming request for traffic monitoring
    print(f"MCP Hub: Received '{method}' request from Worker.")
    
    if method == "tools/call":
        params = body.get("params", {})
        name = params.get("name")
        arguments = params.get("arguments", {})
        
        result_text = handle_tool_call(name, arguments)
        
        # Return in the exact JSON structure the Worker expects
        return {
            "result": {
                "content": [
                    {"type": "text", "text": result_text}
                ]
            }
        }
    
    return {"error": "Method not supported"}, 400

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
