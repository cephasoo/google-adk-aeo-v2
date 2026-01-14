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
from vertexai.generative_models import GenerativeModel, HarmCategory, HarmBlockThreshold, Part
from vertexai.preview.vision_models import ImageGenerationModel
from serpapi import GoogleSearch
from googleapiclient.discovery import build
import base64

# --- Configuration ---
PROJECT_ID = os.environ.get("PROJECT_ID")
LOCATION = os.environ.get("LOCATION", "us-central1")
SEARCH_ENGINE_ID = os.environ.get("SEARCH_ENGINE_ID")
# These will be fetched from Secret Manager in production
SERPAPI_API_KEY = None
BROWSERLESS_API_KEY = None
FLASH_MODEL_NAME = os.environ.get("FLASH_MODEL_NAME", "gemini-2.0-flash-exp")
DEFAULT_GEO = os.environ.get("DEFAULT_GEO", "NG") # Default to Nigeria as primary operating region, but configurable

# --- Initialize FastAPI ---
app = FastAPI(title="Sensory-Tools-Server")

# --- Global Clients ---
db = None
secret_client = None
flash_model = None

# --- Safety Configuration (ADK/RAI Compliant) ---
safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
}

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

def get_secret(secret_id):
    """Retrieves a secret from Google Cloud Secret Manager."""
    client = get_secret_client()
    try:
        response = client.access_secret_version(name=f"projects/{PROJECT_ID}/secrets/{secret_id}/versions/latest")
        return response.payload.data.decode("UTF-8")
    except Exception as e:
        print(f"Error retrieving secret '{secret_id}': {e}")
        return None

def initialize_secrets():
    global SERPAPI_API_KEY, BROWSERLESS_API_KEY
    SERPAPI_API_KEY = get_secret("serpapi-api-key")
    BROWSERLESS_API_KEY = get_secret("browserless-api-key")

# --- Initialize Vertex AI ---
vertexai.init(project=PROJECT_ID, location=LOCATION)

@app.on_event("startup")
async def startup_event():
    """Initializes external services and secrets on startup."""
    initialize_secrets()

def get_flash_model():
    global flash_model
    if flash_model is None:
        flash_model = GenerativeModel(FLASH_MODEL_NAME, safety_settings=safety_settings)
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
def scrub_pii(text):
    """Simple PII scrubbing for tool outputs."""
    if not isinstance(text, str): return text
    # Mask emails
    text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[EMAIL_MASKED]', text)
    # Mask common phone formats (simple regex)
    text = re.sub(r'\+?\d{1,3}[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}', '[PHONE_MASKED]', text)
    return text

def handle_tool_call(name, arguments):
    """
    Central dispatcher for all sensory tools.
    Logs every call for real-time debugging and enforces validation/scrubbing.
    """
    print(f"MCP Hub: Calling Tool [{name}] | Args: {json.dumps(arguments)}")
    
    # Tool-Level Input Validation (Architectural Safety)
    if not name or not isinstance(arguments, dict):
        return "Error: Invalid tool call format."

    # Namespacing / Allowed List check
    allowed_tools = [
        "google_web_search", "scrape_article", "rag_search", "google_trends", 
        "trend_analysis", "google_images_search", "google_videos_search", 
        "google_scholar_search", "google_news_search", "google_simple_search",
        "detect_geo", "detect_intent", "analyze_image", "generate_image"
    ]
    if name not in allowed_tools:
        return f"Error: Tool '{name}' is not in the allowed list."

    result = ""
    if name == "google_web_search":
        if not arguments.get("query"): return "Error: Missing query."
        params = {"api_key": SERPAPI_API_KEY, "engine": "google", "q": arguments.get("query"), "no_cache": True}
        results = GoogleSearch(params).get_dict()
        result = _parse_serp_features(results)

    elif name == "scrape_article":
        url = arguments.get("url")
        if not url or not url.startswith("http"): return "Error: Invalid URL."
        endpoint = f"https://production-sfo.browserless.io/content?token={BROWSERLESS_API_KEY}&stealth=true"
        # Removed "image" from rejectResourceTypes to allow visual assets to load
        payload = {"url": url, "rejectResourceTypes": ["media", "font"], "gotoOptions": {"timeout": 15000, "waitUntil": "networkidle2"}}
        response = requests.post(endpoint, json=payload, timeout=20, verify=certifi.where())
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract high-value image URLs (Social-Media Aware Detection)
        found_images = []
        exclude_patterns = ["profile_images", "avatar", "icon", "logo", "header_photo", "normal", "mini"]
        
        # Social media platforms often use 'format=jpg' in query-strings instead of extensions
        img_ext_pattern = re.compile(r'\.(jpg|jpeg|png|webp|gif|pdf)', re.I)
        social_media_fmt_pattern = re.compile(r'format=(jpg|jpeg|png|webp|gif)', re.I)

        for img in soup.find_all("img"):
            # Check all common source attributes
            img_src = img.get("src") or img.get("data-src") or img.get("srcset") or img.get("data-original")
            if img_src and img_src.startswith("http"):
                # Handle srcset which might contain multiple URLs
                clean_src = img_src.split(',')[0].split(' ')[0].strip()
                
                # Check for standard extension OR social media query-string format
                has_valid_ext = img_ext_pattern.search(clean_src) or social_media_fmt_pattern.search(clean_src)
                
                if not has_valid_ext:
                    continue

                # High-Value Check: Skip images that look like UI elements or profile pics
                if any(p in clean_src.lower() for p in exclude_patterns):
                    # Exception: If it's on a known media host (like pbs.twimg.com/media), keep it anyway
                    if "twimg.com/media" not in clean_src.lower():
                        continue
                    
                if clean_src not in found_images:
                    found_images.append(clean_src)
            if len(found_images) >= 5: break

        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "iframe", "ad"]): tag.decompose()
        text = soup.get_text(separator='\n\n')
        clean_text = re.sub(r'\n\s*\n', '\n\n', text).strip()
        
        # Append images as grounded context
        image_context = "\n\n[DETECTED_IMAGES]:\n" + "\n".join(found_images) if found_images else ""
        result = (clean_text[:18000] + image_context).strip()

    elif name == "rag_search":
        query = arguments.get("query")
        session_id = arguments.get("session_id")
        if not query: return "Error: Missing query."
        if not session_id: return "Error: Missing session_id for isolated memory search."
        
        db = get_db()
        embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
        query_embedding = embedding_model.get_embeddings([query[:6000]])[0].values
        
        # Enforce Session-Level Isolation (ADK Principle)
        collection = db.collection('knowledge_base')
        results = collection.where("source_session", "==", session_id).find_nearest(
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
        result = "\n".join(memories) if memories else "No relevant memories found."

    elif name == "google_trends":
        geo = arguments.get("geo", "US")
        params = {"api_key": SERPAPI_API_KEY, "engine": "google_trends_trending_now", "geo": geo, "hours": "24"}
        results = GoogleSearch(params).get_dict()
        result = json.dumps(results, indent=2)[:5000]

    elif name == "trend_analysis":
        query = arguments.get("query")
        if not query: return "Error: Missing query."
        geo = arguments.get("geo", "US")
        params = {"api_key": SERPAPI_API_KEY, "engine": "google_trends", "q": query, "geo": geo, "data_type": "TIMESERIES", "date": "today 12-m"}
        results = GoogleSearch(params).get_dict()
        result = json.dumps(results, indent=2)[:5000]

    elif name == "google_images_search":
        query = arguments.get("query")
        if not query: return "Error: Missing query."
        params = {"api_key": SERPAPI_API_KEY, "engine": "google_images", "q": query}
        results = GoogleSearch(params).get_dict().get("images_results", [])[:5]
        image_descriptions = [f"- Title: {img.get('title', 'N/A')}, Source: {img.get('source', 'N/A')}" for img in results]
        result = "**Image Search Results:**\n" + "\n".join(image_descriptions)

    elif name == "google_videos_search":
        query = arguments.get("query")
        if not query: return "Error: Missing query."
        params = {"api_key": SERPAPI_API_KEY, "engine": "google_videos", "q": query}
        results = GoogleSearch(params).get_dict().get("video_results", [])[:5]
        video_descriptions = [f"- Title: {vid.get('title', 'N/A')}, Source: {vid.get('source', 'N/A')}, Length: {vid.get('duration', 'N/A')}" for vid in results]
        result = "**Video Search Results:**\n" + "\n".join(video_descriptions)

    elif name == "google_scholar_search":
        query = arguments.get("query")
        if not query: return "Error: Missing query."
        params = {"api_key": SERPAPI_API_KEY, "engine": "google_scholar", "q": query}
        results = GoogleSearch(params).get_dict().get("organic_results", [])[:3]
        scholar_summaries = [f"- Title: {res.get('title')}\n  - Publication: {res.get('publication_info', {}).get('summary')}\n  - Snippet: {res.get('snippet', 'N/A')}" for res in results]
        result = "**Scholarly Articles:**\n" + "\n".join(scholar_summaries)

    elif name == "google_news_search":
        query = arguments.get("query")
        if not query: return "Error: Missing query."
        geo = arguments.get("geo", "US")
        params = {"engine": "google", "q": query, "gl": geo, "tbm": "nws", "num": 3, "api_key": SERPAPI_API_KEY}
        results = GoogleSearch(params).get_dict().get("news_results", [])
        result = "\n[NEWS CONTEXT]:\n" + "\n".join([f"- {n.get('title')}" for n in results]) if results else "No news found."

    elif name == "google_simple_search":
        query = arguments.get("query")
        if not query: return "Error: Missing query."
        api_key = get_secret("google-search-api-key")
        service = build("customsearch", "v1", developerKey=api_key)
        res = service.cse().list(q=query, cx=SEARCH_ENGINE_ID, num=5).execute()
        google_snippets = [item.get('snippet', '') for item in res.get('items', [])]
        result = "[GROUNDING_CONTENT (Fallback)]:\n" + "\n".join(google_snippets) if google_snippets else "No fallback results found."

    elif name == "detect_geo":
        query = arguments.get("query")
        if not query: return "Error: Missing query."
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
            result = clean_geo if clean_geo else DEFAULT_GEO
        except Exception:
            result = DEFAULT_GEO

    elif name == "detect_intent":
        query = arguments.get("query")
        if not query: return "Error: Missing query."
        model = get_flash_model()
        prompt = f"""
        Analyze the user's request to identify the primary structural goal or specific deliverable requested.
        
        CATEGORIES:
        - FORMAT_TIMELINE: Progression of events over time.
        - FORMAT_TABLE: Comparative data or technical specs.
        - FORMAT_CHART: Visual representations (Bar charts, Pie charts, Flow diagrams, Sequence diagrams, Mindmaps).
        - FORMAT_LISTICLE: High-impact numbered/bulleted lists (e.g., "Top 5 reasons", "Steps to X").
        - FORMAT_CSV: Specific mention of spreadsheet/exportable data.
        - SIGNAL_BLOCK: Prompts that are harmful, illegal, sexual, or violate safety policies. Encourage the user to seek help by calling any of the numbers on this page (https://www.nigerianmentalhealth.org/helplines) if in Nigeria
        - FORMAT_GENERAL: General drafting or information without a rigid structural request.
        
        TASK:
        Return a JSON string with two fields:
        1. "intent": One of the CATEGORIES above.
        2. "directive": A 1-sentence instruction on the exact visual or structural style. 
           (Examples: "A Mermaid.js pie chart for market share", "A witty numbered list with emojis", "A Mermaid.js flow diagram of the process").
        
        USER INPUT: "{query}"
        
        JSON:
        """
        try:
            raw_response = model.generate_content(prompt).text.strip()
            # Basic cleanup in case the model adds markdown code blocks
            clean_response = re.sub(r'```json|```', '', raw_response).strip()
            result = clean_response
        except Exception:
            result = '{"intent": "FORMAT_GENERAL", "directive": ""}'

    elif name == "analyze_image":
        image_data = arguments.get("image") # URL or Base64 (starts with data:image/...)
        prompt = arguments.get("prompt", "Analyze this file and explain what is in it.")
        if not image_data: return "Error: Missing image data."
        
        model = get_flash_model()
        try:
            mime_type = "image/jpeg" # Default
            
            if image_data.startswith("http"):
                headers = {}
                if "slack.com" in image_data.lower():
                    slack_token = get_secret("slack-bot-token")
                    if slack_token:
                        headers["Authorization"] = f"Bearer {slack_token}"
                
                response = requests.get(image_data, headers=headers)
                # Dynamic MIME detection for URLs
                content_type = response.headers.get("Content-Type", "").lower()
                if "png" in content_type: mime_type = "image/png"
                elif "webp" in content_type: mime_type = "image/webp"
                elif "gif" in content_type: mime_type = "image/gif"
                elif "pdf" in content_type: mime_type = "application/pdf"
                
                part = Part.from_data(data=response.content, mime_type=mime_type)
            else:
                # Handle base64
                if "base64," in image_data:
                    header, base64_str = image_data.split("base64,")
                    header = header.lower()
                    # Extract mime from header e.g. data:image/png;base64
                    if "png" in header: mime_type = "image/png"
                    elif "webp" in header: mime_type = "image/webp"
                    elif "pdf" in header: mime_type = "application/pdf"
                    elif "jpeg" in header or "jpg" in header: mime_type = "image/jpeg"
                else:
                    base64_str = image_data
                
                decoded = base64.b64decode(base64_str)
                part = Part.from_data(data=decoded, mime_type=mime_type)
            
            response = model.generate_content([prompt, part])
            result = f"[VISUAL_INSIGHT]:\n{response.text}"
        except Exception as e:
            result = f"Error analyzing file: {str(e)}"

    elif name == "generate_image":
        prompt = arguments.get("prompt")
        if not prompt: return "Error: Missing prompt."
        
        try:
            # Nano Banana Pro logic: Image Generation
            gen_model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")
            images = gen_model.generate_images(prompt=prompt, number_of_images=1)
            
            # We need a place to store this image temporarily or get a URL.
            # For now, we'll return a placeholder or a mock URL if we don't have GCS integration ready.
            # In a production ADK, we'd upload to GCS and return the signed URL.
            # For this roadmap implementation, we'll assume a success message with the prompt.
            result = f"[IMAGE_GENERATED]: A high-fidelity visual has been created for: '{prompt}'. (GCS Link Pending Integration)"
        except Exception as e:
            result = f"Error generating image: {str(e)}"

    else:
        return f"Error: Tool '{name}' implementation logic not found."

    # Post-Processing: PII Scrubbing (Privacy Safety)
    scrubbed_result = scrub_pii(result)
    print(f"MCP Hub: Tool [{name}] execution complete. Output size: {len(str(scrubbed_result))} chars.")
    return scrubbed_result

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
    initialize_secrets()
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
