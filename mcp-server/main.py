import os
import re
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
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
from google.cloud.firestore_v1.base_query import FieldFilter
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel, HarmCategory, HarmBlockThreshold, Part
from vertexai.preview.vision_models import ImageGenerationModel
from serpapi import GoogleSearch
from googleapiclient.discovery import build
import base64
import litellm
from litellm import completion

# --- Configuration ---
PROJECT_ID = os.environ.get("PROJECT_ID")
LOCATION = os.environ.get("LOCATION", "us-central1")
SEARCH_ENGINE_ID = os.environ.get("SEARCH_ENGINE_ID")
# These will be fetched from Secret Manager in production
SERPAPI_API_KEY = None
BROWSERLESS_API_KEY = None
FLASH_MODEL_NAME = os.environ.get("FLASH_MODEL_NAME", "gemini-2.0-flash-exp")
ANTHROPIC_API_KEY = None
DEFAULT_GEO = os.environ.get("DEFAULT_GEO", "NG") # Default to Nigeria as primary operating region, but configurable

# --- SEMRush-Based Power Player Benchmarking ---
# Updated for Tiered Authority System (Step: 4500)
GLOBAL_AUTHORITY_DOMAINS = [
    "reddit.com", "wikipedia.org", "linkedin.com", "forbes.com", 
    "youtube.com", "medium.com", "nih.gov", "google.com"
]

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
    # Try Secret Manager first, fallback to Environment Variable (Critical for deployment)
    SERPAPI_API_KEY = get_secret("serpapi-api-key") or os.environ.get("SERPAPI_API_KEY")
    BROWSERLESS_API_KEY = get_secret("browserless-api-key") or os.environ.get("BROWSERLESS_API_KEY")


# --- Initialize Vertex AI ---
vertexai.init(project=PROJECT_ID, location=LOCATION)

@app.on_event("startup")
async def startup_event():
    """Initializes external services and secrets on startup."""
    global ANTHROPIC_API_KEY
    initialize_secrets()
    ANTHROPIC_API_KEY = get_secret("anthropic-api-key") or os.environ.get("ANTHROPIC_API_KEY")

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
        
        if "references" in aio:
            refs = ["**AI Overview References:**"]
            for ref in aio["references"]:
                title = ref.get("title", "N/A")
                link = ref.get("link", "N/A")
                source = ref.get("source", "N/A")
                
                # Tagging for Authority Benchmarking
                tag = ""
                if any(domain in link.lower() for domain in GLOBAL_AUTHORITY_DOMAINS):
                    # Extract the matching domain for the tag
                    matched = next((d for d in GLOBAL_AUTHORITY_DOMAINS if d in link.lower()), "High Authority")
                    tag = f" [CITATION_TIER:GLOBAL: {matched}]"
                
                refs.append(f"- {title} ({source}): {link}{tag}")
            if len(refs) > 1: extracted_features.append("\n".join(refs))

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
        organic_list = []
        for res in organic:
            title = res.get("title", "Untitled")
            snippet = res.get("snippet", "")
            link = res.get("link", "")
            
            # Tagging for Tiered Authority Benchmarking
            tag = ""
            matched_global = next((d for d in GLOBAL_AUTHORITY_DOMAINS if d in link.lower()), None)
            
            if matched_global:
                tag = f" [CITATION_TIER:GLOBAL: {matched_global}]"
            else:
                # Inverse Logic: If it's on Page 1 and NOT Global, it's a Local/Niche Authority
                # We extract the domain for clarity
                try:
                    domain = link.split("//")[-1].split("/")[0].replace("www.", "")
                    tag = f" [CITATION_TIER:LOCAL/NICHE: {domain}]"
                except:
                    tag = " [CITATION_TIER:LOCAL/NICHE]"
            
            organic_list.append(f"- {title}: {snippet} ({link}){tag}")
            
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

def extract_acronym_definitions(text):
    """
    Scans text for patterns like 'Verifiable Digital Credentials (VDC)' or 'VDC (Verifiable Digital Credentials)'.
    Returns a list of clear definitions found to ensure the Writer includes them.
    """
    definitions = []
    # Pattern 1: Full Name (Acronym) -> e.g. "World Health Organization (WHO)"
    # Constraint: Acronym must be 2-6 caps, Name must be capitalized words
    pattern1 = re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,5})\s+\(([A-Z]{2,6})\)', text)
    for name, acronym in pattern1:
        definitions.append(f"{acronym}: {name}")
        
    # Pattern 2: Acronym (Full Name) -> e.g. "WHO (World Health Organization)"
    pattern2 = re.findall(r'([A-Z]{2,6})\s+\(([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,5})\)', text)
    for acronym, name in pattern2:
        definitions.append(f"{acronym}: {name}")
        
    return list(set(definitions))

def intelligent_sample_code(content, max_chars=120000):
    """
    Preserves structural integrity of large files by keeping Head, Tail,
    and a skeleton of the middle section.
    """
    if not isinstance(content, str) or len(content) <= max_chars:
        return content
    
    print(f"MCP Hub: Sampling large file ({len(content)} chars) for intelligence preservation...")
    
    # Head: 40% (Imports, Globals, Configs)
    # Tail: 25% (Main entry points, execution logic)
    head_limit = int(max_chars * 0.4)
    tail_limit = int(max_chars * 0.25)
    
    head = content[:head_limit]
    tail = content[-tail_limit:]
    middle = content[head_limit:-tail_limit]
    
    # Extract Structural Skeleton from the Middle
    # We look for class and function signatures to maintain "Map Awareness"
    skeleton_lines = []
    lines = middle.splitlines()
    
    # Regex to catch Python class/def signatures (including async)
    # Allows for multi-line but we only take the first line for the skeleton
    structural_pattern = re.compile(r'^\s*(async\s+)?(def|class)\s+[\w\d_]+.*')
    
    for line in lines:
        if structural_pattern.match(line):
            # Clean up the line and add a marker
            skeleton_lines.append(f"  {line.strip()} ...")
            if len(skeleton_lines) > 200: # Safety cap for extremely dense files
                skeleton_lines.append("  [... further skeleton truncated for brevity ...]")
                break
    
    skeleton_text = "\n".join(skeleton_lines)
    
    final_content = (
        head + 
        f"\n\n[--- MIDDLE SECTION OMITTED ({len(middle)} chars) ---]\n" +
        f"[--- STRUCTURAL SKELETON OF OMITTED SECTION ---]\n" +
        skeleton_text + 
        f"\n\n[--- END OF SKELETON ---]\n\n" + 
        tail
    )
    
    return final_content

def handle_tool_call(name, arguments):
    """
    Central dispatcher for all sensory tools.
    Logs every call for real-time debugging and enforces validation/scrubbing.
    """
    print(f"TELEMETRY: MCP Hub: Calling Tool [{name}] | Args: {json.dumps(arguments)}")
    
    # Tool-Level Input Validation (Architectural Safety)
    if not name or not isinstance(arguments, dict):
        return "Error: Invalid tool call format."

    # Namespacing / Allowed List check
    allowed_tools = [
        "google_web_search", "scrape_article", "rag_search", "compliance_rag_search",
        "google_trends", "trend_analysis", "google_images_search", "google_videos_search", 
        "google_scholar_search", "google_news_search", "google_simple_search",
        "detect_geo", "detect_intent", "analyze_image", "generate_image",
        "google_ai_overview", "analyze_code_file"
    ]
    if name not in allowed_tools:
        return f"Error: Tool '{name}' is not in the allowed list."

    result = ""
    if name == "google_web_search":
        if not arguments.get("query"): return "Error: Missing query."
        params = {"api_key": SERPAPI_API_KEY, "engine": "google", "q": arguments.get("query"), "no_cache": True}
        results = GoogleSearch(params).get_dict()
        result = _parse_serp_features(results)
        
        # --- OMNI-TOKEN DETECTION (Ultra-Robust) ---
        # We scan everywhere for a token that could trigger AI Overview expansion
        def find_token(data):
            if not isinstance(data, dict): return None
            # 1. Check direct keys
            for key in ["next_page_token", "page_token", "token"]:
                if data.get(key): return data.get(key)
            # 2. Check if this is an AIO block
            if data.get("type") == "ai_overview" and (data.get("next_page_token") or data.get("page_token")):
                return data.get("next_page_token") or data.get("page_token")
            return None

        # Primary Candidates
        token = find_token(results.get("ai_overview")) or find_token(results)
        
        # Secondary Candidates: Related Questions of type ai_overview
        if not token and "related_questions" in results:
            for q in results["related_questions"]:
                token = find_token(q)
                if token: break

        if token:
            print(f"MCP Hub: [DOUBLE-TAP] Expansion token found: {token[:15]}...", flush=True)
            aio_params = {"api_key": SERPAPI_API_KEY, "engine": "google_ai_overview", "page_token": token}
            aio_results = GoogleSearch(aio_params).get_dict()
            
            # Check if expansion results contains actual AIO content
            # Often SerpAPI puts the expanded AIO at the root or under 'ai_overview'
            expanded_context = _parse_serp_features(aio_results)
            print(f"MCP Hub: [DOUBLE-TAP] Expansion length: {len(expanded_context)} chars.", flush=True)
            
            if "[GROUNDING_CONTENT]" in expanded_context:
                clean_expanded = expanded_context.replace("[GROUNDING_CONTENT]\n", "")
                result += "\n\n--- EXPANDED AI INSIGHTS ---\n" + clean_expanded

    elif name == "scrape_article":
        url = arguments.get("url")
        if not url or not url.startswith("http"): return "Error: Invalid URL."
        
        # --- HYBRID DETECTION: Check if PDF or HTML ---
        is_pdf = url.lower().endswith(".pdf")
        
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            # Initial head request to check content-type if not obviously .pdf
            if not is_pdf:
                head_resp = requests.head(url, headers=headers, timeout=10, allow_redirects=True)
                content_type = head_resp.headers.get("Content-Type", "").lower()
                if "application/pdf" in content_type:
                    is_pdf = True

            if is_pdf:
                print(f"MCP Hub: High-Fidelity PDF Detection. Using Gemini for {url}")
                pdf_resp = requests.get(url, headers=headers, timeout=30, verify=certifi.where())
                pdf_resp.raise_for_status()
                
                model = get_flash_model() # Uses gemini-2.0-flash-exp (or whichever is configured)
                pdf_part = Part.from_data(data=pdf_resp.content, mime_type="application/pdf")
                
                # Optimized prompt for regulatory and technical extraction
                extraction_prompt = """
                Extract all text from this document verbatim. 
                Keep the structure (Headings, Articles, Sections).
                Ensure Article numbers and specific legal clauses are perfectly captured.
                Format as clean Markdown.
                """
                
                response = model.generate_content([extraction_prompt, pdf_part])
                result = f"[HIGH_FIDELITY_PDF_EXTRACTION]\n\n{response.text}"
                
            else:
                # Standard HTML Scrape via Browserless
                endpoint = f"https://production-sfo.browserless.io/content?token={BROWSERLESS_API_KEY}&stealth=true"
                payload = {"url": url, "rejectResourceTypes": ["media", "font"], "gotoOptions": {"timeout": 15000, "waitUntil": "networkidle2"}}
                response = requests.post(endpoint, json=payload, timeout=20, verify=certifi.where())
                soup = BeautifulSoup(response.content, 'html.parser')
                
                for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "iframe", "ad"]): tag.decompose()

                found_images = []
                exclude_patterns = ["profile_images", "avatar", "icon", "logo", "header_photo", "normal", "mini"]
                img_ext_pattern = re.compile(r'\.(jpg|jpeg|png|webp|gif|pdf)', re.I)
                social_media_fmt_pattern = re.compile(r'format=(jpg|jpeg|png|webp|gif)', re.I)

                for img in soup.find_all("img"):
                    img_src = img.get("src") or img.get("data-src") or img.get("srcset") or img.get("data-original")
                    if img_src and img_src.startswith("http"):
                        clean_src = img_src.split(',')[0].split(' ')[0].strip()
                        has_valid_ext = img_ext_pattern.search(clean_src) or social_media_fmt_pattern.search(clean_src)
                        if not has_valid_ext: continue
                        if any(p in clean_src.lower() for p in exclude_patterns):
                            if "twimg.com/media" not in clean_src.lower(): continue
                        if clean_src not in found_images:
                            found_images.append(clean_src)
                    if len(found_images) >= 5: break

                found_links = []
                skip_domains = ["twitter.com", "x.com", "facebook.com", "instagram.com", "linkedin.com", "youtube.com"]
                for link in soup.find_all("a", href=True):
                    href = link.get("href", "").strip()
                    text = link.get_text().strip()
                    if not href.startswith("http"): continue
                    if len(text) < 5: continue
                    if any(d in href for d in skip_domains): continue
                    entry = f"- {text} ({href})"
                    if entry not in found_links: found_links.append(entry)
                    if len(found_links) >= 8: break

                text = soup.get_text(separator='\n\n')
                clean_text = re.sub(r'\n\s*\n', '\n\n', text).strip()
                image_context = "\n\n[DETECTED_IMAGES]:\n" + "\n".join(found_images) if found_images else ""
                link_context = "\n\n[DETECTED_LINKS]:\n" + "\n".join(found_links) if found_links else ""
                acronym_defs = extract_acronym_definitions(clean_text)
                def_context = "\n\n[DETECTED_DEFINITIONS]:\n" + "\n".join(acronym_defs) if acronym_defs else ""
                
                result = (def_context + "\n\n" + clean_text[:18000] + image_context + link_context).strip()

        except Exception as e:
            result = f"Error in hybrid scrape: {str(e)}"

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

    elif name == "compliance_rag_search":
        query = arguments.get("query")
        doc_type = arguments.get("doc_type")
        geo_scope = arguments.get("geo_scope")
        limit = arguments.get("limit", 5)
        
        if not query: return "Error: Missing query."
        
        try:
            db = get_db()
            embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
            query_embedding = embedding_model.get_embeddings([query[:6000]])[0].values
            
            collection = db.collection('compliance_knowledge')
            
            # Use modern FieldFilter for robust query construction
            filters = [FieldFilter("tenant_id", "==", "global")]
            
            if doc_type:
                filters.append(FieldFilter("doc_type", "==", doc_type))
            if geo_scope:
                filters.append(FieldFilter("geo_scope", "array_contains", geo_scope))
                
            # Filtered Vector Search
            try:
                # We combine filters into a single query
                query_ref = collection
                for f in filters:
                    query_ref = query_ref.where(filter=f)
                    
                results = query_ref.find_nearest(
                    vector_field="embedding",
                    query_vector=Vector(query_embedding),
                    distance_measure=DistanceMeasure.COSINE,
                    limit=limit
                ).get()
            except Exception as e:
                print(f"WARNING: Filtered vector search failed: {e}. Falling back to Local Semantic Search.")
                # Fallback: Fetch chunks and perform local match
                # This ensures we work even if indexes are building
                local_query = collection.where(filter=FieldFilter("tenant_id", "==", "global"))
                if doc_type: local_query = local_query.where(filter=FieldFilter("doc_type", "==", doc_type))
                if geo_scope: local_query = local_query.where(filter=FieldFilter("geo_scope", "array_contains", geo_scope))
                
                # Fetch up to 100 relevant chunks for local processing
                all_docs = local_query.limit(100).get()
                
                # Simple keyword match + relevance boost
                matched_docs = []
                query_words = set(query.lower().split())
                for doc in all_docs:
                    d = doc.to_dict()
                    content = d.get('content', '').lower()
                    score = sum(1 for word in query_words if word in content)
                    if score > 0:
                        matched_docs.append((score, d))
                
                # Sort by score descending
                matched_docs.sort(key=lambda x: x[0], reverse=True)
                results_data = [d for score, d in matched_docs[:limit]]
                
                if not results_data:
                    return "No relevant compliance regulations found (Local Fallback)."
                
                snippets = []
                for d in results_data:
                    snippet = f"[{d.get('doc_type', 'UNKNOWN').upper()} - {d.get('doc_version', '')}]\nSource: {d.get('doc_source', '')}\nContent: {d.get('content', '')}"
                    snippets.append(snippet)
                return "\n\n---\n\n".join(snippets)
                
            if not results:
                return "No relevant compliance regulations found for this query."
                
            snippets = []
            for doc in results:
                d = doc.to_dict()
                snippet = f"[{d.get('doc_type', 'UNKNOWN').upper()} - {d.get('doc_version', '')}]\nSource: {d.get('doc_source', '')}\nContent: {d.get('content', '')}"
                snippets.append(snippet)
                
            result = "\n\n---\n\n".join(snippets)
        except Exception as e:
            result = f"Error in compliance RAG: {str(e)}"

    elif name == "google_trends":
        geo = arguments.get("geo", DEFAULT_GEO)
        params = {"api_key": SERPAPI_API_KEY, "engine": "google_trends_trending_now", "geo": geo, "hours": "24"}
        results = GoogleSearch(params).get_dict()
        
        trending_searches = results.get("trending_searches", [])
        if not trending_searches:
            result = "No trending searches found for this region."
        else:
            # Dynamic Grouping by Category (No hardcoding)
            categorized_trends = {}
            for trend in trending_searches:
                categories = trend.get("categories", [])
                cat_name = "General"
                if categories and isinstance(categories, list):
                    cat_name = categories[0].get("name", "General") if isinstance(categories[0], dict) else "General"
                
                if cat_name not in categorized_trends:
                    categorized_trends[cat_name] = []
                
                # Semantic extraction (Stripping tokens/bloat for high signal context)
                q = trend.get('query', 'Unknown')
                vol = trend.get('search_volume', 'N/A')
                trend_entry = f"- {q} (Vol: {vol})"
                
                breakdown = trend.get("trend_breakdown", [])
                if breakdown:
                    trend_entry += f" | Context: {', '.join(breakdown[:3])}"
                
                categorized_trends[cat_name].append(trend_entry)
            
            # Construct refined semantic payload
            output_parts = [f"### GOOGLE TRENDS SUMMARY: {geo} ###"]
            for cat, items in categorized_trends.items():
                output_parts.append(f"\n[Category: {cat}]")
                output_parts.extend(items)
            
            result = "\n".join(output_parts)

    elif name == "trend_analysis":
        query = arguments.get("query")
        if not query: return "Error: Missing query."
        geo = arguments.get("geo", DEFAULT_GEO)
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
        geo = arguments.get("geo", DEFAULT_GEO)
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

    elif name == "google_ai_overview":
        token = arguments.get("page_token") or arguments.get("next_page_token")
        if not token: return "Error: Missing page_token or next_page_token."
        params = {"api_key": SERPAPI_API_KEY, "engine": "google_ai_overview", "page_token": token}
        results = GoogleSearch(params).get_dict()
        result = _parse_serp_features(results)

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

    elif name == "analyze_code_file":
        file_content = arguments.get("file_content")
        file_name = arguments.get("file_name", "unknown_file")
        user_prompt = arguments.get("user_prompt", "")
        history_context = arguments.get("history_context", "")

        # CONTENT PROTECTION: If content looks like a Slack error or is empty, abort early
        if not file_content or '{"ok":false' in str(file_content) or '"error":"' in str(file_content)[:100]:
            print(f"MCP Hub: Aborting analysis for {file_name} due to invalid content (Slack Error or Empty)")
            return f"CODE_FILE ({file_name}): [Content unavailable or download error from Slack]"
        
        print(f"MCP Hub: Analyzing code file: {file_name}")
        
        # INTELLIGENT SAMPLING: Preserve structural integrity instead of simple truncation
        file_content = intelligent_sample_code(file_content, max_chars=120000)
        print(f"MCP Hub: Prepared analysis content for {file_name} ({len(file_content)} chars)")
        
        analysis_prompt = f"""
        You are a technical code analyst. Analyze this code file to extract insights that will help answer the user's request.
        
        **Code File**: `{file_name}`
        **User Request**: {user_prompt or "Analyze this code"}
        
        **Code Content**:
        ```
        {file_content}
        ```
        
        **Conversation History** (for context):
        {history_context[:2000] if history_context else "No prior context"}
        
        **Analysis Task**: 
        Provide a structured analysis covering:
        
        1. **Code Summary** (2-3 sentences)
           - What does this code do?
           - What problem does it solve?
        
        2. **Key Technical Concepts**
           - Technologies/frameworks used
           - Design patterns employed
           - Architecture decisions
        
        3. **Notable Features**
           - Interesting implementations
           - Performance optimizations
           - Security considerations
           - Error handling approaches
        
        4. **Potential Content Angles** (based on user's request)
           - What aspects align with the user's question?
           - What technical insights would be valuable?
           - What could be explained or taught?
        
        5. **Code Quality Observations**
           - Strengths of the implementation
           - Areas that could be improved
           - Best practices demonstrated
        
        **CRITICAL**: 
        - Write in clear, technical prose (NOT JSON)
        - Focus on insights relevant to the user's request
        - This analysis will be used as grounding context for content generation
        - For any code snippets you include in the analysis, use markdown fenced code blocks (```language).
        - Be specific and cite actual code patterns you observe
        """
        
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type(Exception),
            reraise=True
        )
        def generate_with_retry():
            # specialist model swap: Switch from Gemini Flash to Claude Sonnet 4.5
            print(f"MCP Hub: Calling Specialist Model (Claude Sonnet 4.5) for {file_name}...")
            
            # Inject key for LiteLLM
            if ANTHROPIC_API_KEY:
                os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY
                
            response = completion(
                model="anthropic/claude-sonnet-4-5",
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.3,
                max_tokens=4096
            )
            return response.choices[0].message.content

        try:
            analysis_text = generate_with_retry()
            print(f"MCP Hub: Code analysis complete for {file_name}")
            result = f"CODE_ANALYSIS ({file_name}):\n{analysis_text}"
        except Exception as e:
            print(f"MCP Hub: Code analysis failed after retries: {e}")
            result = f"CODE_FILE ({file_name}):\n{file_content[:2000]}\n\n[Analysis failed after retries, using raw code]"

    else:
        return f"Error: Tool '{name}' implementation logic not found."

    # Post-Processing: PII Scrubbing (Privacy Safety)
    scrubbed_result = scrub_pii(result)
    
    # 5. Output Telemetry (Precision Observability)
    print(f"TELEMETRY: MCP Hub: [{name}] execution complete. Size: {len(str(scrubbed_result))} chars.")
    if scrubbed_result and len(str(scrubbed_result)) > 0:
        snippet = str(scrubbed_result)[:200].replace('\n', ' ')
        print(f"  -> Snippet: {snippet}...")
        
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
    print(f"TELEMETRY: MCP Hub: Received '{method}' request from Worker.")
    
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
