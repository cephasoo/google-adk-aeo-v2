# --- /worker-story/main.py ---
import functions_framework
from flask import jsonify
import vertexai
from vertexai.generative_models import GenerativeModel
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
import datetime

# --- Configuration ---
PROJECT_ID = os.environ.get("PROJECT_ID")
LOCATION = os.environ.get("LOCATION")
MODEL_NAME = os.environ.get("MODEL_NAME")
FLASH_MODEL_NAME = os.environ.get("FLASH_MODEL_NAME")
SEARCH_ENGINE_ID = os.environ.get("SEARCH_ENGINE_ID")
BROWSERLESS_API_KEY = os.environ.get("BROWSERLESS_API_KEY")
SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY")
N8N_PROPOSAL_WEBHOOK_URL = os.environ.get("N8N_PROPOSAL_WEBHOOK_URL") 
MAX_LOOP_ITERATIONS = 2

# --- Global Clients ---
model = None
flash_model = None
search_api_key = None
db = None

# --- Utils ---
# --- Input Santitation Utility ---
def sanitize_input_text(text):
    """Strips common markdown/Slack formatting from the start and end of a string."""
    if not isinstance(text, str):
        return ""
    # Strips leading/trailing whitespace, asterisks (bold), underscores (italics), and tildes (strikethrough)
    return text.strip().strip('*_~')

def extract_core_topic(user_prompt):
    """
    Uses a fast LLM to distill a user's natural language prompt 
    into a rich, effective search query.
    """
    global flash_model
    print(f"Distilling core topic from: '{user_prompt[:100]}...'")
    
    extraction_prompt = f"""
    You are an expert Google Search operator.
    Convert the user's natural language request into the SINGLE BEST Google Search query to find high-quality results.

    Guidelines:
    1.  **Keep it Rich:** Do not over-simplify. If the user compares X and Y, search for "X vs Y".
    2.  **Use Operators:** Use quotes "" for exact phrases, OR for alternatives, and AND for required concepts.
    3.  **Remove Fluff:** Remove "I want to know about", "Please tell me", etc.
    
    Examples:
    *   User: "What are the latest trends in SEO?" -> Query: "SEO trends 2025"
    *   User: "Compare LangChain and AutoGen" -> Query: "LangChain vs AutoGen features"
    *   User: "Find PDF reports on AI agents" -> Query: "AI agents filetype:pdf"

    USER REQUEST:
    "{user_prompt}"

    SEARCH QUERY:
    """
    
    core_topic = flash_model.generate_content(extraction_prompt).text.strip()

    print(f"Distilled Core Topic: '{core_topic}'")
    return core_topic

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

def search_long_term_memory(query):
    global db
    if db is None: db = firestore.Client()
    
    print(f"Tool: Accessing Hippocampus (Vector Search) for: '{query}'")
    
    # 1. Embed
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
    query_embedding = embedding_model.get_embeddings([query])[0].values
    
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

# 1. The Helper to Parse SerpApi Features
def _parse_serp_features(results):
    """
    A "Smart Extractor" that parses the complex SerpApi JSON payload 
    into a clean, LLM-friendly Markdown format.
    """
    extracted_features = []

    # --- 1. NEW: Top-Level AI Overview (The Hydrated Data) ---
    if "ai_overview" in results:
        aio = results["ai_overview"]
        if "text_blocks" in aio:
            aio_parts = ["**Google AI Overview:**"]
            for block in aio["text_blocks"]:
                if block.get("type") == "heading":
                    aio_parts.append(f"*{block.get('snippet')}*")
                elif block.get("type") == "paragraph":
                    aio_parts.append(block.get('snippet'))
                elif block.get("type") == "list" and "list" in block:
                    list_items = [f"  - {item.get('snippet')}" for item in block["list"]]
                    aio_parts.append("\n".join(list_items))
            
            if len(aio_parts) > 1: # Ensure we found actual text
                extracted_features.append("\n\n".join(aio_parts))

    # 2. Knowledge Graph
    if "knowledge_graph" in results:
        kg = results["knowledge_graph"]
        title = kg.get("title", "Knowledge Graph")
        if "see_results_about" in kg and kg["see_results_about"]:
             first_item = kg["see_results_about"][0]
             name = first_item.get("name", "N/A")
             description = ", ".join(first_item.get("extensions", []))
             extracted_features.append(f"**Knowledge Graph:**\n- **Entity:** {name}\n- **Description:** {description}")
        else:
             description = kg.get("description", "No description available.")
             extracted_features.append(f"**Knowledge Graph: {title}**\n- {description}")

    # 3. Top Stories
    if "top_stories" in results:
        stories = results["top_stories"]
        story_list = [f"- {story.get('title', 'Untitled')} ({story.get('source', 'Unknown Source')})" for story in stories]
        if story_list:
            extracted_features.append("**Top Stories:**\n" + "\n".join(story_list))

    # 4. Related Questions
    if "related_questions" in results:
        questions = results["related_questions"]
        qa_list = []
        for question in questions:
            q_text = question.get("question")
            q_snippet = ""
            # Handle nested AI overview inside PAA (Rare but possible)
            if question.get("type") == "ai_overview" and "text_blocks" in question:
                q_snippet_parts = []
                for block in question["text_blocks"]:
                    if block.get("type") == "heading": q_snippet_parts.append(f"\n*{block.get('snippet')}*")
                    elif block.get("type") == "paragraph": q_snippet_parts.append(block.get('snippet'))
                    elif block.get("type") == "list" and "list" in block:
                        list_items = [f"  - {item.get('snippet')}" for item in block["list"]]
                        q_snippet_parts.append("\n".join(list_items))
                q_snippet = "\n".join(q_snippet_parts)
            else:
                q_snippet = question.get("snippet", "No answer snippet found.")
            
            if q_text and q_snippet:
                qa_list.append(f"- **Q:** {q_text}\n  - **A:** {q_snippet.strip()}")
        
        if qa_list:
            extracted_features.append("**Related Questions (People Also Ask):**\n" + "\n".join(qa_list))

    # 5. Fallback: Organic Results
    if not extracted_features and "organic_results" in results:
        organic = results["organic_results"][:3]
        organic_list = [f"- {res.get('title', 'Untitled')}: {res.get('snippet', '')} ({res.get('link')})" for res in organic]
        if organic_list:
            extracted_features.append("**Web Results:**\n" + "\n".join(organic_list))

    return "\n\n---\n\n".join(extracted_features) if extracted_features else None

# 2. The Specialist Web Scraper Tool
def search_google_web(query):
    """
    Specialist tool for deep analysis of the main web SERP.
    Includes robust logic to 'hydrate' lazy-loaded AI Overviews.
    """
    print(f"Tool: Executing WEB search for: '{query}'")
    try:
        from serpapi import GoogleSearch
        
        # 1. Primary Search (The Standard 'google' Engine)
        params = {"api_key": SERPAPI_API_KEY, "engine": "google", "q": query}
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
                     "q": query 
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


def search_google_images(query, num_results=5):
    """Specialist tool for analyzing image search results."""
    print(f"Tool: Executing IMAGE search for: '{query}'")
    try:
        from serpapi import GoogleSearch
        params = {"api_key": SERPAPI_API_KEY, "engine": "google_images", "q": query}
        results = GoogleSearch(params).get_dict().get("images_results", [])[:num_results]
        if not results: return None
        image_descriptions = [f"- Title: {img.get('title', 'N/A')}, Source: {img.get('source', 'N/A')}" for img in results]
        return "**Image Search Results (Visual Trends):**\n" + "\n".join(image_descriptions)
    except Exception as e:
        print(f"SerpApi Image Search failed: {e}")
        return None

def search_google_videos(query, num_results=5):
    """Specialist tool for analyzing video search results."""
    print(f"Tool: Executing VIDEO search for: '{query}'")
    try:
        from serpapi import GoogleSearch
        params = {"api_key": SERPAPI_API_KEY, "engine": "google_videos", "q": query}
        results = GoogleSearch(params).get_dict().get("video_results", [])[:num_results]
        if not results: return None
        video_descriptions = [f"- Title: {vid.get('title', 'N/A')}, Source: {vid.get('source', 'N/A')}, Length: {vid.get('duration', 'N/A')}" for vid in results]
        return "**Video Search Results (Top Videos):**\n" + "\n".join(video_descriptions)
    except Exception as e:
        print(f"SerpApi Video Search failed: {e}")
        return None

def search_google_scholar(query, num_results=3):
    """Specialist tool for finding academic papers and research."""
    print(f"Tool: Executing SCHOLAR search for: '{query}'")
    try:
        from serpapi import GoogleSearch
        params = {"api_key": SERPAPI_API_KEY, "engine": "google_scholar", "q": query}
        results = GoogleSearch(params).get_dict().get("organic_results", [])[:num_results]
        if not results: return None
        scholar_summaries = [f"- Title: {res.get('title')}\n  - Publication: {res.get('publication_info', {}).get('summary')}\n  - Snippet: {res.get('snippet', 'N/A')}" for res in results]
        return "**Scholarly Articles (Academic Research):**\n" + "\n".join(scholar_summaries)
    except Exception as e:
        print(f"SerpApi Scholar Search failed: {e}")
        return None

# 3. The Router (Updated to use the new names)

def find_trending_keywords(clean_topic, history_context=""):
    """
    An intelligent meta-tool that routes a query to the best specialized search tool.
    Now includes a 'NONE' option to prevent redundant searches.
    """
    global model, flash_model
    print(f"Tool: find_trending_keywords (Sensory Array) for clean_topic: '{clean_topic}'")
    
    tool_logs = []
    context_snippets = []
    
    # Internal RAG (Unchanged)
    internal_context = search_long_term_memory(clean_topic)
    if internal_context:
        context_snippets.append(internal_context)
        tool_logs.append({"event_type": "tool_call", "tool_name": "internal_knowledge_retrieval", "status": "success"})

    # Sensory Array Router
    tool_choice_prompt = f"""
    Analyze the user's query to select the single best research tool.

    CONVERSATION HISTORY:
    {history_context}

    USER'S CORE QUERY: '{clean_topic}'

    Available Tools:
    - WEB: For deep analysis of web results, knowledge graphs, AI Overviews, and top stories.
    - IMAGES: If the user explicitly asks for images, pictures, or visual inspiration.
    - VIDEOS: If the user explicitly asks for videos, clips, or tutorials.
    - SCHOLAR: If the user is asking for academic papers, research, studies, or highly credible sources.
    - SIMPLE: For all other general web searches.
    - NONE: Select this if the answer can be derived ENTIRELY from the CONVERSATION HISTORY (e.g., "Review the previous results", "Summarize what you just found"). Do not waste budget on new searches.

    Which tool is most appropriate? Respond with ONLY the tool name.
    """
    choice = flash_model.generate_content(tool_choice_prompt).text.strip().upper()
    print(f"Sensory Array Router chose tool: {choice}")

    # --- 3. OPTIMIZATION: HANDLE "NONE" ---
    if "NONE" in choice:
        print("Router decided context is sufficient. Skipping external search.")
        # Return whatever internal memory we found, or just an empty context if strictly chat
        return {
            "context": context_snippets, 
            "tool_logs": tool_logs 
        }
    
    research_context = None
    tool_name = "unknown"

    if choice == "WEB":
        research_context = search_google_web(clean_topic) # Replaces get_structured_serp_analysis
        tool_name = "serpapi_web_search"
    elif choice == "IMAGES":
        research_context = search_google_images(clean_topic)
        tool_name = "serpapi_image_search"
    elif choice == "VIDEOS":
        research_context = search_google_videos(clean_topic)
        tool_name = "serpapi_video_search"
    elif choice == "SCHOLAR":
        research_context = search_google_scholar(clean_topic)
        tool_name = "serpapi_scholar_search"
    
    if research_context:
        context_snippets.append(research_context)
        tool_logs.append({"event_type": "tool_call", "tool_name": tool_name, "input": clean_topic, "status": "success", "content": research_context})
    
    # Fallback to Simple Search
    if not research_context and "NONE" not in choice:
        print("Executing simple web search as fallback...")
        try:
            api_key = get_search_api_key()
            service = build("customsearch", "v1", developerKey=api_key)
            tool_logs.append({"event_type": "tool_call", "tool_name": "google_simple_search", "input": clean_topic})
            res = service.cse().list(q=clean_topic, cx=SEARCH_ENGINE_ID, num=5).execute()
            google_snippets = [item.get('snippet', '') for item in res.get('items', [])]
            if google_snippets:
                context_snippets.append("[CONTEXT FROM GOOGLE WEB SEARCH]:\n---\n" + "\n---\n".join(google_snippets))
        except Exception as e:
            print(f"Simple Google search failed: {e}")

    return {
        "context": context_snippets, 
        "tool_logs": tool_logs 
    }

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

def critique_proposal(topic, current_proposal):
    global model
    prompt = f"Review proposal for '{topic}': {json.dumps(current_proposal, indent=2)}. If excellent, respond: APPROVED. Else, provide concise feedback to improve 'Then vs Now' contrast."
    return model.generate_content(prompt).text.strip()

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
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        model = GenerativeModel(MODEL_NAME)
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
        history_events = session_doc.to_dict().get('event_log', [])


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
        triage_prompt = f"""
        Analyze the user's latest message in the context of the conversation history.
        Classify the PRIMARY GOAL into one of four categories:

        1.  **SOCIAL_CONVERSATION**: The user is engaging in small talk, saying hello/goodbye, expressing gratitude, or making a comment that DOES NOT require external information or research.
            *Examples:* "Thanks!", "Great job.", "I have to go now.", "You are funny.", "Okay.", "Good Morning."

        2.  **TOPIC_CLUSTER_PROPOSAL**: The user wants you to GENERATE a NEW topic cluster, pillar page, or content strategy FROM SCRATCH.
            *   **Select this ONLY if the user has NOT provided a pre-existing list or cluster.**

        3.  **THEN_VS_NOW_PROPOSAL**: The user EXPLICITLY asks for a "Then vs Now" story, draft, or structured comparison.

        4.  **DIRECT_ANSWER**: This is the default for KNOWLEDGE TASKS. The user is asking a question, asking for a summary, or giving a command that requires processing information.
            *Examples:* "Summarize this URL", "Who is the minister?", "Create a strategy for this cluster."

        CONVERSATION HISTORY:
        {history_text}

        USER REQUEST: "{sanitized_topic}"

        CRITICAL: Respond with ONLY the category name.
        """
        intent = flash_model.generate_content(triage_prompt).text.strip()
        print(f"Smart Triage V5 classified intent as: {intent}")

        # Initialize variables for the response
        new_events = [{"event_type": "user_request", "text": original_topic, "timestamp": str(datetime.datetime.now())}]
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
            
            db.collection('agent_sessions').document(session_id).update({
                "status": "completed", # Social usually ends a "turn"
                "type": "social",
                "event_log": firestore.ArrayUnion(new_events),
                "last_updated": expire_time
            })
            requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={"session_id": session_id, "type": "social", "message": reply_text, "channel_id": slack_context['channel'], "thread_ts": slack_context['ts']}, verify=True)
            return jsonify({"msg": "Social reply sent"}), 200


        # === PATH B: WORK/RESEARCH (The Heavy Lifting) ===
        # Only now do we pay the cost to distill and research.
        
        # 1. Distill
        clean_topic = extract_core_topic(sanitized_topic)

        # 2. Research (Handle URL or Keywords)
        url = extract_url_from_text(sanitized_topic)
        if url:
            article_text = fetch_article_content(url)
            research_data = {"context": [f"GROUNDING_CONTENT:\n{article_text}"], "tool_logs": []}
        else:
            research_data = find_trending_keywords(clean_topic, history_context=history_text)
        
        if "tool_logs" in research_data: new_events.extend(research_data["tool_logs"])

        # 3. Generate Output based on Intent
        if intent == "DIRECT_ANSWER":
            answer_text = generate_comprehensive_answer(original_topic, research_data['context'], history=history_text)
            new_events.append({"event_type": "agent_answer", "text": answer_text})
            
            db.collection('agent_sessions').document(session_id).update({
                "status": "awaiting_feedback", "type": "work_answer", "topic": clean_topic,
                "slack_context": slack_context, "event_log": firestore.ArrayUnion(new_events),
                "last_updated": expire_time
            })
            requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={"session_id": session_id, "type": "social", "message": answer_text, "channel_id": slack_context['channel'], "thread_ts": slack_context['ts']}, verify=True)
            return jsonify({"msg": "Answer sent"}), 200

        elif intent == "TOPIC_CLUSTER_PROPOSAL":
            cluster_data = generate_topic_cluster(clean_topic, research_data['context'], history=history_text)
            formatted_cluster = f"Here is the topic cluster you requested:\n```\n{json.dumps(cluster_data, indent=2)}\n```"
            new_events.append({"event_type": "agent_answer", "proposal_type": "topic_cluster", "data": cluster_data})
            
            db.collection('agent_sessions').document(session_id).update({
                "status": "awaiting_feedback", "type": "work_proposal", "topic": clean_topic,
                "slack_context": slack_context, "event_log": firestore.ArrayUnion(new_events),
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
                
                db.collection('agent_sessions').document(session_id).update({
                    "status": "awaiting_approval", "type": "work_proposal", "topic": clean_topic,
                    "slack_context": slack_context, "event_log": firestore.ArrayUnion(new_events),
                    "last_updated": expire_time
                })
                requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={"session_id": session_id, "approval_id": approval_id, "proposal": current_proposal['interlinked_concepts'], "thread_ts": slack_context['ts'], "channel_id": slack_context['channel'], "is_initial_post": True }, verify=True)
                return jsonify({"msg": "Proposal sent"}), 200

            except ValueError as e:
                # Fallback
                answer_text = generate_comprehensive_answer(original_topic, research_data['context'], history=history_text)
                new_events.append({"event_type": "agent_answer", "text": answer_text})
                db.collection('agent_sessions').document(session_id).update({
                    "status": "awaiting_feedback", "type": "work_answer", "topic": clean_topic,
                    "slack_context": slack_context, "event_log": firestore.ArrayUnion(new_events),
                    "last_updated": expire_time
                })
                requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={"session_id": session_id, "type": "social", "message": answer_text, "channel_id": slack_context['channel'], "thread_ts": slack_context['ts']}, verify=True)
                return jsonify({"msg": "Fallback answer sent"}), 200

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
    
    batch = db.batch()
    
    # FIX: Renamed 'chunk_text' to 'text_segment' to avoid collision
    for i, (text_segment, embedding_obj) in enumerate(zip(chunks, embeddings)):
        doc_ref = db.collection('knowledge_base').document(f"{session_id}_{i}")
        batch.set(doc_ref, {
            "content": text_segment, # Update reference here too
            "embedding": embedding_obj.values,
            "topic_trigger": topic, 
            "source_session": session_id,
            "chunk_index": i,
            "created_at": datetime.datetime.now(datetime.timezone.utc)
        })
    batch.commit()
    print(f"Ingested {len(chunks)} chunks.")
    return jsonify({"msg": "Knowledge ingested."}), 200