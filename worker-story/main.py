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
SEARCH_ENGINE_ID = os.environ.get("SEARCH_ENGINE_ID")
BROWSERLESS_API_KEY = os.environ.get("BROWSERLESS_API_KEY")
N8N_PROPOSAL_WEBHOOK_URL = os.environ.get("N8N_PROPOSAL_WEBHOOK_URL") 
MAX_LOOP_ITERATIONS = 2

# --- Global Clients ---
model = None
flash_model = None
search_api_key = None
db = None

# --- Utils ---
def get_search_api_key():
    global search_api_key
    if search_api_key is None:
        client = secretmanager.SecretManagerServiceClient()
        response = client.access_secret_version(name=f"projects/{PROJECT_ID}/secrets/google-search-api-key/versions/latest")
        search_api_key = response.payload.data.decode("UTF-8")
    return search_api_key

def extract_json(text):
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match: raise ValueError(f"No JSON found in text: {text[:100]}...")
    return json.loads(match.group(0))

def extract_url_from_text(text):
    url_pattern = r'(https?://[^\s<>|]+)'
    match = re.search(url_pattern, text)
    if match: return match.group(1)
    return None

def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            natural_break = text.rfind('\n', start, end)
            if natural_break == -1: natural_break = text.rfind('.', start, end)
            if natural_break != -1 and natural_break > start + (chunk_size // 2): end = natural_break + 1
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

# In worker-story/main.py

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

def find_trending_keywords(unstructured_topic, history_context=""):
    global model
    print(f"Tool: find_trending_keywords for input: '{unstructured_topic[:100]}...'")
    tool_logs = []
    url = extract_url_from_text(unstructured_topic)
    context_snippets = []
    clean_topic = ""
    is_grounded = False
    
    if url:
        print(f"URL Detected: {url}. Switching to Grounding Mode.")
        tool_logs.append({"event_type": "tool_call", "tool_name": "browserless_scraper", "input": url, "status": "attempting", "timestamp": str(datetime.datetime.now())})
        article_text = fetch_article_content(url)
        status = "success" if "Error" not in article_text else "error"
        tool_logs.append({"event_type": "tool_result", "tool_name": "browserless_scraper", "output_summary": f"Retrieved {len(article_text)} characters", "status": status, "timestamp": str(datetime.datetime.now())})
        context_snippets = [f"GROUNDING_SOURCE_URL: {url}", f"GROUNDING_CONTENT:\n{article_text}"]
        clean_topic = unstructured_topic
        is_grounded = True
    else:
        extraction_prompt = f"User Input: '{unstructured_topic}'... Task: Convert to Google Search Query... Respond with ONLY the search query string."
        search_query = model.generate_content(extraction_prompt).text.strip()
        api_key = get_search_api_key()
        service = build("customsearch", "v1", developerKey=api_key)
        tool_logs.append({"event_type": "tool_call", "tool_name": "google_search", "input": search_query, "timestamp": str(datetime.datetime.now())})
        res = service.cse().list(q=f"{search_query}", cx=SEARCH_ENGINE_ID, num=5).execute()
        google_snippets = [r.get('snippet', '') for r in res.get('items', [])]
        context_snippets = google_snippets
        clean_topic = search_query

    # RAG Retrieval (Hybrid)
    past_memories = search_long_term_memory(clean_topic)
    if past_memories:
        print(f"RAG: Found {len(past_memories)} relevant past experiences.")
        joined_memories = "\n---\n".join(past_memories)
        context_snippets.append(f"INTERNAL KNOWLEDGE BASE:\n{joined_memories}")

    return {"context": context_snippets, "clean_topic": clean_topic, "is_grounded": is_grounded, "tool_logs": tool_logs}

def generate_topic_cluster(topic, context):
    global model
    print("Tool: Topic Cluster Generator")
    prompt = f"""
    You are a master Content Strategist. User wants pillar page on "{topic}".
    CONTEXT: {context}
    CRITICAL: Respond with a JSON object following this exact schema:
    {{
      "pillar_page_title": "The Definitive Guide to [Topic]",
      "clusters": [
        {{ "cluster_title": "...", "sub_topics": ["...", "..."] }}
      ]
    }}
    """
    response = model.generate_content(prompt)
    return extract_json(response.text)

def generate_comprehensive_answer(topic, context):
    global model
    is_grounded = any("GROUNDING_CONTENT" in str(c) for c in context)
    if is_grounded:
        temperature = 0.0
        system_instruction = "CRITICAL INSTRUCTION: You are in READING MODE. Base answer PRIMARILY on 'GROUNDING_CONTENT'."
    else:
        temperature = 0.7
        system_instruction = "You are an expert AI assistant. Use the research context to provide accurate answers."
    prompt = f"{system_instruction}\n\nThe user asked: \"{topic}\"\n\nResearch Context:\n{context}\n\nAnswer:"
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

# --- The Main Worker Function ---
@functions_framework.http
def process_story_logic(request):
    global model, flash_model, db
    if model is None: 
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        model = GenerativeModel(MODEL_NAME) # Pro Model
        flash_model = GenerativeModel("gemini-2.5-flash-lite") # Flash Model
        db = firestore.Client()

    req = request.get_json(silent=True)
    session_id = req['session_id']
    topic = req['topic']
    slack_context = req['slack_context']

    try:
        # --- REFINED: STRICT INTENT TRIAGE ---
        triage_prompt = f"""
        Classify the user's request INTENT into one of three categories:

        1. **THEN_VS_NOW_PROPOSAL**: ONLY if the user EXPLICITLY asks for a "Then vs Now" comparison, a "Nostalgic Tale", or a structured comparison of eras.
           *Examples:* "Create a Then vs Now story about SEO.", "Write a nostalgic tale about dial-up internet."

        2. **TOPIC_CLUSTER_PROPOSAL**: If the user asks for a "Topic Cluster", "Pillar Page", or "Content Strategy" breakdown.

        3. **DIRECT_ANSWER**: For ALL other requests. This includes general creative writing, drafting manifestos, writing blog posts, summarizing articles, or answering questions.
           *Examples:* "Write a manifesto for a design agency.", "Draft a blog post about coffee.", "Summarize this URL."

        USER REQUEST: "{topic}"

        CRITICAL: Respond with ONLY the category name.
        """
        intent = flash_model.generate_content(triage_prompt).text.strip()
        print(f"Smart Triage V2 classified intent as: {intent}")

        # --- 2. Research (Shared) ---
        research_data = find_trending_keywords(topic)
        clean_topic = research_data['clean_topic']
        
        new_events = [{"event_type": "user_request", "text": topic, "timestamp": str(datetime.datetime.now())}]
        if "tool_logs" in research_data: new_events.extend(research_data["tool_logs"])

        # --- 3. Routing ---
        if intent == "DIRECT_ANSWER":
            # This path now handles "Draft a manifesto" correctly as a text generation task
            # It creates a 'agent_answer' event, which your new Universal Approval logic CAN verify.
            answer_text = generate_comprehensive_answer(topic, research_data['context'])
            new_events.append({"event_type": "agent_answer", "text": answer_text})
            
            db.collection('agent_sessions').document(session_id).update({
                "status": "awaiting_feedback", "type": "work_answer", "topic": clean_topic,
                "slack_context": slack_context, "event_log": firestore.ArrayUnion(new_events)
            })
            requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={"session_id": session_id, "type": "social", "message": answer_text, "channel_id": slack_context['channel'], "thread_ts": slack_context['ts']}, verify=True)
            return jsonify({"msg": "Answer sent"}), 200

        elif intent == "TOPIC_CLUSTER_PROPOSAL":
            cluster_data = generate_topic_cluster(clean_topic, research_data['context'])
            formatted_cluster = f"Here is the topic cluster you requested:\n```\n{json.dumps(cluster_data, indent=2)}\n```"
            new_events.append({"event_type": "agent_answer", "proposal_type": "topic_cluster", "data": cluster_data})
            
            db.collection('agent_sessions').document(session_id).update({
                "status": "awaiting_feedback", "type": "work_proposal", "topic": clean_topic,
                "slack_context": slack_context, "event_log": firestore.ArrayUnion(new_events)
            })
            requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={"session_id": session_id, "type": "social", "message": formatted_cluster, "channel_id": slack_context['channel'], "thread_ts": slack_context['ts']}, verify=True)
            return jsonify({"msg": "Topic cluster sent"}), 200

        elif intent == "THEN_VS_NOW_PROPOSAL":
            try:
                # This path is now RESERVED for the specialized template
                current_proposal = create_euphemistic_links(research_data)
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
                    "slack_context": slack_context, "event_log": firestore.ArrayUnion(new_events)
                })
                requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={"session_id": session_id, "approval_id": approval_id, "proposal": current_proposal['interlinked_concepts'], "thread_ts": slack_context['ts'], "channel_id": slack_context['channel'], "is_initial_post": True }, verify=True)
                return jsonify({"msg": "Proposal sent"}), 200

            except ValueError as e:
                # Fallback
                answer_text = generate_comprehensive_answer(topic, research_data['context'])
                new_events.append({"event_type": "agent_answer", "text": answer_text})
                db.collection('agent_sessions').document(session_id).update({
                    "status": "awaiting_feedback", "type": "work_answer", "topic": clean_topic,
                    "slack_context": slack_context, "event_log": firestore.ArrayUnion(new_events)
                })
                requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={"session_id": session_id, "type": "social", "message": answer_text, "channel_id": slack_context['channel'], "thread_ts": slack_context['ts']}, verify=True)
                return jsonify({"msg": "Fallback answer sent"}), 200

        else: 
            return jsonify({"error": f"Unknown intent: {intent}"}), 500

    except Exception as e:
        print(f"Worker Error: {e}")
        return jsonify({"error": str(e)}), 500

# --- FUNCTION: Ingest Knowledge ---
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