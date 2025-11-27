# --- /worker-story/main.py ---
import functions_framework
from flask import jsonify
import vertexai
from vertexai.generative_models import GenerativeModel
import os
import json
import re
import uuid
import requests
from bs4 import BeautifulSoup
from google.cloud import secretmanager, firestore
from googleapiclient.discovery import build
import datetime

# --- Configuration ---
PROJECT_ID = os.environ.get("PROJECT_ID")
LOCATION = os.environ.get("LOCATION")
MODEL_NAME = os.environ.get("MODEL_NAME")
SEARCH_ENGINE_ID = os.environ.get("SEARCH_ENGINE_ID")
BROWSERLESS_API_KEY = os.environ.get("BROWSERLESS_API_KEY")
N8N_PROPOSAL_WEBHOOK_URL = os.environ.get("N8N_PROPOSAL_WEBHOOK_URL") 
# MAX_LOOP_ITERATIONS moved to logic or kept as const
MAX_LOOP_ITERATIONS = 2 
PROPOSAL_KEYWORDS = ["outline", "draft", "proposal", "story", "brief"]

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
    if not match: raise ValueError(f"No JSON found: {text}")
    return json.loads(match.group(0))

def extract_url_from_text(text):
    url_pattern = r'(https?://[^\s<>|]+)'
    match = re.search(url_pattern, text)
    if match: return match.group(1)
    return None

# --- Tools (The Heavy Lifters) ---

def fetch_article_content(url):
    """Robustly scrapes content using Browserless REST API."""
    print(f"Tool: Reading URL via Browserless: {url}")
    if not BROWSERLESS_API_KEY: return "Error: Browserless API Key is missing."
    endpoint = f"https://production-sfo.browserless.io/content?token={BROWSERLESS_API_KEY}&stealth=true"
    payload = {
        "url": url,
        "rejectResourceTypes": ["image", "media", "font"],
        "gotoOptions": {"timeout": 15000, "waitUntil": "networkidle2"}
    }
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
        # ... (Google Search Logic - Unchanged) ...
        # (Condensed for brevity - logic identical to previous working main.py)
        extraction_prompt = f"User Input: '{unstructured_topic}'... Task: Convert to Google Search Query..."
        search_query = model.generate_content(extraction_prompt).text.strip()
        api_key = get_search_api_key()
        service = build("customsearch", "v1", developerKey=api_key)
        tool_logs.append({"event_type": "tool_call", "tool_name": "google_search", "input": search_query, "timestamp": str(datetime.datetime.now())})
        res = service.cse().list(q=f"{search_query}", cx=SEARCH_ENGINE_ID, num=10).execute()
        google_snippets = [r.get('snippet', '') for r in res.get('items', [])]
        context_snippets = google_snippets
        clean_topic = search_query

    return {"context": context_snippets, "clean_topic": clean_topic, "is_grounded": is_grounded, "tool_logs": tool_logs}

def generate_topic_cluster(topic, context):
    """Generates a structured topic cluster for a pillar page."""
    global model
    print("Tool: Topic Cluster Generator")
    prompt = f"""
    You are a master Content Strategist. The user wants to create a pillar page on the topic of "{topic}".
    Use the provided context to generate a detailed topic cluster. The cluster should have a main pillar and several sub-topics.
    
    CONTEXT:
    {context}

    CRITICAL: Respond with a JSON object following this exact schema:
    {{
      "pillar_page_title": "The Definitive Guide to [Topic]",
      "clusters": [
        {{
          "cluster_title": "Understanding the Fundamentals",
          "sub_topics": [
            "What is [Topic]?",
            "The History of [Topic]",
            "Why [Topic] Matters in 2025"
          ]
        }},
        {{
          "cluster_title": "Advanced Strategies",
          "sub_topics": [
            "Advanced Technique A for [Topic]",
            "Integrating [Topic] with Other Systems",
            "Case Study: A Successful Implementation of [Topic]"
          ]
        }}
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
        system_instruction = "CRITICAL INSTRUCTION: You are in READING MODE. Base answer PRIMARILY on 'GROUNDING_CONTENT'. Do not hallucinate."
    else:
        temperature = 0.7
        system_instruction = "You are an expert AI assistant. Use the research context to provide accurate answers."

    prompt = f"{system_instruction}\n\nThe user asked: \"{topic}\"\n\nResearch Context:\n{context}\n\nAnswer:"
    return model.generate_content(prompt, generation_config={"temperature": temperature}).text.strip()

def create_euphemistic_links(keyword_context):
    global model
    prompt = f"Topic: '{keyword_context['clean_topic']}'... Create 'Then' and 'Now' clusters..."
    return extract_json(model.generate_content(prompt).text)

def critique_proposal(topic, current_proposal):
    global model
    return model.generate_content(f"Review proposal for '{topic}'...").text.strip()

def refine_proposal(topic, current_proposal, critique):
    global model
    return extract_json(model.generate_content(f"REWRITE proposal...").text)

# --- The Main Worker Function ---
@functions_framework.http
def process_story_logic(request):
    global model, flash_model, db
    if model is None: 
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        model = GenerativeModel(MODEL_NAME) # This is your powerful model (Pro)
        flash_model = GenerativeModel("gemini-2.5-flash-lite") # The fast triage model
        db = firestore.Client()

    req = request.get_json(silent=True)
    session_id = req['session_id']
    topic = req['topic'] # This is the full user prompt
    slack_context = req['slack_context']

    try:
        # --- NEW: LLM-BASED TRIAGE ---
        triage_prompt = f"""
        Classify the user's request into one of three categories:
        1.  DIRECT_ANSWER: The user is asking a direct question.
        2.  THEN_VS_NOW_PROPOSAL: The user is asking for a "Then vs Now" story, draft, or proposal.
        3.  TOPIC_CLUSTER_PROPOSAL: The user is asking for a topic cluster, pillar page ideas, or content strategy.

        USER REQUEST: "{topic}"

        Respond with ONLY the category name (e.g., DIRECT_ANSWER).
        """
        # We can use a cheaper/faster model for this classification if we wanted
        intent = model.generate_content(triage_prompt).text.strip()
        print(f"Smart Triage classified intent as: {intent}")

        # 2. Research (This is now shared by all paths)
        research_data = find_trending_keywords(topic)
        clean_topic = research_data['clean_topic']
        
        new_events = [{"event_type": "user_request", "text": topic, "timestamp": str(datetime.datetime.now())}]
        if "tool_logs" in research_data:
            new_events.extend(research_data["tool_logs"])

        # --- NEW: INTENT-BASED ROUTING ---
        if intent == "DIRECT_ANSWER":
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
            # Here you would decide how to format this JSON for N8N/Slack
            # For now, we'll just send the raw JSON as a text block
            formatted_cluster = f"Here is the topic cluster you requested:\n```\n{json.dumps(cluster_data, indent=2)}\n```"
            new_events.append({"event_type": "agent_answer", "proposal_type": "topic_cluster", "data": cluster_data})
            
            db.collection('agent_sessions').document(session_id).update({
                "status": "awaiting_feedback", "type": "work_proposal", "topic": clean_topic,
                "slack_context": slack_context, "event_log": firestore.ArrayUnion(new_events)
            })
            requests.post(N8N_PROPOSAL_WEBHOOK_URL, json={"session_id": session_id, "type": "social", "message": formatted_cluster, "channel_id": slack_context['channel'], "thread_ts": slack_context['ts']}, verify=True)
            return jsonify({"msg": "Topic cluster sent"}), 200

        elif intent == "THEN_VS_NOW_PROPOSAL":
            # This is your original proposal logic, now correctly routed
            current_proposal = create_euphemistic_links(research_data)
            new_events.append({"event_type": "loop_draft", "iteration": 0, "proposal_data": current_proposal})
            loop_count = 0
            while loop_count < MAX_LOOP_ITERATIONS:
                critique = critique_proposal(clean_topic, current_proposal)
                if "APPROVED" in critique: break
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

        else: # Fallback
            return jsonify({"error": f"Unknown intent: {intent}"}), 500

    except Exception as e:
        print(f"Worker Error: {e}")
        return jsonify({"error": str(e)}), 500

# --- FUNCTION: Ingest Knowledge ---
@functions_framework.http
def ingest_knowledge(request):
    global db, model
    if db is None: db = firestore.Client()
    if model is None:
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        model = GenerativeModel(MODEL_NAME)

    request_json = request.get_json(silent=True)
    session_id = request_json.get('session_id')
    final_story = request_json.get('story')
    topic = request_json.get('topic')
    
    if not final_story: return jsonify({"error": "Missing story"}), 400

    tagging_prompt = f"""
    Analyze story about "{topic}". Generate 5-10 searchable keywords.
    Response JSON format: {{ "keywords": ["tag1", "tag2"] }}
    STORY: {final_story[:2000]}
    """
    try:
        # Re-using the utility from the top of the file
        metadata = extract_json(model.generate_content(tagging_prompt).text)
    except Exception:
        metadata = {"keywords": [topic or "story"]}

    db.collection('knowledge_base').document(session_id).set({
        "topic": topic, "content": final_story, "keywords": metadata.get('keywords', []),
        "created_at": datetime.datetime.now(datetime.timezone.utc), "source_session": session_id
    })
    return jsonify({"msg": "Knowledge ingested."}), 200