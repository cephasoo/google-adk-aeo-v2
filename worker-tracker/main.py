import functions_framework
import os
import json
import logging
import requests
import datetime
import gspread
import google.auth
import litellm
from google.cloud import firestore
from google.cloud import secretmanager
from google.auth.transport.requests import Request as GoogleRequest
from googleapiclient.discovery import build
import google.oauth2.id_token
import litellm

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)
os.environ["LITELLM_LOG"] = "INFO"

# --- LiteLLM Configuration (Matching worker-story) ---
litellm.drop_params = True 

# --- Configuration ---
PROJECT_ID = os.environ.get("PROJECT_ID")
LOCATION = os.environ.get("LOCATION", "us-central1")
MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL") # URL of the Sensory Hub
SHEET_ID = os.environ.get("SHEET_ID") # The Google Sheet ID acting as CMD Center
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") 

# Configure LiteLLM (Redundant but Safe)
litellm.drop_params = True 

# --- Clients ---
db = None
secrets_client = None
sheet_client = None

def get_db():
    global db
    if db is None:
        db = firestore.Client(project=PROJECT_ID)
    return db

def get_secrets_client():
    global secrets_client
    if secrets_client is None:
        secrets_client = secretmanager.SecretManagerServiceClient()
    return secrets_client

def get_sheet_client():
    global sheet_client
    if sheet_client is None:
        # Authenticate using the default service account credentials
        credentials, project = google.auth.default(scopes=[
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive'
        ])
        sheet_client = gspread.authorize(credentials)
    return sheet_client

def get_secret(secret_id):
    """Retrieves a secret from Google Cloud Secret Manager."""
    client = get_secrets_client()
    try:
        name = f"projects/{PROJECT_ID}/secrets/{secret_id}/versions/latest"
        response = client.access_secret_version(name=name)
        return response.payload.data.decode("UTF-8")
    except Exception as e:
        logging.error(f"Failed to access secret {secret_id}: {e}")
        return None

# --- Classes ---

class RemoteTools:
    """
    Acts as a bridge to the MCP Sensory Tools Server.
    Reuse of the robust client from worker-story.
    """
    def __init__(self, server_url):
        self.server_url = server_url.rstrip("/")
        # Automatically detect if we need authentication (Cloud Run or Cloud Functions)
        self.is_gcp = "run.app" in server_url or "cloudfunctions.net" in server_url

    def _get_id_token(self):
        """Fetches an OIDC ID token from the metadata server (only works on GCP)."""
        try:
            logging.info(f"DEBUG: Fetching ID token for audience: {self.server_url}")
            auth_req = google.auth.transport.requests.Request()
            # The audience must be the service URL
            token = google.oauth2.id_token.fetch_id_token(auth_req, self.server_url)
            logging.info(f"DEBUG: Token generated successfully. Prefix: {token[:10] if token else 'None'}")
            return token
        except Exception as e:
            logging.error(f"DEBUG: Auth Hint: If local, ensure you are authenticated. Error: {e}")
            return None

    def call(self, tool_name, arguments):
        logging.info(f"MCP: Calling remote tool '{tool_name}' with args {arguments}")
        
        headers = {}
        if self.is_gcp:
            token = self._get_id_token()
            if token:
                headers["Authorization"] = f"Bearer {token}"

        try:
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
                timeout=60
            )
            if response.status_code != 200:
                logging.error(f"MCP Server Error {response.status_code}: {response.text}")
                return None
            
            data = response.json()
            content = data.get("result", {}).get("content", [])
            if content:
                return content[0].get("text", "")
            return ""
        except Exception as e:
            logging.error(f"MCP Client Error: {str(e)}")
            return None

class UnifiedModel:
    """
    Abstractions for LiteLLM to provide a unified interface for different providers.
    """
    def __init__(self, model_name="claude-sonnet-4-5", temperature=0.0):
        self.model_name = model_name
        self.temperature = temperature

    def generate(self, prompt, system_instruction=None):
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        
        try:
            # LiteLLM handles the specific provider logic (Anthropic, Vertex, OpenAI)
            response = litellm.completion(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"UnifiedModel Generation Failed ({self.model_name}): {e}")
            return None

    def classify_intent(self, query):
        """
        Classifies a query as 'Informational' or 'Commercial/Transactional'.
        """
        prompt = f"Classify the search intent of this query: '{query}'. Reply with only 'Informational' or 'Commercial'."
        try:
            # Reusing the generate method but with stricter constraints if needed
            res = self.generate(prompt)
            if res and "info" in res.lower():
                return "Informational"
            return "Commercial"
        except Exception as e:
            logging.error(f"Intent Triage Failed: {e}")
            return "Informational" # Default to info for AEO safety

class DiscoveryRadar:
    """
    Connects to Google Search Console to find 'Zero-Click' opportunities.
    """
    def __init__(self):
        try:
            # We use the same default credentials
            credentials, _ = google.auth.default(scopes=['https://www.googleapis.com/auth/webmasters.readonly'])
            self.service = build('searchconsole', 'v1', credentials=credentials)
        except Exception as e:
            logging.error(f"GSC Init Failed: {e}")
            self.service = None

    def fetch_zero_click_candidates(self, site_url, days=7, imps_threshold=50):
        """
        Fetches queries with High Impressions, Low CTR, and Position < 20.
        """
        if not self.service: return []
        
        end_date = datetime.date.today().isoformat()
        # Expanding to 7 days because GSC data is often delayed by 48-72 hours
        start_date = (datetime.date.today() - datetime.timedelta(days=7)).isoformat()
        
        request = {
            "startDate": start_date,
            "endDate": end_date,
            "dimensions": ["query"],
            "rowLimit": 1000
        }

        # Country/Region Filter (ISO 3-letter code e.g. 'usa')
        region_code = kwargs.get("region_code")
        if region_code:
            request["dimensionFilterGroups"] = [{
                "filters": [{
                    "dimension": "country",
                    "operator": "equals",
                    "expression": region_code.lower()
                }]
            }]
        
        try:
            # Domain Property Handling:
            # If the user passed a URL, we attempt to resolve the property name.
            # Domain properties must be prefixed with 'sc-domain:'
            gsc_site_name = site_url
            if "://" in site_url and "sc-domain:" not in site_url:
                # Strip protocol and trailing slash for domain property check
                domain = site_url.split("://")[1].split("/")[0]
                gsc_site_name = f"sc-domain:{domain}"
                logging.info(f"Normalizing {site_url} to Domain Property: {gsc_site_name}")

            response = self.service.searchanalytics().query(siteUrl=gsc_site_name, body=request).execute()
            rows = response.get('rows', [])
            logging.info(f"GSC Response: Found {len(rows)} raw rows for {gsc_site_name}")
            
            candidates = []
            for row in rows:
                imps = row.get('impressions', 0)
                ctr = row.get('ctr', 0)
                pos = row.get('position', 0)
                query = row.get('keys', [""])[0]
                logging.info(f"GSC Row: {query} (Imps: {imps}, Pos: {pos})")
                
                # Dynamic threshold for bootstrap vs production
                if imps >= imps_threshold: 
                    candidates.append({
                        "query": query,
                        "imps": imps,
                        "ctr": ctr,
                        "pos": pos,
                        "metrics": f"Imps: {imps}, CTR: {ctr:.1%}, Pos: {pos:.1f}"
                    })
            
            # Sort by impressions (highest first)
            candidates.sort(key=lambda x: float(x['metrics'].split(',')[0].split(':')[1]), reverse=True)
            return candidates[:20] # Return top 20
            
        except Exception as e:
            logging.error(f"GSC Query Failed: {e}")
            return []

class SocialListener:
    """
    Uses MCP (SerpAPI) to find 'Narrative Gaps' in social discussions.
    Current MVP: Focused on Reddit. Expandable to LinkedIn/Quora.
    """
    def __init__(self, mcp_client):
        self.mcp = mcp_client

    def check_narrative_gaps(self, keywords):
        """
        Searches Reddit/Quora for high-engagement threads on these keywords.
        Returns a list of 'suggested prompts' if the topic is hot.
        """
        suggestions = []
        for kw in keywords:
            # Check Reddit (Primary 'Trust Seed' source for MVP)
            # Todo: Add site:linkedin.com and site:quora.com in future iterations
            query = f"site:reddit.com {kw}"
            try:
                # We use the generic tool or a specific logical check
                # For simplicity, we assume we get a text blob back and look for density
                result = self.mcp.call("google_web_search", {"query": query})
                
                # Heuristic: If we find a result with "comments" > 20, it's a candidate
                # This is a basic implementation; a real one would parse the SERP JSON structure
                if "comments" in str(result).lower(): 
                   suggestions.append({
                       "query": f"What is the consensus on {kw}?",
                       "metrics": "Source: Reddit (High Engagement)"
                   })
            except Exception as e:
                logging.error(f"Social Listen Failed for {kw}: {e}")
                
        return suggestions

# --- Sheet Constants ---
WS_CAMPAIGNS = "Campaign Manager"
WS_PROMPTS = "LLM Prompts"
WS_LOGS = "The Scoreboard (Output)"
WS_SUGGESTIONS = "Suggested Prompts"

class SheetManager:
    """
    Handles all interactions with the Google Sheet CMD Center.
    """
    def __init__(self, gc, sheet_id):
        self.gc = gc
        self.sheet_id = sheet_id
        try:
            self.sh = self.gc.open_by_key(self.sheet_id)
        except Exception as e:
            raise ValueError(f"Could not open Sheet ID {sheet_id}: {e}")
            
        # Ensure Critical Tabs Exist
        self._ensure_tab(WS_SUGGESTIONS, ["Query", "Metrics", "Added Date", "Status", "AI Insights"])

    def _ensure_tab(self, tab_name, default_headers):
        """Creates a tab if it doesn't exist."""
        try:
            self.sh.worksheet(tab_name)
        except gspread.WorksheetNotFound:
            logging.warning(f"'{tab_name}' not found. Creating it.")
            ws = self.sh.add_worksheet(tab_name, rows=100, cols=10)
            ws.append_row(default_headers)

    def _get_col(self, row_dict, possible_keys):
        """Helper to get value from dict using multiple possible keys."""
        for k in possible_keys:
            if k in row_dict:
                return row_dict[k]
        return None

    def get_active_prompts(self):
        """
        Reads 'Campaign Manager' to find active IDs, then fetches matching rows from 'LLM Prompts'.
        """
        # 1. Get Active Campaigns
        try:
            campaigns_ws = self.sh.worksheet(WS_CAMPAIGNS)
        except gspread.WorksheetNotFound:
            # Fallback if user deleted it
            return []

        campaigns = campaigns_ws.get_all_records()
        
        # Robust status check
        active_ids = set()
        for c in campaigns:
            status = self._get_col(c, ["Status", "Status (Active/Paused)", "status"])
            cid = self._get_col(c, ["Campaign_ID", "Campaign ID", "id"])
            if status and str(status).upper() == 'ACTIVE' and cid:
                active_ids.add(cid)
        
        if not active_ids:
            logging.info("No ACTIVE campaigns found.")
            return []

        # 2. Get Prompts for those Campaigns
        try:
            prompts_ws = self.sh.worksheet(WS_PROMPTS)
        except gspread.WorksheetNotFound:
            return []

        all_prompts = prompts_ws.get_all_records()
        active_prompts = []
        
        for p in all_prompts:
            p_cid = self._get_col(p, ["Campaign_ID", "Campaign ID"])
            p_text = self._get_col(p, ["Prompt_Text", "Prompt Text", "prompt", "Prompt/Question"])
            p_brand = self._get_col(p, ["Brand", "Target Brand", "brand"])  # Matches user's new column
            
            if p_cid in active_ids:
                # Find the brand from the campaign list if not in prompts list
                if not p_brand:
                    campaign = next((c for c in campaigns if self._get_col(c, ["Campaign_ID", "Campaign ID", "id"]) == p_cid), {})
                    p_brand = self._get_col(campaign, ["Target Brand", "Brand", "brand_name", "Brand Name"])

                # Fetch Focus Attributes for the campaign
                p_attributes = self._get_col(p, ["Attributes", "Focus Attributes", "core_concepts"])
                if not p_attributes:
                    campaign = next((c for c in campaigns if self._get_col(c, ["Campaign_ID", "Campaign ID", "id"]) == p_cid), {})
                    p_attributes = self._get_col(campaign, ["Focus Attributes", "Attributes", "core_concepts", "Keywords"])

                # Normalize the prompt dict for the worker
                p_region = self._get_col(p, ["Region", "Country", "region"])
                if not p_region:
                    campaign = next((c for c in campaigns if self._get_col(c, ["Campaign_ID", "Campaign ID", "id"]) == p_cid), {})
                    p_region = self._get_col(campaign, ["Region", "Country", "region"])

                active_prompts.append({
                    "Campaign_ID": p_cid,
                    "Brand": p_brand or p_cid, # Fallback to CID if no brand name
                    "Attributes": p_attributes or "General relevance",
                    "Prompt_Text": p_text,
                    "Region": p_region or "us"
                })
        
        # 3. Get Approved Suggestions (The Triage Loop)
        try:
            sugg_ws = self.sh.worksheet(WS_SUGGESTIONS)
            all_suggestions = sugg_ws.get_all_records()
            for s in all_suggestions:
                s_status = self._get_col(s, ["Status", "status"])
                if s_status and str(s_status).upper() == "APPROVED":
                    s_query = self._get_col(s, ["Query", "query"])
                    # Default suggestions to the first active campaign brand for now
                    default_brand = "General Brand" 
                    if active_prompts:
                        default_brand = active_prompts[0]["Brand"]
                        
                    active_prompts.append({
                        "Campaign_ID": "TRIAGE-SUGG",
                        "Brand": default_brand,
                        "Prompt_Text": s_query
                    })
        except Exception as e:
            logging.warning(f"Failed to fetch suggestions during audit: {e}")

        logging.info(f"Found {len(active_prompts)} total prompts (Campaign + Approved Suggestions) to audit.")
        return active_prompts

    def log_audit_result(self, result_dict):
        """
        Writes a single audit result to 'The Scoreboard (Output)'.
        Adapts to user's headers if they exist.
        """
        try:
            logs_ws = self.sh.worksheet(WS_LOGS)
        except gspread.WorksheetNotFound:
            logs_ws = self.sh.add_worksheet(WS_LOGS, rows=1000, cols=10)
            logs_ws.append_row(["Timestamp", "Brand", "Prompt", "Mentioned (Yes/No)", "Sentiment", "Link Position", "AI Text Snippet", "Source Tool"])

        # Create row data mapping
        row_data = {
            "Timestamp": datetime.datetime.now().isoformat(),
            "Brand": result_dict.get("campaign_id", ""), # Mapping Campaign ID to Brand
            "Campaign ID": result_dict.get("campaign_id", ""),
            "Prompt": result_dict.get("prompt", ""),
            "Mentioned (Yes/No)": "Yes" if result_dict.get("cited") else "No",
            "Cited": "Yes" if result_dict.get("cited") else "No",
            "Sentiment": result_dict.get("sentiment_score", 0),
            "Sentiment Score": result_dict.get("sentiment_score", 0),
            "Link Position": result_dict.get("position", 0),
            "Position": result_dict.get("position", 0),
            "AI Text Snippet": result_dict.get("snippet", "")[:5000],
            "Narrative Snippet": result_dict.get("snippet", "")[:5000],
            "Source Tool": result_dict.get("tool", "Unknown")
        }

        # Get current headers
        headers = logs_ws.row_values(1)
        if not headers:
            headers = ["Timestamp", "Brand", "Prompt", "Mentioned (Yes/No)", "Sentiment", "Link Position", "AI Text Snippet", "Source Tool"]
            logs_ws.append_row(headers)

        # Build row based on specific headers found
        final_row = []
        for h in headers:
            val = row_data.get(h, "") # Default to empty if header unknown
            # Fallback for approximate matches if needed, but strict is safer for now
            final_row.append(val)
        
        # If the user is missing 'Prompt', we really should append it because it's critical
        if "Prompt" not in headers:
            # We append it to the row AND the header if it's not there? 
            # Safe approach: Just append it to the data row, GSheets handles extra cells fine.
            # But better to check.
            if len(final_row) < 20: # Arbitrary safety
               final_row.append(f"[Promt: {result_dict.get('prompt','')}]") 

        logs_ws.append_row(final_row)

    def queue_suggestions(self, candidates):
        """
        Writes GSC candidates to 'Suggested Prompts', avoiding duplicates.
        """
        if not candidates: return
        
        # We ensure tab exists in __init__, so just get it
        try:
            sugg_ws = self.sh.worksheet(WS_SUGGESTIONS)
        except gspread.WorksheetNotFound:
            # Should happen in init, but safety first
            self._ensure_tab(WS_SUGGESTIONS, ["Query", "Metrics", "Added Date", "Status"])
            sugg_ws = self.sh.worksheet(WS_SUGGESTIONS)

        # Get existing queries to avoid duplicates
        try:
            existing_vals = sugg_ws.get_all_values()
            if len(existing_vals) > 1:
                existing = set(row[0] for row in existing_vals[1:])
            else:
                existing = set()
        except:
            existing = set()
        
        new_rows = []
        today = datetime.date.today().isoformat()
        
        for cand in candidates:
            if cand['query'] not in existing:
                row_status = cand.get('status', 'NEW')
                ai_insight = cand.get('ai_insight', '')
                new_rows.append([cand['query'], cand['metrics'], today, row_status, ai_insight])
        
        if new_rows:
            sugg_ws.append_rows(new_rows)
            logging.info(f"Queued {len(new_rows)} new suggestions from GSC.")


# --- Analysis Logic ---

def analyze_citation(unified_model, prompt_text, scrape_content):
    """
    Uses the UnifiedModel (Claude via LiteLLM) to perform an Entity-Attribute Audit.
    """
    try:
        if not scrape_content:
            return {"cited": False, "position": 0, "sentiment_score": 0, "snippet": "No content to analyze."}

        analysis_prompt = f"""
        You are a World-Class AEO (Answer Engine Optimization) Strategist. Your task is to perform a Narrative Triangulation Audit.
        
        TARGET BRAND: "{unified_model.current_brand if hasattr(unified_model, 'current_brand') else 'Target Brand'}"
        QUERY PROMPT: "{prompt_text}"
        STRATEGIC FOCUS ATTRIBUTES: "{unified_model.current_attributes if hasattr(unified_model, 'current_attributes') else 'General Authority'}"
        
        SOURCE CONTENT:
        {scrape_content[:15000]} 

        TASK:
        1. **Citation Audit**: Is the TARGET BRAND explicitly cited? Provide True/False and link position.
        2. **Consensus Leaders**: Identify the specific sites the AI trusts as its primary sources for this query.
        3. **Entity-Attribute Audit**: 
           - Does the AI associate the TARGET BRAND with the specific STRATEGIC FOCUS ATTRIBUTES mentioned above?
           - Is the brand framed as the "Originator" (Category Leader) or an "Example"?
        4. **The Semantic Gap**: If not cited, identify the "Missing Piece"—what specific expert proof or data is the cited leader providing that you are not?
        
        OUTPUT FORMAT (STRICT JSON):
        Return a JSON object with this EXACT structure:
        {{
            "is_cited": boolean,
            "link_position": integer,
            "sentiment_score": 1-10,
            "snippet": "Use the following structured format inside this string field:
            
            THE ACTUAL AUTHORITATIVE SOURCES CITED ARE:
            (1) [Source Name] - [Why they were chosen]
            (2) [Source Name] - [Why they were chosen]
            
            --- NARRATIVE GAP ANALYIS ---
            * Attribute Association: [How well does the AI link the brand to: {unified_model.current_attributes if hasattr(unified_model, 'current_attributes') else 'the core concepts'}]
            * The Missing Piece: [Specific Content Recommendation to close the gap]"
        }}
        """
        
        logging.info(f"LLM Audit Triggered. Scrape Content Length: {len(scrape_content)}. Preview: {scrape_content[:200]}")
        # Use the injected model instance
        content = unified_model.generate(analysis_prompt)
        
        if not content:
             return {"cited": False, "position": 0, "sentiment_score": 0, "snippet": "Generation Empty"}
             
        text = content.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
        snippet_text = data.get("snippet", "").strip()
        if not snippet_text:
            snippet_text = "No citation found."

        return {
            "cited": data.get("is_cited", False),
            "position": data.get("link_position", 0),
            "sentiment_score": data.get("sentiment_score", 5),
            "snippet": snippet_text
        }
    except Exception as e:
        logging.error(f"UnifiedModel Analysis Failed: {e}")
        return {"cited": False, "position": 0, "sentiment_score": 0, "snippet": "Analysis Error"}


# --- Core Logic ---

@functions_framework.http
def process_tracker_logic(request):
    """
    HTTP Entry point for the Worker Tracker.
    """
    try:
        logging.info("Starting Worker Tracker execution...")
        
        # --- ROBUST PAYLOAD PARSING ---
        try:
            # 1. Standard approach
            req_json = request.get_json(silent=True)
            
            # 2. Fallback: Parse raw data
            if not req_json and request.data:
                raw_str = request.data.decode('utf-8').strip()
                try:
                    req_json = json.loads(raw_str)
                except:
                    # 3. Recovery for malformed CLI strings
                    try:
                        import ast
                        req_json = ast.literal_eval(raw_str)
                    except:
                        # 4. Lazy Regex Parser fallback
                        try:
                            import re
                            pairs = re.findall(r'([a-zA-Z0-9_-]+):\s*([^,}]+)', raw_str)
                            recovered = {k.strip(): v.strip() for k, v in pairs}
                            if recovered: req_json = recovered
                        except: pass
            
            req_json = req_json or {}
        except Exception as e:
            logging.error(f"Payload parse failed: {e}")
            req_json = {}
        
        site_url = req_json.get("site_url") 
        
        if not SHEET_ID:
            return "Error: SHEET_ID environment variable not set.", 500
        
        # 0. Initialize
        # Note: No explicit vertexai.init here as we use LiteLLM now
        gc = get_sheet_client()
        sheet_mgr = SheetManager(gc, SHEET_ID)
        mcp = RemoteTools(MCP_SERVER_URL) if MCP_SERVER_URL else None
        # Initialize UnifiedModel with the desired Anthropic model
        unimodel = UnifiedModel(model_name="claude-sonnet-4-5") 
        
        # --- PHASE 1: DISCOVERY RADAR ---
        all_sites = set()
        if site_url:
            all_sites.add(site_url)
        else:
            # Multi-Site Dynamic Loop: Fetch unique URLs from active campaigns
            active_campaigns = sheet_mgr.sh.worksheet(WS_CAMPAIGNS).get_all_records()
            for c in active_campaigns:
                status = sheet_mgr._get_col(c, ["Status", "status"])
                s_url = sheet_mgr._get_col(c, ["Site URL", "site_url", "Domain", "domain"])
                if status and str(status).upper() == "ACTIVE" and s_url:
                    all_sites.add(s_url)
        
        gsc_candidates = []
        radar = DiscoveryRadar()
        for s in all_sites:
            # Multi-Region Discovery: Find the first active campaign with this URL to get its region
            s_region = "usa" # Default
            active_campaigns = sheet_mgr.sh.worksheet(WS_CAMPAIGNS).get_all_records()
            for c in active_campaigns:
                if sheet_mgr._get_col(c, ["Site URL", "site_url"]) == s:
                    s_region = sheet_mgr._get_col(c, ["Region", "Country", "region"]) or "usa"
                    break

            logging.info(f"Running Discovery Radar for {s} (Region: {s_region})...")
            # Reverted to Production Threshold (50 Imps)
            site_candidates = radar.fetch_zero_click_candidates(s, imps_threshold=50, region_code=s_region)
            
            # --- TIERED TRIAGE LOOP ---
            for cand in site_candidates:
                intent = unimodel.classify_intent(cand['query'])
                cand['intent'] = intent
                
                # Flag 1: Metric-Based Cannibalization
                is_low_ctr = cand['ctr'] < 0.01  # < 1%
                is_high_rank = cand['pos'] < 5.0
                
                cand['status'] = "NEW" # Preserve the user's dropdown logic
                
                if is_high_rank and is_low_ctr:
                    if intent == "Informational":
                        cand['ai_insight'] = "[AEO INTENT VALIDATED]" # Flag 2
                    else:
                        cand['ai_insight'] = "[AEO CANNIBALIZED]" # Flag 1
                
                # Flag 3: AEO Opportunity (Informational + High Imps)
                elif intent == "Informational":
                    cand['ai_insight'] = "[AEO OPPORTUNITY]"
                else:
                    cand['ai_insight'] = "Standard Discovery"

            sheet_mgr.queue_suggestions(site_candidates)
            gsc_candidates.extend(site_candidates)
            
        # --- PHASE 2: SOCIAL LISTENER ---
        # We trigger this occasionally, maybe if 'listen=true' in request
        if req_json.get("listen"):
            logging.info("Running Social Listener...")
            listener = SocialListener(mcp)
            # In a real app, these keywords would come from a 'Clusters' tab
            # For now, we seed with main topics
            hot_topics = listener.check_narrative_gaps(["HITL SEO", "Programmatic SEO Ethics"])
            sheet_mgr.queue_suggestions(hot_topics)
        
        # --- PHASE 3: AUDIT LOOP ---
        # 1. Fetch Work (Manual + Suggestions)
        prompts = sheet_mgr.get_active_prompts()
        
        # 2. Event-Driven Filter (Gated by GSC if requested)
        if req_json.get("event_driven"):
            logging.info("AEO Event-Driven Gate Active. Filtering for GSC signals...")
            gsc_queries = {c['query'] for c in gsc_candidates}
            # Only keep prompts that are either manual or match a GSC candidate
            prompts = [p for p in prompts if p.get("Prompt_Text") in gsc_queries or p.get("Campaign_ID") != "TRIAGE-SUGG"]

        if not prompts:
            return "No active prompts matching trigger criteria.", 200
            
        results_count = 0
        
        # 2. Execution Loop
        for item in prompts:
            prompt_text = item.get("Prompt_Text")
            campaign_id = item.get("Campaign_ID")
            brand_name = item.get("Brand")
            
            if not prompt_text: continue
            
            logging.info(f"Auditing: {prompt_text} for Brand: {brand_name}")
            
            # A. Scrape (Using SerpAPI via MCP)
            if not mcp:
                logging.error("MCP Client not initialized (Missing URL). Skipping scrape.")
                continue
            
            # Use 'Region' for gl (country) and hl (language)
            # Standardizing 3-letter (GSC) to 2-letter (Search) codes
            target_region = str(item.get("Region", "us")).lower()
            # ISO 3166-1 alpha-3 to alpha-2 map for common markets
            iso_map = {
                "usa": "us", "gbr": "gb", "can": "ca", "nga": "ng", 
                "aus": "au", "ind": "in", "deu": "de", "fra": "fr"
            }
            gl_code = iso_map.get(target_region, target_region[:2]) # Fallback to first 2 letters
                
            # We use 'google_web_search' to get the SERP which includes AI Overviews
            scrape_result = mcp.call("google_web_search", {
                "query": prompt_text,
                "gl": gl_code,
                "hl": "en"
            })
            
            # Setup model for this run
            unimodel.current_brand = brand_name
            unimodel.current_attributes = item.get("Attributes", "General Relevance")
            # B. Analyze with LiteLLM (Claude) via UnifiedModel
            audit_result = analyze_citation(unimodel, prompt_text, scrape_result)
            
            # Inject Usage Metadata
            audit_result["campaign_id"] = campaign_id
            audit_result["brand_name"] = brand_name
            audit_result["prompt"] = prompt_text
            audit_result["tool"] = "serpapi_web_search"
            
            # C. Log
            sheet_mgr.log_audit_result(audit_result)
            results_count += 1
            
        return f"Audit Complete. Processed {results_count} prompts. Semantic Gap Analysis active.", 200

    except Exception as e:
        logging.error(f"Worker Tracker Failed: {e}")
        return f"Error: {e}", 500
