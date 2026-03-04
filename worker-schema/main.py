# --- /worker-schema/main.py ---
import functions_framework
from flask import jsonify
import os
import json
import re
import datetime
import hashlib
import requests
import logging
from bs4 import BeautifulSoup
from google.cloud import firestore

from shared.utils import (
    get_secret,
    _firestore_call_with_timeout
)

# --- Configuration ---
PROJECT_ID = os.environ.get("PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("LOCATION", "us-central1")
BROWSERLESS_API_KEY = os.environ.get("BROWSERLESS_API_KEY")

# --- Lazy Global Clients (gRPC Safety) ---
_db = None

def get_db():
    global _db
    if _db is None:
        _db = firestore.Client(project=PROJECT_ID)
    return _db

# --- SCHEMA REGISTRY ---
# Defines required and high-signal fields for Google-supported schema types
# Reference: https://developers.google.com/search/docs/appearance/structured-data/search-gallery
SCHEMA_REGISTRY = {
    "Organization": ["name", "url", "logo", "sameAs"],
    "Person": ["name", "url", "sameAs", "knowsAbout"],
    "Product": ["name", "image", "description", "offers", "brand"],
    "FAQPage": ["mainEntity"],
    "Article": ["headline", "image", "datePublished", "author"],
    "BlogPosting": ["headline", "image", "datePublished", "author"],
    "BreadcrumbList": ["itemListElement"],
    "Event": ["name", "startDate", "location", "image"],
    "HowTo": ["name", "step", "supply", "tool"],
    "JobPosting": ["title", "description", "hiringOrganization", "jobLocation", "datePosted"],
    "LocalBusiness": ["name", "image", "address", "telephone"],
    "Recipe": ["name", "image", "recipeIngredient", "recipeInstructions"],
    "Review": ["itemReviewed", "reviewRating", "author"],
    "SoftwareApplication": ["name", "operatingSystem", "applicationCategory", "offers"],
    "VideoObject": ["name", "description", "thumbnailUrl", "uploadDate"],
    "WebPage": ["name", "description"],
    "WebSite": ["name", "url"],
    "ItemList": ["itemListElement"],
    "Dataset": ["name", "description"],
    "Course": ["name", "description", "provider"],
    "ProfilePage": ["mainEntity"],
    "Service": ["name", "serviceType", "provider"]
}

# --- CORE LOGIC: EXTRACTION ---
def extract_on_page_schema(url):
    """Fetches page via Browserless and extracts JSON-LD blocks."""
    api_key = BROWSERLESS_API_KEY or get_secret("browserless-api-key")
    if not api_key:
        raise ValueError("Missing BROWSERLESS_API_KEY")

    endpoint = f"https://production-sfo.browserless.io/content?token={api_key}&stealth=true"
    payload = {
        "url": url,
        "rejectResourceTypes": ["image", "font", "media"],
        "gotoOptions": {"timeout": 20000, "waitUntil": "networkidle2"}
    }
    
    response = requests.post(endpoint, json=payload, timeout=25)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.content, 'html.parser')
    schema_blocks = []
    
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string)
            if isinstance(data, list):
                schema_blocks.extend(data)
            else:
                schema_blocks.append(data)
        except Exception as e:
            logging.warning(f"Schema Parse Error: {e}")
            
    return schema_blocks

# --- CORE LOGIC: VALIDATION & SCORING ---
def calculate_eds(schema_blocks):
    """
    Calculates Entity Disambiguation Score (EDS).
    Focuses on 'Authority Anchoring' and 'Identity Resolution'.
    """
    if not schema_blocks:
        return 0, ["No JSON-LD schema found on page."], set()

    score = 30 # Base score for having any schema
    findings = []
    
    # 0. Detected Types (Metadata)
    found_types = set()
    for s in schema_blocks:
        if isinstance(s, dict) and s.get("@type"):
            stype = s["@type"]
            if isinstance(stype, list): found_types.update(stype)
            else: found_types.add(stype)
            
    if found_types:
        findings.append(f"SUCCESS: Detected Schema types: {', '.join(found_types)}")
    else:
        findings.append("WARNING: No Schema.org @type detected in JSON-LD blocks.")
    
    # 1. Identity Resolution (@id) - 30%
    primary_entities = [s for s in schema_blocks if isinstance(s, dict) and s.get("@id")]
    if primary_entities:
        score += 30
        ids = [e["@id"][:50] + "..." if len(e["@id"]) > 50 else e["@id"] for e in primary_entities]
        findings.append(f"SUCCESS: Found {len(primary_entities)} entities with @id URIs (e.g., {ids[0]}).")
    else:
        findings.append("CRITICAL: Missing @id. Identity is 'floating' (Harder for AI to cluster).")

    # 2. Authority Anchoring (sameAs) - 20%
    all_same_as = []
    for s in schema_blocks:
        if isinstance(s, dict) and s.get("sameAs"):
            val = s["sameAs"]
            if isinstance(val, list): all_same_as.extend(val)
            else: all_same_as.append(val)
    
    if len(all_same_as) >= 5:
        score += 20
        findings.append(f"High-Density social proof: {len(all_same_as)} sameAs links.")
    elif len(all_same_as) > 0:
        score += 10
        findings.append(f"Basic social proof: {len(all_same_as)} sameAs links.")
    else:
        findings.append("Missing sameAs: No external authority anchoring.")

    # 3. Expertise Mapping (knowsAbout / memberOf) - 20%
    expertise_anchors = 0
    for s in schema_blocks:
        if not isinstance(s, dict): continue
        if s.get("knowsAbout") or s.get("memberOf") or s.get("alumniOf"):
            expertise_anchors += 1
            
    if expertise_anchors > 0:
        score += 20
        findings.append("Expertise/Org mapping detected (knowsAbout/memberOf).")
    else:
        findings.append("Missing Expertise anchors (knowsAbout). Identity lacks context.")

    return min(score, 100), findings, found_types

# --- CORE LOGIC: REMEDIATION & CROSS-REFERENCE ---
def generate_remediations(schema_blocks):
    """Identifies missing fields in existing schema based on the registry."""
    remediations = []
    found_types = set()
    
    for s in schema_blocks:
        if not isinstance(s, dict): continue
        stype = s.get("@type")
        if stype in SCHEMA_REGISTRY:
            found_types.add(stype)
            missing = [f for f in SCHEMA_REGISTRY[stype] if f not in s]
            if missing:
                remediations.append({
                    "type": stype,
                    "action": "ADD_FIELDS",
                    "fields": missing,
                    "reason": f"Required for Google {stype} RICH SNIPPET compliance."
                })
    
    if not found_types:
        remediations.append({
            "type": "GLOBAL",
            "action": "CREATE_SCHEMA",
            "suggestion": "No supported schema types found. Recommend starting with 'Organization' or 'Person' schema.",
            "fields": SCHEMA_REGISTRY["Organization"]
        })
        
    return remediations

def get_latest_css_score(session_id):
    """Retrieves the latest CSS score from worker-tracker's firestore logs."""
    try:
        db = get_db()
        runs_ref = db.collection('historical_runs').where(filter=firestore.FieldFilter("session_id", "==", session_id))
        runs = _firestore_call_with_timeout(lambda: runs_ref.order_by("timestamp", direction=firestore.Query.DESCENDING).limit(1).get())
        
        if runs:
            data = runs[0].to_dict()
            metrics = data.get("metrics", {})
            return metrics.get("CSS") or metrics.get("citation_sentiment_score")
    except Exception as e:
        logging.warning(f"CSS Cross-Ref Error: {e}")
    return None

# --- CORE LOGIC: GHOST INJECTION ---
def generate_ghost_jwt(admin_api_key):
    """Generates a JWT for Ghost Admin API authentication."""
    try:
        import jwt
        key_id, secret = admin_api_key.split(':')
        
        iat = int(datetime.datetime.now(datetime.timezone.utc).timestamp())
        header = {'alg': 'HS256', 'typ': 'JWT', 'kid': key_id}
        payload = {
            'iat': iat,
            'exp': iat + 5 * 60, # 5 minute expiry
            'aud': '/admin/'
        }
        
        return jwt.encode(payload, bytes.fromhex(secret), algorithm='HS256', headers=header)
    except Exception as e:
        logging.error(f"JWT Generation Error: {e}")
        return None

def inject_schema_to_ghost(post_id, schema_json, post_type="posts"):
    """Injects JSON-LD into a Ghost post or page."""
    admin_api_key = get_secret("ghost-admin-api-key")
    ghost_url = get_secret("ghost-url")
    
    if not admin_api_key or not ghost_url:
        raise ValueError("Missing Ghost Admin credentials.")

    token = generate_ghost_jwt(admin_api_key)
    headers = {'Authorization': f'Ghost {token}'}
    
    # 1. Fetch current document to preserve existing code injection
    url = f"{ghost_url.rstrip('/')}/ghost/api/admin/{post_type}/{post_id}/"
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    
    post_data = resp.json()[post_type][0]
    current_head = post_data.get('codeinjection_head') or ""
    
    # 2. Cleanup & Merge
    # Remove any existing [ADK_REMEDIATION] blocks to avoid duplicates
    clean_head = re.sub(r'<!-- \[ADK_REMEDIATION_START\] -->.*?<!-- \[ADK_REMEDIATION_END\] -->', '', current_head, flags=re.DOTALL)
    
    new_block = f"""
        <!-- [ADK_REMEDIATION_START] -->
        <script type="application/ld+json">
        {json.dumps(schema_json, indent=2)}
        </script>
        <!-- [ADK_REMEDIATION_END] -->
        """
    updated_head = clean_head.strip() + "\n" + new_block
    
    # 3. Update the document
    update_payload = {
        post_type: [{
            "id": post_id,
            "codeinjection_head": updated_head,
            "updated_at": post_data['updated_at']
        }]
    }
    
    put_resp = requests.put(url, headers=headers, json=update_payload)
    put_resp.raise_for_status()
    return True

# --- HTTP HANDLER ---
@functions_framework.http
def process_schema_logic(request):
    start_time = datetime.datetime.now(datetime.timezone.utc)
    try:
        req = request.get_json(silent=True)
        if not req: return "No payload", 400
        
        target_url = req.get("site_url")
        session_id = req.get("session_id")
        mode = req.get("mode", "validate")
        ghost_post_id = req.get("ghost_post_id")
        ghost_post_type = req.get("ghost_post_type", "posts") # posts or pages
        
        if not target_url: return "Missing site_url", 400

        print(f"TELEMETRY: [MISSION START] {start_time.strftime('%Y-%m-%d %H:%M:%S')} | Function: process_schema_logic")
        logging.info(f"TELEMETRY: Schema Analysis Started: {target_url} [Mode: {mode}] [Session: {session_id}]")
        
        # 1. Extract
        blocks = extract_on_page_schema(target_url)
        logging.info(f"TELEMETRY: Extraction complete. Found {len(blocks)} JSON-LD blocks.")
        
        # 2. Score & Remediate
        eds_score, findings, detected_types = calculate_eds(blocks)
        remediations = generate_remediations(blocks)
        logging.info(f"TELEMETRY: Scoring complete. EDS Score: {eds_score}")
        
        # 3. Cross-Reference CSS
        css_score = None
        priority = "NORMAL"
        if session_id:
            css_score = get_latest_css_score(session_id)
            logging.info(f"TELEMETRY: CSS Cross-Ref lookup complete. Score: {css_score}")
            if css_score is not None and css_score < 0.2 and eds_score < 60:
                priority = "HIGH"
                findings.append(f"🚩 HIGH PRIORITY: Critical Authority Gap (EDS:{eds_score}, CSS:{css_score})")
                logging.warning(f"TELEMETRY: [PRIORITY ESCALATION] High priority flagged for {target_url}")

        # 4. Injection Logic
        injection_status = "N/A"
        if mode == "inject" and ghost_post_id:
            base_remedy = {
                "@context": "https://schema.org",
                "@type": "WebPage",
                "name": findings[0] if findings else "Remediated Page",
                "url": target_url,
                "description": "Auto-remediated schema for identity resolution."
            }
            if inject_schema_to_ghost(ghost_post_id, base_remedy, ghost_post_type):
                injection_status = "SUCCESS"
                findings.append(f"✅ Injected remediated JSON-LD into Ghost {ghost_post_type} {ghost_post_id}")

        # 5. Store in Firestore (Wrapped)
        db = get_db()
        doc_id = hashlib.sha256(target_url.encode()).hexdigest()[:16]
        doc_ref = db.collection('schema_audits').document(doc_id)
        
        audit_data = {
            "url": target_url,
            "session_id": session_id,
            "eds_score": eds_score,
            "css_score": css_score,
            "detected_types": list(detected_types),
            "findings": findings,
            "remediations": remediations,
            "injection_status": injection_status,
            "priority": priority,
            "timestamp": datetime.datetime.now(datetime.timezone.utc)
        }
        
        _firestore_call_with_timeout(lambda: doc_ref.set(audit_data))
        logging.info(f"TELEMETRY: Firestore Audit stored [Doc: {doc_id}]")

        end_time = datetime.datetime.now(datetime.timezone.utc)
        duration = (end_time - start_time).total_seconds()
        print(f"TELEMETRY: [MISSION END] {end_time.strftime('%Y-%m-%d %H:%M:%S')} | Latency: {duration:.2f}s")

        return jsonify({
            "status": "success",
            "audit": audit_data,
            "blocks_found": len(blocks)
        }), 200

    except Exception as e:
        logging.error(f"TELEMETRY: Worker-Schema Critical Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
