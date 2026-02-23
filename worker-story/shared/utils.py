import os
import certifi
import requests
import concurrent.futures
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from google.cloud import secretmanager
import google.auth
import litellm
from litellm import completion
import vertexai
from vertexai.generative_models import GenerativeModel, HarmCategory, HarmBlockThreshold

litellm.drop_params = True

# --- GLOBAL CONFIG & SAFETY ---
PROJECT_ID = os.environ.get("PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT")
if not PROJECT_ID:
    try:
        _, project_id_auth = google.auth.default()
        PROJECT_ID = project_id_auth
    except:
        PROJECT_ID = None

LOCATION = os.environ.get("LOCATION", "us-central1")
N8N_PROPOSAL_WEBHOOK_URL = os.environ.get("N8N_PROPOSAL_WEBHOOK_URL") 
# Keys will be fetched dynamically below to avoid gRPC deadlocks

safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

# --- N8N Payload Helpers ---
def get_n8n_operation_type(intent, original_topic="", sanitized_topic="", ghost_post_id=None):
    """
    Standardized mapper to ensure high-fidelity N8N operation types.
    Prevents hardcoding strings like 'social' when a specific triage intent is present.
    """
    if not intent: return "social"
    intent_str = str(intent).upper()
    
    # PSEO Logic: Differentiate between Create and Update for Posts vs Pages
    if intent_str in ["PSEO_ARTICLE", "PSEO_PAGE"]:
        is_page_target = (intent_str == "PSEO_PAGE") or any(kw in original_topic.lower() or kw in sanitized_topic.lower() for kw in ["pseo page", "collection page", "page template", "ghost page", "page slug"])
        if is_page_target: 
            return "pseo_page_update" if ghost_post_id else "pseo_page_create"
        else: 
            return "pseo_update" if ghost_post_id else "pseo_draft"
            
    # Standard mapping for other intents
    return intent_str.lower()

def get_output_target(intent: str) -> str:
    """
    Centralized mapping logic for target-aware formatting.
    """
    if not intent: return "MODERATOR_VIEW"
    intent_str = str(intent).upper().strip()
    if intent_str in ["PSEO_ARTICLE", "PSEO_PAGE"]:
        return "CMS_DRAFT"
    return "MODERATOR_VIEW"

secret_client = None
def get_secret(secret_id):
    """Retrieves a secret from Cloud Secret Manager with env var fallback."""
    global secret_client
    
    env_key = secret_id.upper().replace("-", "_")
    env_val = os.environ.get(env_key)
    if env_val:
        return env_val.strip()
        
    try:
        if secret_client is None:
            secret_client = secretmanager.SecretManagerServiceClient()
        
        name = f"projects/{PROJECT_ID}/secrets/{secret_id}/versions/latest"
        response = secret_client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8").strip()
    except Exception as e:
        print(f"⚠️ Secret Manager error for {secret_id}: {e}")
        return None

# --- gRPC CIRCUIT BREAKER ---
def _firestore_call_with_timeout(callable_fn, timeout_secs=20):
    """
    Executes a callable in a separate thread.
    If it hangs longer than standard timeout, it immediately abandons the thread 
    and raises TimeoutError, bypassing TSI_DATA_CORRUPTED lockups natively.
    """
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(callable_fn)
    try:
        return future.result(timeout=timeout_secs)
    except concurrent.futures.TimeoutError as exc:
        print(f"⚠️ gRPC CRITICAL TIMEOUT: Operation hung beyond {timeout_secs}s.")
        # Abandon the thread. NEVER use 'with' or wait=True, or it will deadlock the instance.
        executor.shutdown(wait=False, cancel_futures=True)
        raise TimeoutError(f"gRPC Thread Pool Timeout after {timeout_secs}s") from exc
    finally:
        # On success, clean up immediately without blocking
        executor.shutdown(wait=False)

# --- RESILIENT HTTP DELIVERY ---
def safe_n8n_delivery(payload, timeout=45):
    """Robust pure HTTP delivery to bypass gRPC failures."""
    webhook_url = N8N_PROPOSAL_WEBHOOK_URL
    if not webhook_url:
        print("⚠️ safe_n8n_delivery: No webhook URL configured.")
        return False

    session = requests.Session()
    retries = Retry(
        total=3, 
        backoff_factor=2, 
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["POST"]
    )
    session.mount('https://', HTTPAdapter(max_retries=retries))
    
    try:
        print(f"DEBUG: Attempting safe_n8n_delivery [Session: {payload.get('session_id')}]")
        response = session.post(
            webhook_url, 
            json=payload, 
            verify=certifi.where(), 
            timeout=timeout
        )
        response.raise_for_status()
        print(f"✅ safe_n8n_delivery: Success (200 OK)")
        return True
    except requests.exceptions.SSLError as e:
        print(f"❌ safe_n8n_delivery: SSL Handshake Failed: {e}")
        import time
        try:
            time.sleep(2)
            requests.post(webhook_url, json=payload, verify=certifi.where(), timeout=timeout)
            return True
        except Exception as inner_e:
            print(f"❌ safe_n8n_delivery: Emergency fallback also failed: {inner_e}")
            return False
    except Exception as e:
        print(f"⚠️ safe_n8n_delivery: Payload delivery failed: {e}")
        return False

# --- UNIFIED MODEL ADAPTER (The Brain Switch) ---
class UnifiedModel:
    """Routes requests to Vertex AI or fallback LLM providers and mitigates gRPC hangs."""
    def __init__(self, provider, model_name):
        self.provider = provider
        self.model_name = model_name
        
        if provider == "vertex_ai":
            # Per-request gRPC initialization
            if PROJECT_ID:
                vertexai.init(project=PROJECT_ID, location=LOCATION)
            else:
                vertexai.init()
            self._native_model = GenerativeModel(model_name, safety_settings=safety_settings)
            print(f"✅ Executed clean vertexai.init for {model_name}.")

    def generate_content(self, prompt, generation_config=None, max_retries=3, system_instruction=None):
        import time
        import random
        
        if self.provider == "vertex_ai":
            retries = 0
            while retries <= max_retries:
                try:
                    if generation_config is None: generation_config = {}
                    if "max_output_tokens" not in generation_config: generation_config["max_output_tokens"] = 8192
                    
                    model = GenerativeModel(
                        self.model_name, 
                        safety_settings=safety_settings,
                        system_instruction=system_instruction
                    )
                    
                    def _call_vertex():
                        return model.generate_content(prompt, generation_config=generation_config)
                    
                    # Wrap the API call in our thread circuit-breaker to prevent hanging
                    response = _firestore_call_with_timeout(_call_vertex, timeout_secs=45)
                    
                    if not response.candidates or response.candidates[0].finish_reason == 3: 
                         raise ValueError("Safety Block via FinishReason")
                    
                    return response
                except Exception as e:
                    error_msg = str(e).lower()
                    if "429" in error_msg or "resource exhausted" in error_msg:
                        if retries < max_retries:
                            wait_time = (2 ** retries) + random.uniform(0, 1)
                            print(f"⚠️ Vertex 429: Retrying in {wait_time:.2f}s... (Attempt {retries+1}/{max_retries})")
                            time.sleep(wait_time)
                            retries += 1
                            continue
                            
                    print(f"⚠️ Vertex AI Safety/gRPC Error: {e}. Attempting Specialist Failover to Claude 4.5...")
                    failover_model = UnifiedModel(provider="anthropic", model_name="claude-sonnet-4-5")
                    return failover_model.generate_content(prompt, system_instruction=system_instruction, generation_config=generation_config)

        # Universal Route (Anthropic via HTTP/LiteLLM)
        else:
            print(f"🔄 Fallback to HTTP via LiteLLM ({self.provider})...", flush=True)
            if self.provider == "anthropic":
                key = os.environ.get("ANTHROPIC_API_KEY") or get_secret("anthropic-api-key")
                if key: os.environ["ANTHROPIC_API_KEY"] = key
            elif self.provider == "openai":
                key = os.environ.get("OPENAI_API_KEY") or get_secret("openai-api-key")
                if key: os.environ["OPENAI_API_KEY"] = key
            
            temp = generation_config.get('temperature', 0.7) if generation_config else 0.7
            
            try:
                extra_headers = {}
                if "claude-sonnet-4-5" in self.model_name or "claude-3-5-sonnet" in self.model_name:
                    extra_headers = {"anthropic-beta": "max-tokens-3-5-sonnet-2024-07-15"}

                messages = []
                if system_instruction:
                    messages.append({"role": "system", "content": str(system_instruction)})
                messages.append({"role": "user", "content": prompt})

                response = completion(
                    model=self.model_name, 
                    messages=messages,
                    temperature=temp,
                    max_tokens=8192,
                    extra_headers=extra_headers
                )
                
                class MockResponse:
                    def __init__(self, content): self.text = content
                
                content = response.choices[0].message.content
                if not content:
                    content = "The model was unable to generate a response."
                return MockResponse(content)
                
            except Exception as e:
                print(f"❌ HTTP LiteLLM Error: {e}")
                class MockResponse:
                    def __init__(self, content): self.text = content
                return MockResponse("Error generating content.")
