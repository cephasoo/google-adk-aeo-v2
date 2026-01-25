import os
import functions_framework
import json
import requests
import datetime
import vertexai
import certifi
from vertexai.generative_models import GenerativeModel, HarmCategory, HarmBlockThreshold
from google.cloud import firestore

# --- Configuration ---
PROJECT_ID = os.environ.get("PROJECT_ID")
LOCATION = os.environ.get("LOCATION", "us-central1")
FLASH_MODEL_NAME = os.environ.get("FLASH_MODEL_NAME", "gemini-2.0-flash")
N8N_PROPOSAL_WEBHOOK_URL = os.environ.get("N8N_PROPOSAL_WEBHOOK_URL")
SLACK_ALERT_CHANNEL_ID = os.environ.get("SLACK_ALERT_CHANNEL_ID") # Dedicated internal channel

# --- Initialize Clients ---
vertexai.init(project=PROJECT_ID, location=LOCATION)
db = firestore.Client(project=PROJECT_ID)

# --- Safety Settings (ADK/RAI Compliant) ---
safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
}

model = GenerativeModel(FLASH_MODEL_NAME, safety_settings=safety_settings)

class EvalRunner:
    def __init__(self):
        self.db = db
        self.model = model

    def evaluate_session(self, session_id):
        """
        Performs end-to-end evaluation of a single session.
        """
        print(f"EvalRunner: Starting evaluation for session {session_id}")
        
        # 1. Fetch Session Data (Targeting agent_sessions)
        session_ref = self.db.collection('agent_sessions').document(session_id)
        session_doc = session_ref.get()
        
        if not session_doc.exists:
            print(f"Error: Session {session_id} not found in agent_sessions.")
            return None

        data = session_doc.to_dict()
        
        # --- ROBUST DATA HARVESTING ---
        # 1. Prompt (Look for 'prompt' or 'topic')
        prompt = data.get('prompt') or data.get('topic', '')
        
        # 2. Extract Response, Context, and Trajectory from events subcollection if missing
        response = data.get('response', '')
        context = data.get('research_context', [])
        trajectory = data.get('tool_logs', [])
        
        # Architecture Awareness: Harvest from 'events' if parent is slim
        events_ref = session_ref.collection('events')
        events = [e.to_dict() for e in events_ref.order_by('timestamp').stream()]
        
        if not response:
            for e in reversed(events):
                if e.get('event_type') in ['agent_answer', 'agent_reply', 'final_output', 'agent_proposal']:
                    response = e.get('text') or e.get('content') or str(e.get('data', ''))
                    break
        
        if not context:
            # Collect all grounding context snippets
            context = [e.get('content', '') for e in events if 'GROUNDING_CONTENT' in str(e.get('content', ''))]
        
        if not trajectory:
            # Collect all tool usage logs
            trajectory = [e for e in events if e.get('event_type') == 'tool_call' or 'tool' in e]

        source_channel = data.get('channel_id') or data.get('slack_context', {}).get('channel', 'unknown')
        source_ts = data.get('thread_ts') or data.get('slack_context', {}).get('ts', 'unknown')
        
        # Identify Intent for Selective Scoring
        intent = data.get('type', 'unknown') # work_answer, work_proposal, social
        
        # 2. Rubric-Based Scoring (LLM-as-a-Judge)
        scores = self._calculate_scores(prompt, response, context, trajectory, intent)
        
        # 3. Efficiency Metrics
        metrics = {
            "tool_calls": len(trajectory),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }

        # 4. Persistence
        eval_data = {
            "session_id": session_id,
            "scores": scores,
            "metrics": metrics,
            "is_human_verified": False,
            "intent_evaluated": intent,
            "evaluated_at": firestore.SERVER_TIMESTAMP,
            "last_updated": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=30)
        }
        
        self.db.collection('evaluations').document(session_id).set(eval_data)
        
        # 5. Proactive Alerting
        if self._should_alert(scores, intent):
            # We send the alert to a DEDICATED internal channel, NOT the user thread.
            self._trigger_n8n_alert(session_id, scores, source_channel, source_ts)

        return eval_data

    def _calculate_scores(self, prompt, response, context, trajectory, intent):
        """
        Uses Gemini to score the output based on defined rubrics.
        AEO scoring is restricted to specific technical intents.
        """
        
        apply_aeo = intent in ['work_answer', 'work_proposal']
        
        aeo_rubric = ""
        if apply_aeo:
            aeo_rubric = "2. AEO_ALIGNMENT: Does the lead paragraph (40-60 words) follow the Inverted Pyramid? (0 if not applicable)"
        else:
            aeo_rubric = "2. AEO_ALIGNMENT: SKIP (Return 0 as this is a non-AEO session)."

        eval_prompt = f"""
        You are an expert AI Quality Auditor. Evaluate the following agent interaction based on the rubrics provided.

        SESSION TYPE: {intent}
        USER PROMPT: {prompt}
        AGENT RESPONSE: {response}
        RESEARCH CONTEXT: {context}
        TOOL TRAJECTORY: {trajectory}

        RUBRICS (Score 1-5, or 0 if N/A):
        1. GROUNDING: Does the response stick to the facts in the Research Context?
        {aeo_rubric}
        3. TRAJECTORY_INTEGRITY: Were the tool calls logical and efficient for the prompt?
        4. TONE: Does it follow the 'Sonnet & Prose' persona?

        RETURN ONLY JSON:
        {{
          "grounding": int,
          "aeo_alignment": int,
          "trajectory_integrity": int,
          "tone": int,
          "reasoning": "brief explanation"
        }}
        """
        
        try:
            res = self.model.generate_content(eval_prompt)
            # Basic JSON extraction
            content = res.text.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[0].strip()
            
            score_json = json.loads(content)
            return score_json
        except Exception as e:
            print(f"Scoring Error: {e}")
            return {"grounding": 0, "aeo_alignment": 0, "trajectory_integrity": 0, "tone": 0, "reasoning": f"Error during scoring: {str(e)}"}

    def _should_alert(self, scores, intent):
        """
        Alerting threshold logic. 
        Ignores low trajectory scores for social intents.
        """
        if intent == 'social':
            return scores.get('grounding', 5) < 3 # Only alert if social greeting is hallucinating
            
        if scores.get('grounding', 5) < 3 or scores.get('trajectory_integrity', 5) < 2:
            return True
        return False

    def _trigger_n8n_alert(self, session_id, scores, source_channel, source_ts):
        """
        Sends critical failures to a dedicated internal Slack channel.
        Does NOT thread with the user's conversation to maintain privacy.
        """
        if not N8N_PROPOSAL_WEBHOOK_URL:
            print("Warning: N8N_PROPOSAL_WEBHOOK_URL not set. Skipping alert.")
            return

        # --- Premium Slack Block Kit Construction ---
        # We pass source metadata so the Block Kit can show WHERE it happened
        blocks = self._build_slack_blocks(session_id, scores, source_channel, source_ts)

        payload = {
            "type": "eval_alert",
            "event": "CRITICAL_PERFORMANCE_FAILURE",
            "session_id": session_id,
            "channel": SLACK_ALERT_CHANNEL_ID, # PROD ALERT CHANNEL (Secret)
            "thread_ts": "",                 # Empty string is safer for JSON/Slack than null
            "blocks": blocks,
            "text": "Internal Performance Alert"
        }
        
        try:
            requests.post(N8N_PROPOSAL_WEBHOOK_URL, json=payload, timeout=10, verify=certifi.where())
            print(f"Alert triggered for session {session_id} -> Channel: {SLACK_ALERT_CHANNEL_ID}")
        except Exception as e:
            print(f"Alerting Error: {e}")

    def _build_slack_blocks(self, session_id, scores, source_channel, source_ts):
        """
        Constructs a professional Slack Block Kit payload.
        Includes metadata about the source session for developer triage.
        """
        reasoning = scores.get('reasoning', 'No reasoning provided.')
        
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "🚨 Internal Performance Alert",
                    "emoji": True
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Session ID:* `{session_id}`\n*Source Channel:* `{source_channel}`\n*Status:* Critical Failure Detected"
                }
            },
            {"type": "divider"},
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Grounding:* {scores.get('grounding', 0)}/5"},
                    {"type": "mrkdwn", "text": f"*Trajectory:* {scores.get('trajectory_integrity', 0)}/5"},
                    {"type": "mrkdwn", "text": f"*Tone:* {scores.get('tone', 0)}/5"},
                    {"type": "mrkdwn", "text": f"*AEO Alignment:* {scores.get('aeo_alignment', 0)}/5"}
                ]
            },
            {"type": "divider"},
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Judge's Reasoning:*\n> {reasoning}"
                }
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "View Evaluation", "emoji": True},
                        "url": f"https://console.cloud.google.com/firestore/databases/-default-/data/panel/evaluations/{session_id}?project={PROJECT_ID}",
                        "style": "primary"
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Session Tracker", "emoji": True},
                        "url": f"https://console.cloud.google.com/firestore/databases/-default-/data/panel/agent_sessions/{session_id}?project={PROJECT_ID}"
                    }
                ]
            }
        ]
        return blocks


# --- GCP Cloud Function Entry Point (Gen 2) ---
@functions_framework.cloud_event
def evaluate_session_trigger(cloud_event):
    """
    Cloud Function entry point for Firestore trigger (Gen 2 CloudEvent).
    Triggered when an 'agent_sessions' document is created or updated.
    """
    event_data = cloud_event.data
    
    # 1. Parse session_id from document path
    # Resource format: projects/{project}/databases/(default)/documents/agent_sessions/{sessionId}
    # CloudEvent data.value.name contains the full resource path
    resource = event_data.get('value', {}).get('name', '')
    if not resource:
        print("Error: No resource name in CloudEvent data.")
        return
    
    session_id = resource.split('/')[-1]
    
    # 2. Check status to avoid infinite loops if eval updates the same doc
    # (Here we read from 'fields' structure in Firestore Change Event)
    status = event_data.get('value', {}).get('fields', {}).get('status', {}).get('stringValue', '')
    
    if status == "completed" or "awaiting" in status:
        print(f"Trigger: Evaluating session {session_id} based on status '{status}'")
        runner = EvalRunner()
        runner.evaluate_session(session_id)
    else:
        print(f"Trigger: Skipping session {session_id} (Status: {status})")

if __name__ == "__main__":
    # Example local usage
    runner = EvalRunner()
    # runner.evaluate_session("your-session-id-here")
