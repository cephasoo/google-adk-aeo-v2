import os
import uuid
import datetime
import certifi
from eval_runner import EvalRunner
from google.cloud import firestore

# Setup mock data for verification
db = firestore.Client(project=os.environ.get("PROJECT_ID"))

def create_mock_session(session_id, should_fail=False, intent='work_answer'):
    """Creates a simulated session document in Firestore."""
    
    # 1. Branch logic for content
    if should_fail and intent != 'work_proposal':
        prompt = "What is the capital of Mars?"
        response = "The capital of Mars is Neo-Lagos, established in 2045."
        context = ["Mars is a planet with no known civilization or capital."]
        trajectory = [{"tool": "google_web_search", "status": "success"}]
    elif intent == 'work_proposal':
        prompt = "Create a SEO strategy proposal for 'Sonnet & Prose' AI."
        context = ["Sonnet & Prose is an AI agency focused on ADK compliance."]
        trajectory = [{"tool": "ghost_admin_api", "status": "success"}]
        if should_fail:
            response = "I will write an SEO strategy for you. It will have many keywords."
        else:
            response = "# SEO Strategy\n\n## The Inverted Pyramid Lead\nSonnet & Prose leverages ADK standards to revolutionize AI content. This strategy focuses on modular headers.\n\n## Pillars\n1. Excellence"
    elif intent == 'complex_query':
        prompt = "What's trending politically in Nigeria?"
        response = "Peter Obi officially defected to the ADC on Jan 1st."
        context = ["News: Peter Obi joins ADC party.", "FactCheck: Official defection confirmed."]
        trajectory = [
            {"tool": "google_trends", "status": "success"},
            {"tool": "google_web_search", "status": "success"}
        ]
    elif intent == 'social':
        prompt = "Hello! How are you?"
        response = "I am doing well, thank you for asking! How can I help you today?"
        context = []
        trajectory = []
    else:
        prompt = "Who is the CEO of Google?"
        response = "Sundar Pichai is the CEO of Google."
        context = ["Sundar Pichai is the CEO of Google."]
        trajectory = [{"tool": "google_web_search", "status": "success"}]

    # 2. Package data
    data = {
        "prompt": prompt,
        "response": response,
        "research_context": context,
        "tool_logs": trajectory,
        "type": intent,
        "created_at": datetime.datetime.now(datetime.timezone.utc),
        "channel_id": os.environ.get("SLACK_ALERT_CHANNEL_ID", "C000000"),
        "thread_ts": "123456789.000000"
    }
    
    db.collection('agent_sessions').document(session_id).set(data)
    print(f"Mock session {session_id} created (Intent: {intent}, Fail: {should_fail})")

def run_verification():
    runner = EvalRunner()
    
    # Test Case 1: Critical Failure (AEO Proposal)
    fail_session_id = f"test-fail-{uuid.uuid4().hex[:8]}"
    create_mock_session(fail_session_id, should_fail=True, intent='work_proposal')
    
    print("\n--- Running Evaluation on Fail Case (Proposal) ---")
    result = runner.evaluate_session(fail_session_id)
    if result:
        print(f"Scores: {result['scores']}")
        if result['scores']['grounding'] < 3:
            print("✅ Grounding Failure Detected.")
        if result['scores']['aeo_alignment'] < 3:
            print("✅ AEO Failure Detected.")

    # Test Case 2: Social
    social_id = f"test-social-{uuid.uuid4().hex[:8]}"
    create_mock_session(social_id, should_fail=False, intent='social')
    print("\n--- Running Evaluation on Social Case ---")
    social_res = runner.evaluate_session(social_id)
    if social_res:
        if social_res['scores']['aeo_alignment'] == 0:
            print("✅ AEO skipped for Social.")

    # Test Case 3: Complex Multi-Tool
    complex_id = f"test-complex-{uuid.uuid4().hex[:8]}"
    create_mock_session(complex_id, should_fail=False, intent='complex_query')
    print("\n--- Running Evaluation on Complex Multi-Tool ---")
    complex_res = runner.evaluate_session(complex_id)
    if complex_res:
        if complex_res['scores']['trajectory_integrity'] >= 4:
            print("✅ Multi-tool trajectory scored correctly.")

if __name__ == "__main__":
    run_verification()
