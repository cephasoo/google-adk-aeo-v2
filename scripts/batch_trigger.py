import requests
import time
import json

# --- CONFIGURATION ---
# 1. Get this URL from your Google Cloud Console (Cloud Run -> dispatcher-story)
DISPATCHER_URL = "https://start-story-workflow-gjkm6kxlea-uc.a.run.app" 

# 2. Mock Slack Context (Since we are bypassing Slack, we provide dummy IDs)
# This ensures the logs don't crash when trying to access 'channel' or 'ts'
MOCK_SLACK_CONTEXT = {
    "channel": "BATCH_TRIGGER_CHANNEL",
    "ts": "0000000000.000000",
    "thread_ts": "0000000000.000000"
}

# 3. Your pSEO Topics List
# Format: "Strategy" + "Context"
topics = [
    # Example 1: B2B SaaS Niche
    "Implementing Semantic SEO for B2B SaaS Startups",
    
    # Example 2: Local Service Niche
    "The Impact of AI Overviews on Local Real Estate SEO",
    
    # Example 3: E-commerce Niche
    "Optimizing Shopify Stores for Voice Search in 2025",
    
    # Example 4: Non-Profit Niche
    "Data Privacy Strategies for Non-Profit Donor Management"
]

def trigger_pseo_workflow():
    print(f"🚀 Starting Batch Dispatch to: {DISPATCHER_URL}")
    print(f"📦 Total Topics: {len(topics)}\n")

    for i, topic in enumerate(topics):
        print(f"[{i+1}/{len(topics)}] Processing: {topic}")
        
        # Construct the payload exactly as 'dispatcher-story' expects it
        # We explicitly ask for a "pSEO Article" to trigger the new Intent we just built
        payload = {
            "topic": f"Draft a pSEO article about '{topic}'. Focus on actionable implementation.",
            "slack_channel": MOCK_SLACK_CONTEXT["channel"],
            "slack_ts": MOCK_SLACK_CONTEXT["ts"],
            "slack_thread_ts": MOCK_SLACK_CONTEXT["thread_ts"]
        }
        
        try:
            # Send POST request
            response = requests.post(DISPATCHER_URL, json=payload)
            
            if response.status_code in [200, 202]:
                print(f"   ✅ Success! Workflow started. (Status: {response.status_code})")
            else:
                print(f"   ❌ Failed. Status: {response.status_code}")
                print(f"   Response: {response.text}")
                
        except Exception as e:
            print(f"   🔥 Error connecting to Dispatcher: {e}")

    print("🎉 Batch Job Complete.")

if __name__ == "__main__":
    trigger_pseo_workflow()
