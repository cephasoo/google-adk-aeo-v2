import csv
import json
import requests
import os
import sys
import argparse

# --- Configuration ---
API_URL = "https://us-central1-agent-dev-sandbox-478214.cloudfunctions.net/ingest-knowledge"

def get_api_key():
    """Security: Fetch API key from environment variable or Cloud Secret Manager."""
    # 1. Try environment variables
    api_key = os.environ.get("INGESTION_API_KEY") or os.environ.get("KNOWLEDGE_INGESTION_API_KEY")
    if api_key:
        return api_key.strip()
    
    # 2. Try Google Cloud Secret Manager via CLI
    print(" 🔑 API key not in env vars. Attempting to fetch 'ingestion-api-key' from Cloud Secret Manager...")
    try:
        import subprocess
        # Using the project ID from n8n-key.json context: agent-dev-sandbox-478214
        project_id = "agent-dev-sandbox-478214" 
        result = subprocess.check_output(
            ["gcloud", "secrets", "versions", "access", "latest", "--secret=ingestion-api-key", f"--project={project_id}"],
            shell=True,
            stderr=subprocess.DEVNULL
        )
        api_key = result.decode("utf-8").strip()
        print(f"   ✅ Successfully retrieved API key from Secret Manager. (Key length: {len(api_key)}, Starts with: {api_key[:4]}...)")
        return api_key
    except Exception as e:
        print(f"   ⚠️ Could not fetch secret: {e}")

    # 3. Fail if not found
    print("❌ Error: INGESTION_API_KEY environment variable not set and Secret Manager retrieval failed.")
    print("   Run: setx INGESTION_API_KEY \"your_key\" (Windows) or export INGESTION_API_KEY=\"your_key\" (Unix)")
    sys.exit(1)

def ingest_data(csv_path, region, industry, data_insight, session_id):
    if not os.path.exists(csv_path):
        print(f"❌ Error: CSV not found at {csv_path}")
        return

    api_key = get_api_key()
    
    print(f"Reading data from {csv_path}...")
    
    records = []
    
    try:
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            
            # Identify year columns (assuming they are 4-digit years)
            year_keys = [k for k in reader.fieldnames if k.strip().isdigit() and len(k.strip()) == 4]
            # Sort years descending to prioritize most recent data
            year_keys.sort(key=int, reverse=True)
            
            if not year_keys:
                print("⚠️ Warning: No year columns found in CSV header.")
                return

            print(f"Found year columns: {year_keys}")

            for row in reader:
                location = row.get('Country Name') or row.get('Country')
                if not location:
                    continue

                metric_val = None
                found_year = None
                
                # Find the most recent year with data
                for yk in year_keys:
                    val = row.get(yk)
                    if val and val.strip():
                        metric_val = val
                        found_year = yk
                        break
                
                if metric_val and location:
                    records.append({
                        "location": location,
                        "value": metric_val,
                        "year": found_year,
                        "region": region,
                        "industry": industry,
                        "insight_type": data_insight
                    })

    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    print(f"Found {len(records)} records. Starting ingestion for topic: {data_insight} ({region})...")

    for entry in records:
        # Create a "Technical Story" for the RAG engine to digest
        # We inject the metadata context clearly into the story for vector search retrieval
        story = f"Grounding Data: In {entry['year']}, the {entry['insight_type']} for {entry['location']} ({entry['region']}) was {entry['value']}. "
        story += f"This data point is categorized under the {entry['industry']} industry and serves as a primary driver for regional {entry['insight_type']} analysis."
        
        # FIX: Appending location to session_id to ensure unique Firestore Document IDs
        # Server uses f"{session_id}_{i}" as key. Without this, all entries overwrite each other at index 0.
        sanitized_loc = entry['location'].replace(" ", "_").replace(",", "")
        unique_session_id = f"{session_id}_{sanitized_loc}"

        payload = {
            "session_id": unique_session_id,
            "topic": f"{entry['insight_type']} | {entry['location']} | {entry['region']}",
            "story": story,
            "type": "grounding_data",
            "metadata": {
                "location": entry['location'],
                "region": entry['region'],
                "industry": entry['industry'],
                "insight": entry['insight_type'],
                "value": entry['value'],
                "year": entry['year']
            }
        }
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            print(f" -> Ingesting {entry['location']}...")
            response = requests.post(API_URL, json=payload, headers=headers)
            if response.status_code == 200:
                print(f" ✅ Success: {entry['location']}")
            else:
                print(f" ❌ Failed: {entry['location']} ({response.status_code}) - {response.text}")
        except Exception as e:
            print(f" ❌ Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Tenant ADK Knowledge Ingester")
    parser.add_argument("--csv", required=True, help="Path to the CSV file")
    parser.add_argument("--region", required=True, help="Geographical Region (e.g., Sub-Saharan Africa)")
    parser.add_argument("--industry", required=True, help="Industry Sector (e.g., Economic Data)")
    parser.add_argument("--insight", required=True, help="Type of Insight (e.g., Consumption Expenditure)")
    parser.add_argument("--session", default="pseo_grounding_mission", help="Session ID for grouping")

    args = parser.parse_args()
    
    ingest_data(args.csv, args.region, args.industry, args.insight, args.session)
