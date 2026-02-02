"""
Batch Ingestion Script for Compliance Documents
Processes all 9 regulatory PDFs with proper metadata
Secured with API key authentication
"""

import requests
import time
import os
import sys

# Cloud Function endpoint (update after deployment)
ENDPOINT = "https://ingest-compliance-docs-gjkm6kxlea-uc.a.run.app"

# Security: API key from environment variable or command-line argument
API_KEY = os.environ.get("INGESTION_API_KEY")
if not API_KEY and len(sys.argv) > 1:
    API_KEY = sys.argv[1]
    print("Using API key from command-line argument")
if not API_KEY:
    raise ValueError("INGESTION_API_KEY environment variable not set and no command-line argument provided")

COMPLIANCE_DOCS = [
    {
        "doc_url": "https://assets.kpmg.com/content/dam/kpmg/ng/pdf/nigeria-data-protection-act2023_kpmg-review.pdf",
        "doc_type": "ndpa",
        "version": "2023-official",
        "geo_scope": ["NG"],
        "industry_scope": ["all"]
    },
    {
        "doc_url": "https://cert.gov.ng/ngcert/resources/CyberCrime__Prohibition_Prevention_etc__Act__2024.pdf",
        "doc_type": "ng_cybercrimes",
        "version": "2024-official",
        "geo_scope": ["NG"],
        "industry_scope": ["all"]
    },
    {
        "doc_url": "https://www.epsu.org/sites/default/files/article/files/GDPR_FINAL_EPSU.pdf",
        "doc_type": "gdpr",
        "version": "2016-679",
        "geo_scope": ["EU", "UK"],
        "industry_scope": ["all"]
    },
    {
        "doc_url": "https://assets.kpmg.com/content/dam/kpmgsites/in/pdf/2024/08/payment-card-industry-data-security-standard-version-4.0.1.pdf.coredownload.inline.pdf",
        "doc_type": "pci_dss",
        "version": "4.0.1",
        "geo_scope": ["global"],
        "industry_scope": ["finance", "retail"]
    },
    {
        "doc_url": "https://www.blazeinfosec.com/wp-content/uploads/2024/02/soc2-pentest-guide-ebook.pdf",
        "doc_type": "soc2",
        "version": "2024-guide",
        "geo_scope": ["global"],
        "industry_scope": ["saas", "tech"]
    },
    {
        "doc_url": "https://www.maine.edu/general-counsel/wp-content/uploads/sites/49/2019/12/hipaa.pdf",
        "doc_type": "hipaa",
        "version": "1996-amended",
        "geo_scope": ["US"],
        "industry_scope": ["healthcare"]
    },
    {
        "doc_url": "https://studentprivacy.ed.gov/sites/default/files/resource_document/file/An%20Eligible%20Student%20Guide%20to%20FERPA_0.pdf",
        "doc_type": "ferpa_student",
        "version": "2024-guide",
        "geo_scope": ["US"],
        "industry_scope": ["education"]
    },
    {
        "doc_url": "https://studentprivacy.ed.gov/sites/default/files/resource_document/file/A%20parent%20guide%20to%20ferpa_508.pdf",
        "doc_type": "ferpa_parent",
        "version": "2024-guide",
        "geo_scope": ["US"],
        "industry_scope": ["education"]
    },
    {
        "doc_url": "https://cppa.ca.gov/regulations/pdf/cppa_regs.pdf",
        "doc_type": "ccpa",
        "version": "2023-regulations",
        "geo_scope": ["US"],
        "industry_scope": ["all"]
    }
]

def ingest_all():
    """Ingest all compliance documents with progress tracking"""
    print("=" * 60)
    print("COMPLIANCE DOCUMENT BATCH INGESTION")
    print("=" * 60)
    print(f"Total documents: {len(COMPLIANCE_DOCS)}\n")
    
    results = []
    
    for idx, doc in enumerate(COMPLIANCE_DOCS, 1):
        print(f"[{idx}/{len(COMPLIANCE_DOCS)}] Ingesting {doc['doc_type']}...")
        print(f"  URL: {doc['doc_url'][:80]}...")
        
        try:
            headers = {"Authorization": f"Bearer {API_KEY}"}
            response = requests.post(ENDPOINT, json=doc, headers=headers, timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                print(f"  ✅ SUCCESS: {result['msg']}")
                print(f"  📦 Chunks: {result.get('chunks_count', 'N/A')}")
                print(f"  🔗 Source: {result.get('doc_source', 'N/A')[:60]}...")
                results.append({"doc_type": doc['doc_type'], "status": "success", "chunks": result.get('chunks_count')})
            else:
                print(f"  ❌ FAILED: {response.status_code} - {response.text}")
                results.append({"doc_type": doc['doc_type'], "status": "failed", "error": response.text})
        
        except Exception as e:
            print(f"  ❌ ERROR: {str(e)}")
            results.append({"doc_type": doc['doc_type'], "status": "error", "error": str(e)})
        
        print("  " + "-" * 56)
        
        # Rate limiting: wait 5 seconds between requests
        if idx < len(COMPLIANCE_DOCS):
            print(f"  ⏳ Waiting 5 seconds before next document...\n")
            time.sleep(5)
    
    # Summary
    print("\n" + "=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)
    
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] != 'success']
    total_chunks = sum(r.get('chunks', 0) for r in successful)
    
    print(f"✅ Successful: {len(successful)}/{len(COMPLIANCE_DOCS)}")
    print(f"❌ Failed: {len(failed)}/{len(COMPLIANCE_DOCS)}")
    print(f"📦 Total chunks ingested: {total_chunks}")
    
    if failed:
        print("\nFailed documents:")
        for r in failed:
            print(f"  - {r['doc_type']}: {r.get('error', 'Unknown error')}")
    
    print("=" * 60)

if __name__ == "__main__":
    ingest_all()
