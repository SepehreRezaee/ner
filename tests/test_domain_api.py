import requests
import os

# API configuration
BASE_URL = "http://localhost:8000"

def test_process_text(text: str, domains: list[str] = None, labels: list[str] = None):
    url = f"{BASE_URL}/ner/process-text"
    params = {}
    if domains:
        params["domain"] = domains
    if labels:
        params["labels"] = labels
        
    print(f"\n--- Testing /ner/process-text ---")
    print(f"Text: {text[:50]}...")
    print(f"Domains: {domains}, Labels: {labels}")
    
    try:
        response = requests.post(url, json={"text": text}, params=params)
        response.raise_for_status()
        result = response.json()
        print(f"Found {len(result.get('entities', []))} entities.")
        for ent in result.get('entities', []):
            print(f"  - [{ent['label']}] {ent['text']}")
    except Exception as e:
        print(f"Error: {e}")

def test_process_file(file_path: str, domains: list[str] = None):
    url = f"{BASE_URL}/ner/process-file"
    params = {}
    if domains:
        params["domain"] = domains
        
    print(f"\n--- Testing /ner/process-file ---")
    print(f"File: {file_path}")
    print(f"Domains: {domains}")
    
    if not os.path.exists(file_path):
        print("File not found.")
        return

    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f)}
        try:
            response = requests.post(url, files=files, params=params)
            response.raise_for_status()
            result = response.json()
            print(f"Found {len(result.get('entities', []))} entities.")
            for ent in result.get('entities', []):
                print(f"  - [{ent['label']}] {ent['text']}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    # 1. Test Semantic Domain Resolution ('security' -> 'cyber')
    test_process_text("Apple is hiring a software engineer in San Francisco.", domains=["security"])
    
    # 2. Test Semantic Label Resolution ('attacker' -> 'threat_actor', 'virus' -> 'malware')
    test_process_text("APT5 group using cobalt strike malware was detected.", labels=["attacker", "virus"])
    
    # 3. Test Mixed Resolution
    test_process_text("Article 45 trial for Microsoft.", domains=["law"], labels=["human"])
    
    # 4. Test Strict Filtering (Unknown label 'alien_artifact' is discarded)
    # Since we explicitly provided labels, the system MUST NOT fall back to multi-pass.
    # Expected: 0 entities found.
    test_process_text("The UFO landed in Area 51.", labels=["alien_artifact"])

    # 5. Test Unknown Domain (Should return 0 entities, no default fallback)
    test_process_text("Apple is in Cupertino.", domains=["unknown_domain"])
