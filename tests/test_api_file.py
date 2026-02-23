import requests
import os

# API configuration
BASE_URL = "http://localhost:8000"
ENDPOINT = "/ner/process-file"

def test_file_upload(file_path: str, language: str = None):
    """
    Tests the /ner/process-file endpoint by uploading a local file.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    url = f"{BASE_URL}{ENDPOINT}"
    
    # Optional query parameters: language, labels
    params = {}
    if language:
        params["language"] = language

    print(f"Testing upload for: {file_path}")
    
    with open(file_path, "rb") as f:
        # Standard FastAPI/Starlette UploadFile mapping
        files = {"file": (os.path.basename(file_path), f, "application/octet-stream")}
        
        try:
            response = requests.post(url, files=files, params=params)
            response.raise_for_status()
            
            result = response.json()
            print(f"Success! Detected Language: {result.get('language')}")
            print(f"Found {len(result.get('entities', []))} entities.")
            
            for ent in result.get('entities', []):
                print(f"  - [{ent['label']}] {ent['text']} (score: {ent['score']:.2f})")
                
            print(f"Usage: {result.get('usage')}")
            
        except requests.exceptions.HTTPError as err:
            print(f"HTTP Error: {err}")
            print(f"Response: {response.text}")
        except Exception as err:
            print(f"Unexpected Error: {err}")

if __name__ == "__main__":
    # Create a small dummy file for testing if it doesn't exist
    test_txt = "test_sample.txt"
    if not os.path.exists(test_txt):
        with open(test_txt, "w") as f:
            f.write("Apple Inc. founded by Steve Jobs is located in Cupertino, California.")

    test_file_upload(test_txt)
