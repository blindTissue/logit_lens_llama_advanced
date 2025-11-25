import urllib.request
import json
import urllib.error
import time

API_URL = "http://localhost:8000"

def post_json(url, data):
    req = urllib.request.Request(
        url, 
        data=json.dumps(data).encode('utf-8'), 
        headers={'Content-Type': 'application/json'}
    )
    try:
        with urllib.request.urlopen(req) as response:
            return response.status, json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode('utf-8')

def test_memory_eviction():
    # 1. Load Model A
    model_a = "meta-llama/Llama-3.2-1B"
    print(f"Loading Model A: {model_a}")
    status, res = post_json(f"{API_URL}/load_model", {"model_name": model_a})
    if status != 200:
        print(f"Failed to load Model A: {res}")
        return
    print("Model A loaded.")

    # 2. Load Model B (should trigger eviction of A)
    model_b = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"Loading Model B: {model_b}")
    status, res = post_json(f"{API_URL}/load_model", {"model_name": model_b})
    if status != 200:
        print(f"Failed to load Model B: {res}")
        return
    print("Model B loaded. (Model A should have been evicted)")

    # 3. Reload Model A (should trigger eviction of B)
    print(f"Reloading Model A: {model_a}")
    status, res = post_json(f"{API_URL}/load_model", {"model_name": model_a})
    if status != 200:
        print(f"Failed to reload Model A: {res}")
        return
    print("Model A reloaded. (Model B should have been evicted)")

if __name__ == "__main__":
    test_memory_eviction()
