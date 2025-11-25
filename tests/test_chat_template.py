import urllib.request
import json
import urllib.error

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

def test_chat_template():
    # 1. Load Instruct Model
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    print(f"Loading model: {model_name}")
    status, res = post_json(f"{API_URL}/load_model", {"model_name": model_name})
    if status != 200:
        print(f"Failed to load model: {res}")
        return

    # 2. Run Inference WITH Chat Template
    print("Running inference WITH chat template...")
    text = "What is the capital of France?"
    status, res = post_json(f"{API_URL}/inference", {
        "text": text,
        "apply_chat_template": True
    })
    
    if status == 200:
        print("Success! Response with chat template received.")
    else:
        print(f"Failed inference with chat template: {res}")

    # 3. Run Inference WITHOUT Chat Template
    print("Running inference WITHOUT chat template...")
    status, res = post_json(f"{API_URL}/inference", {
        "text": text,
        "apply_chat_template": False
    })
    
    if status == 200:
        print("Success! Response without chat template received.")
    else:
        print(f"Failed inference without chat template: {res}")

if __name__ == "__main__":
    test_chat_template()
