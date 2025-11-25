import urllib.request
import json
import numpy as np

API_URL = "http://localhost:8000"

def post_json(url, data):
    req = urllib.request.Request(
        url, 
        data=json.dumps(data).encode('utf-8'), 
        headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode('utf-8'))

def get_json(url):
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read().decode('utf-8'))

def test_attention():
    # 1. Check if model is loaded
    print("Checking model status...")
    res = get_json(f"{API_URL}/model_status")
    if not res["loaded"]:
        print("Loading model...")
        post_json(f"{API_URL}/load_model", {"model_name": "meta-llama/Llama-3.2-1B"})
    
    # 2. Test Layer Mean Aggregation
    print("\nTesting Layer Mean Aggregation...")
    payload = {
        "text": "The capital of France is",
        "return_attention": True,
        "attention_aggregation": "layer_mean"
    }
    data = post_json(f"{API_URL}/inference", payload)
        
    if "attention" not in data or data["attention"] is None:
        print("FAIL: Attention data missing")
        return
        
    attn = data["attention"]
    print(f"Received attention data type: {type(attn)}")
    if isinstance(attn, list):
        print(f"Number of layers: {len(attn)}")
        if len(attn) > 0:
            layer0 = np.array(attn[0])
            print(f"Layer 0 shape: {layer0.shape}")
            # Should be [heads, seq, seq]
            if len(layer0.shape) == 3:
                print("PASS: Received full attention tensor [layers, heads, seq, seq]")
            else:
                print(f"FAIL: Expected 3 dims for layer (heads, seq, seq), got {layer0.shape}")
            
    # 3. Test All Average (Client side now, but we can check if data is same)
    # The API should ignore aggregation params now
    print("\nTesting Aggregation Params Ignored...")
    payload["attention_aggregation"] = "all_average"
    data = post_json(f"{API_URL}/inference", payload)
    attn = data["attention"]
    if isinstance(attn, list):
        # Should still be full tensor
        layer0 = np.array(attn[0])
        print(f"Layer 0 shape: {layer0.shape}")

if __name__ == "__main__":
    test_attention()
