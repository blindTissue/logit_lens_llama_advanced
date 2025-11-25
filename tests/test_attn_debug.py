import requests
import json

def test_attention_blocking_debug():
    # Load model
    print("Loading model...")
    requests.post("http://localhost:8000/load_model", json={"model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"})
    
    text = "Hello world"
    
    # Test with attention blocking
    print("\n=== Testing attention blocking ===")
    print(f"Text: '{text}'")
    print(f"Blocking attention from token 0 ('Hello') to token 1 ('world')")
    
    res = requests.post("http://localhost:8000/inference", json={
        "text": text,
        "interventions": {
            "layer_5_attention": {
                "type": "block_attention",
                "source_tokens": [0],
                "target_tokens": [1],
                "all_layers": False
            }
        }
    })
    
    if res.status_code == 200:
        data = res.json()
        print(f"✓ Request successful, got {len(data['logit_lens'])} layers")
    else:
        print(f"✗ Request failed: {res.status_code}")
        print(res.text)

if __name__ == "__main__":
    test_attention_blocking_debug()
