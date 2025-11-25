import requests
import json

def test_attention_blocking():
    # Load model
    print("Loading model...")
    requests.post("http://localhost:8000/load_model", json={"model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"})
    
    text = "The capital of France is"
    
    # Test 1: No intervention (baseline)
    print("\n=== Test 1: Baseline (no intervention) ===")
    res1 = requests.post("http://localhost:8000/inference", json={"text": text})
    baseline = res1.json()
    print(f"Layer 5 top prediction: {baseline['logit_lens'][6]['predictions'][4]['token']}")
    print(f"Layer 10 top prediction: {baseline['logit_lens'][11]['predictions'][4]['token']}")
    print(f"Layer 15 top prediction: {baseline['logit_lens'][16]['predictions'][4]['token']}")
    
    # Test 2: Block attention from token 0 ("The") to token 4 ("is") in layer 10
    print("\n=== Test 2: Block attention 0→4 in layer 10 ===")
    res2 = requests.post("http://localhost:8000/inference", json={
        "text": text,
        "interventions": {
            "layer_10_attention": {
                "type": "block_attention",
                "source_tokens": [0],
                "target_tokens": [4],
                "all_layers": False
            }
        }
    })
    single_block = res2.json()
    print(f"Layer 5 top prediction: {single_block['logit_lens'][6]['predictions'][4]['token']}")
    print(f"Layer 10 top prediction: {single_block['logit_lens'][11]['predictions'][4]['token']}")
    print(f"Layer 15 top prediction: {single_block['logit_lens'][16]['predictions'][4]['token']}")
    
    # Test 3: Block attention 0→4 across ALL layers
    print("\n=== Test 3: Block attention 0→4 in ALL layers ===")
    res3 = requests.post("http://localhost:8000/inference", json={
        "text": text,
        "interventions": {
            "all_layers_attention": {
                "type": "block_attention",
                "source_tokens": [0],
                "target_tokens": [4],
                "all_layers": True
            }
        }
    })
    all_block = res3.json()
    print(f"Layer 5 top prediction: {all_block['logit_lens'][6]['predictions'][4]['token']}")
    print(f"Layer 10 top prediction: {all_block['logit_lens'][11]['predictions'][4]['token']}")
    print(f"Layer 15 top prediction: {all_block['logit_lens'][16]['predictions'][4]['token']}")
    
    # Compare
    print("\n=== Analysis ===")
    baseline_l10 = baseline['logit_lens'][11]['predictions'][4]['token']
    single_l10 = single_block['logit_lens'][11]['predictions'][4]['token']
    all_l10 = all_block['logit_lens'][11]['predictions'][4]['token']
    
    print(f"Layer 10 changed with single-layer block: {baseline_l10 != single_l10}")
    print(f"Layer 10 changed with all-layers block: {baseline_l10 != all_l10}")
    
    baseline_l5 = baseline['logit_lens'][6]['predictions'][4]['token']
    single_l5 = single_block['logit_lens'][6]['predictions'][4]['token']
    all_l5 = all_block['logit_lens'][6]['predictions'][4]['token']
    
    print(f"Layer 5 unchanged with single-layer block (expected): {baseline_l5 == single_l5}")
    print(f"Layer 5 changed with all-layers block: {baseline_l5 != all_l5}")
    
    print("\n=== Expected behavior ===")
    print("- Single-layer block should affect layer 10 and beyond")
    print("- All-layers block should affect all layers including layer 5")
    print("- Predictions should differ when attention is blocked")

if __name__ == "__main__":
    test_attention_blocking()
