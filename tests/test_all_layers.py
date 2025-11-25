import requests
import json

# Test all_layers functionality
def test_all_layers():
    # Load model
    print("Loading model...")
    requests.post("http://localhost:8000/load_model", json={"model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"})
    
    text = "The capital"
    
    # Test 1: No intervention (baseline)
    print("\n=== Test 1: No intervention ===")
    res1 = requests.post("http://localhost:8000/inference", json={"text": text})
    baseline = res1.json()
    print(f"Layer 5 prediction: {baseline['logit_lens'][6]['predictions'][1]['token']}")
    print(f"Layer 10 prediction: {baseline['logit_lens'][11]['predictions'][1]['token']}")
    
    # Test 2: Single layer intervention (layer 5)
    print("\n=== Test 2: Single layer (layer 5) zero intervention ===")
    res2 = requests.post("http://localhost:8000/inference", json={
        "text": text,
        "interventions": {
            "layer_5_output": {
                "type": "zero",
                "all_layers": False
            }
        }
    })
    single = res2.json()
    print(f"Layer 5 prediction: {single['logit_lens'][6]['predictions'][1]['token']}")
    print(f"Layer 10 prediction: {single['logit_lens'][11]['predictions'][1]['token']}")
    
    # Test 3: All layers intervention
    print("\n=== Test 3: All layers zero intervention ===")
    res3 = requests.post("http://localhost:8000/inference", json={
        "text": text,
        "interventions": {
            "all_layers_output": {
                "type": "zero",
                "all_layers": True
            }
        }
    })
    all_layers = res3.json()
    print(f"Layer 5 prediction: {all_layers['logit_lens'][6]['predictions'][1]['token']}")
    print(f"Layer 10 prediction: {all_layers['logit_lens'][11]['predictions'][1]['token']}")
    
    # Compare
    print("\n=== Comparison ===")
    print(f"Baseline L5 == Single L5: {baseline['logit_lens'][6]['predictions'][1]['token'] == single['logit_lens'][6]['predictions'][1]['token']}")
    print(f"Baseline L10 == Single L10: {baseline['logit_lens'][11]['predictions'][1]['token'] == single['logit_lens'][11]['predictions'][1]['token']}")
    print(f"Single L5 == All L5: {single['logit_lens'][6]['predictions'][1]['token'] == all_layers['logit_lens'][6]['predictions'][1]['token']}")
    print(f"Single L10 == All L10: {single['logit_lens'][11]['predictions'][1]['token'] == all_layers['logit_lens'][11]['predictions'][1]['token']}")
    
    print("\n=== Expected behavior ===")
    print("- Baseline should differ from interventions")
    print("- Single layer should only affect layer 5 and beyond")
    print("- All layers should affect all layers (both 5 and 10 should change)")

if __name__ == "__main__":
    test_all_layers()
