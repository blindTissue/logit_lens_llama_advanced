"""
Debug script to test if custom backend is working correctly.

Usage:
    python tests_backend/debug_inference.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backends.custom_backend import CustomBackend

def test_custom_backend():
    print("Testing Custom Backend...")
    print("="*60)

    backend = CustomBackend()

    # Load a small model
    print("Loading TinyLlama...")
    result = backend.load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0", device="cpu")
    print(f"Load result: {result}")

    # Test simple inference
    test_text = "The capital of France is"
    print(f"\nTesting with: '{test_text}'")

    result = backend.run_inference(
        text=test_text,
        interventions={},
        lens_type="block_output",
        apply_chat_template=False,
        return_attention=False
    )

    print(f"\nInput tokens: {result['input_tokens']}")
    print(f"\nFinal layer predictions (last token):")
    final_layer = result['logit_lens'][-1]
    final_predictions = final_layer['predictions'][-1]  # Last token

    for i, pred in enumerate(final_predictions[:5]):
        print(f"  {i+1}. '{pred['token']}' (prob: {pred['prob']:.4f})")

    print("\nDoes this look correct? The top prediction should be related to 'France' or 'Paris'")

if __name__ == "__main__":
    test_custom_backend()
