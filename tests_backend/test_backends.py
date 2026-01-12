"""
Simple test script to verify both backends work correctly.

Usage:
    python tests_backend/test_backends.py

This will load a small model with both backends and compare their outputs.
"""
import sys
import os
# Add parent directory to path so we can import backends
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from backends.custom_backend import CustomBackend
from backends.transformerlens_backend import TransformerLensBackend


def test_backend(backend, backend_name, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """Test a backend with a small model."""
    print(f"\n{'='*60}")
    print(f"Testing {backend_name} Backend")
    print(f"{'='*60}")

    # Load model
    print(f"Loading {model_name}...")
    try:
        result = backend.load_model(model_name, device="cpu")
        print(f"✓ Model loaded: {result}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return None

    # Run simple inference
    test_text = "The capital of France is"
    print(f"\nRunning inference on: '{test_text}'")

    try:
        result = backend.run_inference(
            text=test_text,
            interventions={},
            lens_type="block_output",
            apply_chat_template=False,
            return_attention=False
        )

        print(f"✓ Inference successful")
        print(f"  - Input tokens: {result['input_tokens']}")
        print(f"  - Number of layers: {len(result['logit_lens'])}")
        print(f"  - Layer names: {[layer['layer_name'] for layer in result['logit_lens']]}")

        # Show top prediction at final layer
        final_layer = result['logit_lens'][-1]
        final_token_preds = final_layer['predictions'][-1]  # Predictions for last token
        print(f"\n  Top predictions at final layer for last token:")
        for i, pred in enumerate(final_token_preds[:3]):
            print(f"    {i+1}. '{pred['token']}' (prob: {pred['prob']:.4f})")

        return result

    except Exception as e:
        print(f"✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_results(result1, result2):
    """Compare results from two backends."""
    print(f"\n{'='*60}")
    print("Comparing Results")
    print(f"{'='*60}")

    if result1 is None or result2 is None:
        print("✗ Cannot compare - one or both backends failed")
        return

    # Compare number of layers
    if len(result1['logit_lens']) == len(result2['logit_lens']):
        print(f"✓ Same number of layers: {len(result1['logit_lens'])}")
    else:
        print(f"✗ Different number of layers: {len(result1['logit_lens'])} vs {len(result2['logit_lens'])}")

    # Compare token predictions at final layer
    final_layer1 = result1['logit_lens'][-1]
    final_layer2 = result2['logit_lens'][-1]

    print("\nTop-3 predictions for last token (final layer):")
    print(f"  Custom Backend:")
    for i, pred in enumerate(final_layer1['predictions'][-1][:3]):
        print(f"    {i+1}. '{pred['token']}' (prob: {pred['prob']:.4f})")

    print(f"  TransformerLens Backend:")
    for i, pred in enumerate(final_layer2['predictions'][-1][:3]):
        print(f"    {i+1}. '{pred['token']}' (prob: {pred['prob']:.4f})")

    # Check if top-1 prediction matches
    top1_custom = final_layer1['predictions'][-1][0]['token']
    top1_tl = final_layer2['predictions'][-1][0]['token']

    if top1_custom == top1_tl:
        print(f"\n✓ Top-1 predictions match: '{top1_custom}'")
    else:
        print(f"\n⚠ Top-1 predictions differ: '{top1_custom}' vs '{top1_tl}'")
        print("  (Small differences are expected due to implementation details)")


def main():
    print("Backend Comparison Test")
    print("This will test both backends with a small model (TinyLlama 1.1B)")
    print("\nNote: This requires ~5GB RAM and may take a few minutes...")

    # Test custom backend
    custom_backend = CustomBackend()
    custom_result = test_backend(custom_backend, "Custom")

    # Test TransformerLens backend
    try:
        tl_backend = TransformerLensBackend()
        tl_result = test_backend(tl_backend, "TransformerLens")
    except ImportError as e:
        print(f"\n✗ TransformerLens not available: {e}")
        print("Install with: uv pip install transformer-lens")
        tl_result = None

    # Compare results
    if custom_result and tl_result:
        compare_results(custom_result, tl_result)

    print("\n" + "="*60)
    print("Test Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
