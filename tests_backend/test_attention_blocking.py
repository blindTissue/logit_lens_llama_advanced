"""
Test script to verify attention blocking works in both backends.

Usage:
    python tests_backend/test_attention_blocking.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from backends.custom_backend import CustomBackend
from backends.transformerlens_backend import TransformerLensBackend


def test_attention_blocking(backend, backend_name, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """Test attention blocking intervention."""
    print(f"\n{'='*60}")
    print(f"Testing Attention Blocking: {backend_name} Backend")
    print(f"{'='*60}")

    # Load model
    print(f"Loading {model_name}...")
    try:
        result = backend.load_model(model_name, device="cpu")
        print(f"✓ Model loaded")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return None

    test_text = "The quick brown fox"

    # Test 1: No intervention (baseline)
    print(f"\n--- Test 1: Baseline (No Intervention) ---")
    print(f"Text: '{test_text}'")

    try:
        result_baseline = backend.run_inference(
            text=test_text,
            interventions={},
            lens_type="block_output",
            apply_chat_template=False,
            return_attention=True
        )

        print(f"✓ Baseline inference successful")
        print(f"  Input tokens: {result_baseline['input_tokens']}")

        # Get attention for first layer
        if "attention" in result_baseline:
            # Attention shape: [layers][heads][seq][seq]
            layer0_attn = np.array(result_baseline["attention"][0])  # First layer
            print(f"  Attention shape: {layer0_attn.shape}")  # [heads, seq, seq]

            # Show average attention from token 0 to token 1
            avg_attn_0_to_1 = layer0_attn[:, 1, 0].mean()
            print(f"  Avg attention from token 0 → token 1: {avg_attn_0_to_1:.4f}")
        else:
            print("  ⚠ No attention data returned")

    except Exception as e:
        print(f"✗ Baseline inference failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Test 2: Block attention from token 0 to token 1 at layer 0
    print(f"\n--- Test 2: Block Attention (Token 0 → Token 1, Layer 0) ---")

    interventions = {
        "layer_0_attention": [{
            "type": "block_attention",
            "source_tokens": [0],  # Block from token 0
            "target_tokens": [1],  # To token 1
        }]
    }

    try:
        result_blocked = backend.run_inference(
            text=test_text,
            interventions=interventions,
            lens_type="block_output",
            apply_chat_template=False,
            return_attention=True
        )

        print(f"✓ Intervention inference successful")

        if "attention" in result_blocked:
            layer0_attn_blocked = np.array(result_blocked["attention"][0])

            # Check if attention from token 0 to token 1 is ~0
            avg_attn_0_to_1_blocked = layer0_attn_blocked[:, 1, 0].mean()
            print(f"  Avg attention from token 0 → token 1: {avg_attn_0_to_1_blocked:.4f}")

            # Verify blocking worked
            if avg_attn_0_to_1_blocked < 0.01:  # Should be close to 0
                print(f"  ✓ Attention successfully blocked!")
            else:
                print(f"  ⚠ Attention not fully blocked (expected ~0, got {avg_attn_0_to_1_blocked:.4f})")

            # Check that other attention patterns are not affected
            avg_attn_1_to_0 = layer0_attn_blocked[:, 0, 1].mean()
            print(f"  Avg attention from token 1 → token 0: {avg_attn_1_to_0:.4f} (should be unchanged)")

        else:
            print("  ✗ No attention data returned")

    except Exception as e:
        print(f"✗ Intervention inference failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Test 3: Block attention across all layers
    print(f"\n--- Test 3: Block Attention Across All Layers ---")

    interventions_all = {
        "all_layers_attention": [{
            "type": "block_attention",
            "source_tokens": [0],
            "target_tokens": [1],
            "all_layers": True
        }]
    }

    try:
        result_all_blocked = backend.run_inference(
            text=test_text,
            interventions=interventions_all,
            lens_type="block_output",
            apply_chat_template=False,
            return_attention=True
        )

        print(f"✓ All-layers intervention successful")

        if "attention" in result_all_blocked:
            # Check multiple layers
            num_layers = len(result_all_blocked["attention"])
            blocked_count = 0

            for layer_idx in range(min(3, num_layers)):  # Check first 3 layers
                layer_attn = np.array(result_all_blocked["attention"][layer_idx])
                avg_attn = layer_attn[:, 1, 0].mean()

                if avg_attn < 0.01:
                    blocked_count += 1

                print(f"  Layer {layer_idx}: Avg attn 0→1 = {avg_attn:.4f} {'✓' if avg_attn < 0.01 else '✗'}")

            if blocked_count >= 3:
                print(f"  ✓ Attention blocked across multiple layers!")
            else:
                print(f"  ⚠ Attention not consistently blocked across layers")

    except Exception as e:
        print(f"✗ All-layers intervention failed: {e}")
        import traceback
        traceback.print_exc()

    return True


def main():
    print("Attention Blocking Test")
    print("=" * 60)
    print("This test verifies that attention blocking works correctly")
    print("in both Custom and TransformerLens backends.")
    print()
    print("Note: Uses TinyLlama 1.1B (~2GB RAM)")

    # Test custom backend
    print("\n" + "="*60)
    print("CUSTOM BACKEND")
    print("="*60)
    custom_backend = CustomBackend()
    test_attention_blocking(custom_backend, "Custom")

    # Test TransformerLens backend
    print("\n" + "="*60)
    print("TRANSFORMERLENS BACKEND")
    print("="*60)
    try:
        tl_backend = TransformerLensBackend()
        test_attention_blocking(tl_backend, "TransformerLens")
    except ImportError as e:
        print(f"✗ TransformerLens not available: {e}")
        print("Install with: uv pip install transformer-lens")

    print("\n" + "="*60)
    print("Test Complete!")
    print("="*60)
    print()
    print("Summary:")
    print("- Attention blocking modifies attention scores to -inf")
    print("- This prevents information flow between specified tokens")
    print("- Both backends should show ~0 attention for blocked pairs")
    print("- Other attention patterns should remain unchanged")


if __name__ == "__main__":
    main()
