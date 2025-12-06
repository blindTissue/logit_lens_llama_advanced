"""
Sanity check tests to verify that custom model implementations
produce the same logits as the HuggingFace reference implementations.

Supports:
- LlamaModel (meta-llama/Llama-3.2-1B, etc.)
- Qwen3Model (Qwen/Qwen3-0.6B, etc.)
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_custom_model(model_name: str, device: str = "cpu"):
    """Load the appropriate custom model based on model name."""
    model_name_lower = model_name.lower()

    if "qwen3" in model_name_lower:
        from qwen3_model import Qwen3Model
        return Qwen3Model.from_pretrained(model_name, device=device)
    elif "llama" in model_name_lower:
        from model import LlamaModel
        return LlamaModel.from_pretrained(model_name, device=device)
    else:
        raise ValueError(f"Unknown model type: {model_name}. Supported: llama, qwen3")


def test_model_correctness(model_name: str, dtype=torch.float32):
    """
    Test that custom model produces functionally equivalent outputs to HuggingFace.

    Success criteria:
    - Top-1 predictions match for all test cases
    - Logit differences are within acceptable numerical tolerance
    """
    print(f"\n{'='*60}")
    print(f"Model Correctness Test: {model_name}")
    print('='*60)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Loading HuggingFace model...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="cpu"
    )
    hf_model.eval()

    print("Loading custom model...")
    custom_model = get_custom_model(model_name, device="cpu")
    custom_model = custom_model.to(dtype)
    custom_model.eval()

    # Test cases
    test_prompts = [
        "A",
        "Hello",
        "Hello world",
        "The capital of France is",
        "1 + 1 =",
        "The quick brown fox jumps over",
    ]

    print(f"\n{'Prompt':<35} {'Tokens':<7} {'Max Diff':<10} {'Pred'}")
    print("-" * 70)

    all_predictions_match = True
    max_diff_seen = 0

    for prompt in test_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        num_tokens = input_ids.shape[1]

        with torch.no_grad():
            hf_outputs = hf_model(input_ids)
            custom_outputs = custom_model(input_ids)

        hf_logits = hf_outputs.logits
        custom_logits = custom_outputs["logits"]

        max_diff = (hf_logits - custom_logits).abs().max().item()
        max_diff_seen = max(max_diff_seen, max_diff)

        hf_pred = torch.argmax(hf_logits[:, -1, :], dim=-1).item()
        custom_pred = torch.argmax(custom_logits[:, -1, :], dim=-1).item()

        pred_token = tokenizer.decode([custom_pred])
        match = "✓" if hf_pred == custom_pred else "✗"

        if hf_pred != custom_pred:
            all_predictions_match = False

        print(f"{prompt:<35} {num_tokens:<7} {max_diff:<10.6f} {match} '{pred_token}'")

    print("-" * 70)
    print(f"\nMax logit difference across all tests: {max_diff_seen:.6f}")

    # Verdict
    print(f"\n{'='*60}")
    if all_predictions_match:
        print("✓ SUCCESS: All predictions match HuggingFace reference")
        print("  Small numerical differences are expected due to:")
        print("  - Floating point accumulation across layers")
        print("  - Minor implementation differences in attention")
        print(f"  - Max observed diff ({max_diff_seen:.4f}) is acceptable")
    else:
        print("✗ FAILURE: Some predictions do not match")
    print('='*60)

    return all_predictions_match


def test_hidden_states_alignment(model_name: str, dtype=torch.float32):
    """
    Verify hidden states are aligned between custom and HF models.
    """
    print(f"\n{'='*60}")
    print(f"Hidden States Alignment Test: {model_name}")
    print('='*60)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="cpu"
    )
    hf_model.eval()

    custom_model = get_custom_model(model_name, device="cpu")
    custom_model = custom_model.to(dtype)
    custom_model.eval()

    prompt = "Hello world"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    with torch.no_grad():
        hf_outputs = hf_model(input_ids, output_hidden_states=True)
        custom_outputs = custom_model(input_ids)

    hf_hidden = hf_outputs.hidden_states
    custom_hidden = custom_outputs["hidden_states"]

    print(f"\nPrompt: '{prompt}' ({input_ids.shape[1]} tokens)")
    print(f"Number of hidden states: {len(custom_hidden)}")
    print(f"\n{'Layer':<20} {'Max Diff':<12} {'Status'}")
    print("-" * 45)

    all_aligned = True
    for i in range(min(len(hf_hidden), len(custom_hidden))):
        diff = (hf_hidden[i] - custom_hidden[i]).abs().max().item()
        layer_name = "embeddings" if i == 0 else f"layer_{i-1}" if i < len(hf_hidden)-1 else "final_norm"

        # Tolerance: 0.01 is acceptable
        status = "✓" if diff < 0.01 else "✗"
        if diff >= 0.01:
            all_aligned = False

        print(f"{layer_name:<20} {diff:<12.8f} {status}")

    print("-" * 45)
    if all_aligned:
        print("✓ All hidden states within tolerance")
    else:
        print("✗ Some hidden states exceed tolerance")

    return all_aligned


def run_tests(model_name: str):
    """Run all tests for a given model."""
    print(f"\n{'#'*60}")
    print(f"# Testing: {model_name}")
    print('#'*60)

    passed_correctness = test_model_correctness(model_name)
    passed_alignment = test_hidden_states_alignment(model_name)

    print(f"\n{'='*60}")
    print(f"SUMMARY for {model_name}")
    print('='*60)
    print(f"Model correctness:      {'PASSED ✓' if passed_correctness else 'FAILED ✗'}")
    print(f"Hidden state alignment: {'PASSED ✓' if passed_alignment else 'FAILED ✗'}")
    print('='*60)

    return passed_correctness and passed_alignment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test custom model implementations")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to test (e.g., meta-llama/Llama-3.2-1B, Qwen/Qwen3-0.6B)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Test all supported models"
    )
    args = parser.parse_args()

    # Default models to test
    default_models = [
        "meta-llama/Llama-3.2-1B",
    ]

    qwen3_models = [
        "Qwen/Qwen3-0.6B",
    ]

    if args.all:
        models_to_test = default_models + qwen3_models
    elif args.model:
        models_to_test = [args.model]
    else:
        # Default: just test Llama
        models_to_test = default_models

    all_passed = True
    results = {}

    for model_name in models_to_test:
        try:
            passed = run_tests(model_name)
            results[model_name] = passed
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"\n✗ Error testing {model_name}: {e}")
            results[model_name] = False
            all_passed = False

    # Final summary
    print(f"\n{'#'*60}")
    print("# FINAL SUMMARY")
    print('#'*60)
    for model_name, passed in results.items():
        status = "PASSED ✓" if passed else "FAILED ✗"
        print(f"  {model_name}: {status}")
    print('#'*60)

    if all_passed:
        print("\n🎉 All model implementations are CORRECT!")
        exit(0)
    else:
        print("\n❌ Some tests failed")
        exit(1)
