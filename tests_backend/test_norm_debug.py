"""
Debug script to identify the exact source of probability differences.

This script compares:
1. Hidden states before normalization
2. Hidden states after normalization
3. Final logits
4. Final probabilities

Between Custom and TransformerLens backends.

Usage:
    python tests_backend/test_norm_debug.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
from backends.custom_backend import CustomBackend
from backends.transformerlens_backend import TransformerLensBackend


def compare_tensors(t1, t2, name):
    """Compare two tensors and report differences."""
    print(f"\n{name}:")
    print(f"  Shape: {t1.shape} vs {t2.shape}")

    if t1.shape != t2.shape:
        print(f"  ⚠ Shape mismatch!")
        return

    abs_diff = torch.abs(t1 - t2)
    rel_diff = abs_diff / (torch.abs(t1) + 1e-8)

    print(f"  Max absolute difference: {abs_diff.max().item():.6f}")
    print(f"  Mean absolute difference: {abs_diff.mean().item():.6f}")
    print(f"  Max relative difference: {rel_diff.max().item():.6f}")
    print(f"  Mean relative difference: {rel_diff.mean().item():.6f}")


def debug_normalization(model_name="meta-llama/Llama-3.2-1B"):
    """Debug normalization differences."""
    print("="*80)
    print("Normalization Debug Test")
    print("="*80)

    # Load backends
    print("\nLoading Custom Backend...")
    custom = CustomBackend()
    custom.load_model(model_name, device="cpu")

    print("Loading TransformerLens Backend...")
    tl = TransformerLensBackend()
    tl.load_model(model_name, device="cpu")

    test_text = "The capital of France is"

    print(f"\nTest text: '{test_text}'")
    print("\nRunning inference on both backends...")

    # Run inference
    result_custom = custom.run_inference(
        text=test_text,
        interventions={},
        lens_type="block_output",
        apply_chat_template=False,
        return_attention=False
    )

    result_tl = tl.run_inference(
        text=test_text,
        interventions={},
        lens_type="block_output",
        apply_chat_template=False,
        return_attention=False
    )

    print("\n" + "="*80)
    print("Comparing Final Layer Processing")
    print("="*80)

    # Get the hidden states before final norm
    # Custom: outputs["hidden_states"][-1] is AFTER norm (line 298 in model.py)
    # We need the state BEFORE norm

    # Let's manually compute the logit lens for the final layer

    # Get the pre-norm hidden state for custom backend
    # This is tricky - the custom backend's outputs["hidden_states"][-1] is already normalized
    # We need to get the state before line 298 in model.py

    print("\nManually computing logits for final layer...")

    # Get tokenizer and create input_ids
    tokenizer = custom.tokenizer
    input_ids = tokenizer.encode(test_text, return_tensors="pt").to(
        custom.model.embed_tokens.weight.device
    )

    # Run model forward pass and capture intermediate states
    print("\nRunning custom model forward pass...")
    with torch.no_grad():
        outputs_custom = custom.model(
            input_ids,
            interventions={},
            attention_masks={},
            output_attentions=False
        )

    # Get the state BEFORE final norm (should be outputs["hidden_states"][-2] then through last layer)
    # Actually, looking at model.py line 269-301:
    # - hidden_states after last layer is added to all_hidden_states (line 271, 301)
    # - Then norm is applied (line 298)
    # - Then it's added again to all_hidden_states (line 301)
    #
    # So all_hidden_states has:
    # [embeddings, layer_0_out, layer_1_out, ..., layer_N-1_out, final_normed]

    state_before_norm_custom = outputs_custom["hidden_states"][-2]  # Last layer output before norm
    state_after_norm_custom = outputs_custom["hidden_states"][-1]   # After norm

    print(f"Custom - State before norm shape: {state_before_norm_custom.shape}")
    print(f"Custom - State after norm shape: {state_after_norm_custom.shape}")

    # Manually apply norm
    manually_normed_custom = custom.model.norm(state_before_norm_custom)

    print("\nComparing custom model's norm application:")
    compare_tensors(state_after_norm_custom, manually_normed_custom, "Custom norm consistency")

    # Get logits
    logits_custom = custom.model.lm_head(state_after_norm_custom)

    # Now for TransformerLens
    print("\n" + "-"*80)
    print("\nRunning TransformerLens model forward pass...")

    # TransformerLens uses run_with_cache
    with torch.no_grad():
        _, cache = tl.model.run_with_cache(input_ids)

    # Get the final residual state (before norm)
    # blocks.{N-1}.hook_resid_post is the last layer output
    num_layers = tl.model.cfg.n_layers
    state_before_norm_tl = cache[f"blocks.{num_layers-1}.hook_resid_post"]

    print(f"TL - State before norm shape: {state_before_norm_tl.shape}")

    # Apply norm
    state_after_norm_tl = tl.model.ln_final(state_before_norm_tl)

    print(f"TL - State after norm shape: {state_after_norm_tl.shape}")

    # Get logits
    logits_tl = tl.model.unembed(state_after_norm_tl)

    # Compare hidden states BEFORE norm
    print("\n" + "="*80)
    print("COMPARISON: Hidden States BEFORE Normalization")
    print("="*80)
    compare_tensors(
        state_before_norm_custom.float(),
        state_before_norm_tl.float(),
        "Hidden states (pre-norm)"
    )

    # Compare hidden states AFTER norm
    print("\n" + "="*80)
    print("COMPARISON: Hidden States AFTER Normalization")
    print("="*80)
    compare_tensors(
        state_after_norm_custom.float(),
        state_after_norm_tl.float(),
        "Hidden states (post-norm)"
    )

    # Compare logits
    print("\n" + "="*80)
    print("COMPARISON: Final Logits")
    print("="*80)
    compare_tensors(
        logits_custom.float(),
        logits_tl.float(),
        "Final logits"
    )

    # Compare probabilities for last token
    print("\n" + "="*80)
    print("COMPARISON: Probabilities (last token)")
    print("="*80)

    probs_custom = F.softmax(logits_custom[0, -1], dim=-1)
    probs_tl = F.softmax(logits_tl[0, -1], dim=-1)

    # Get top-5 predictions
    top5_custom_probs, top5_custom_ids = torch.topk(probs_custom, 5)
    top5_tl_probs, top5_tl_ids = torch.topk(probs_tl, 5)

    print("\nTop-5 Predictions:")
    print(f"{'Rank':<6} {'Custom':<40} {'TransformerLens':<40}")
    print("-" * 90)

    for i in range(5):
        token_custom = tokenizer.decode([top5_custom_ids[i].item()])
        prob_custom = top5_custom_probs[i].item()
        token_tl = tokenizer.decode([top5_tl_ids[i].item()])
        prob_tl = top5_tl_probs[i].item()

        match = "✓" if top5_custom_ids[i] == top5_tl_ids[i] else " "

        print(f"{i+1:<6} {token_custom:<15} ({prob_custom:.6f})         {token_tl:<15} ({prob_tl:.6f}) {match}")

    # Check if "Paris" probabilities match
    paris_token_ids = tokenizer.encode(" Paris", add_special_tokens=False)
    if len(paris_token_ids) > 0:
        paris_id = paris_token_ids[0]
        paris_prob_custom = probs_custom[paris_id].item()
        paris_prob_tl = probs_tl[paris_id].item()

        print(f"\nParis (token {paris_id}) probability:")
        print(f"  Custom: {paris_prob_custom:.6f}")
        print(f"  TL:     {paris_prob_tl:.6f}")
        print(f"  Diff:   {abs(paris_prob_custom - paris_prob_tl):.6f} ({abs(paris_prob_custom - paris_prob_tl)/paris_prob_custom*100:.2f}%)")

    # Note: TransformerLens RMSNorm stores weights differently
    # Skipping direct parameter comparison as it's implementation-specific

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\nThis test reveals where the probability differences originate:")
    print("1. If pre-norm states differ → Issue in transformer layers")
    print("2. If post-norm states differ → Issue in normalization layer")
    print("3. If logits differ → Issue in LM head or weight loading")
    print("4. If only probabilities differ → Numerical precision in softmax")
    print("="*80)


if __name__ == "__main__":
    debug_normalization()
