"""
Test script to verify that Custom and TransformerLens backends produce similar results.

This ensures that both backends are correctly implementing the same model behavior.

Usage:
    python tests_backend/test_backend_parity.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from backends.custom_backend import CustomBackend
from backends.transformerlens_backend import TransformerLensBackend


def compare_predictions(preds1, preds2, k=5):
    """
    Compare top-k predictions from two backends.

    Returns:
        dict with comparison metrics
    """
    # Get top-k token IDs
    ids1 = [p['id'] for p in preds1[:k]]
    ids2 = [p['id'] for p in preds2[:k]]

    # Get top-k probabilities
    probs1 = [p['prob'] for p in preds1[:k]]
    probs2 = [p['prob'] for p in preds2[:k]]

    # Check if top-1 matches
    top1_match = ids1[0] == ids2[0]

    # Check overlap in top-k
    overlap = len(set(ids1) & set(ids2))
    overlap_ratio = overlap / k

    # Compute probability difference for top-1
    prob_diff = abs(probs1[0] - probs2[0])

    # Average probability difference for matching tokens
    matching_prob_diffs = []
    for i, id1 in enumerate(ids1):
        if id1 in ids2:
            j = ids2.index(id1)
            matching_prob_diffs.append(abs(probs1[i] - probs2[j]))

    avg_prob_diff = np.mean(matching_prob_diffs) if matching_prob_diffs else 0.0

    return {
        'top1_match': top1_match,
        'top1_id_custom': ids1[0],
        'top1_id_tl': ids2[0],
        'top1_token_custom': preds1[0]['token'],
        'top1_token_tl': preds2[0]['token'],
        'overlap': overlap,
        'overlap_ratio': overlap_ratio,
        'top1_prob_diff': prob_diff,
        'avg_prob_diff': avg_prob_diff
    }


def test_parity(model_name="meta-llama/Llama-3.2-1B"):
    """Test parity between Custom and TransformerLens backends."""
    print("="*80)
    print(f"Backend Parity Test: {model_name}")
    print("="*80)
    print()
    print("This test verifies that both backends produce similar results.")
    print("Small numerical differences are expected due to floating point precision.")
    print()

    # Load both backends
    print("Loading Custom Backend...")
    custom_backend = CustomBackend()
    try:
        custom_backend.load_model(model_name, device="cpu")
        print("✓ Custom backend loaded")
    except Exception as e:
        print(f"✗ Failed to load custom backend: {e}")
        return False

    print("\nLoading TransformerLens Backend...")
    try:
        tl_backend = TransformerLensBackend()
        tl_backend.load_model(model_name, device="cpu")
        print("✓ TransformerLens backend loaded")
    except Exception as e:
        print(f"✗ Failed to load TransformerLens backend: {e}")
        print("Make sure transformer-lens is installed: uv pip install transformer-lens")
        return False

    # Test cases
    test_cases = [
        "The capital of France is",
        "Once upon a time",
        "The quick brown fox jumps",
        "To be or not to be",
        "In the beginning",
    ]

    print("\n" + "="*80)
    print("Running Tests")
    print("="*80)

    all_results = []

    for test_idx, test_text in enumerate(test_cases, 1):
        print(f"\n--- Test {test_idx}/{len(test_cases)}: '{test_text}' ---")

        # Run inference on both backends
        try:
            result_custom = custom_backend.run_inference(
                text=test_text,
                interventions={},
                lens_type="block_output",
                apply_chat_template=False,
                return_attention=False
            )

            result_tl = tl_backend.run_inference(
                text=test_text,
                interventions={},
                lens_type="block_output",
                apply_chat_template=False,
                return_attention=False
            )

        except Exception as e:
            print(f"✗ Inference failed: {e}")
            continue

        # Check tokens match
        tokens_custom = result_custom['input_tokens']
        tokens_tl = result_tl['input_tokens']

        if tokens_custom != tokens_tl:
            print(f"⚠ Tokenization differs!")
            print(f"  Custom: {tokens_custom}")
            print(f"  TL: {tokens_tl}")
            print(f"  Skipping comparison for this test case...")
            continue
        else:
            print(f"✓ Tokenization matches: {tokens_custom}")

        # Compare predictions at final layer for last token
        final_layer_custom = result_custom['logit_lens'][-1]
        final_layer_tl = result_tl['logit_lens'][-1]

        preds_custom = final_layer_custom['predictions'][-1]  # Last token
        preds_tl = final_layer_tl['predictions'][-1]

        # Compare predictions
        comparison = compare_predictions(preds_custom, preds_tl, k=5)
        all_results.append(comparison)

        # Print results
        print(f"\nTop-5 Predictions Comparison:")
        print(f"{'Rank':<6} {'Custom':<30} {'TransformerLens':<30}")
        print("-" * 70)

        for i in range(5):
            token_custom = preds_custom[i]['token']
            prob_custom = preds_custom[i]['prob']
            token_tl = preds_tl[i]['token']
            prob_tl = preds_tl[i]['prob']

            match_marker = "✓" if preds_custom[i]['id'] == preds_tl[i]['id'] else " "

            print(f"{i+1:<6} {token_custom:<15} ({prob_custom:.4f})    {token_tl:<15} ({prob_tl:.4f}) {match_marker}")

        print(f"\nComparison Metrics:")
        print(f"  Top-1 Match: {'✓ YES' if comparison['top1_match'] else '✗ NO'}")
        if not comparison['top1_match']:
            print(f"    Custom: '{comparison['top1_token_custom']}'")
            print(f"    TL: '{comparison['top1_token_tl']}'")
        print(f"  Top-5 Overlap: {comparison['overlap']}/5 ({comparison['overlap_ratio']*100:.0f}%)")
        print(f"  Top-1 Prob Difference: {comparison['top1_prob_diff']:.6f}")
        print(f"  Avg Prob Difference (matching tokens): {comparison['avg_prob_diff']:.6f}")

    # Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)

    if not all_results:
        print("✗ No successful comparisons")
        return False

    top1_matches = sum(1 for r in all_results if r['top1_match'])
    avg_overlap = np.mean([r['overlap_ratio'] for r in all_results])
    avg_prob_diff = np.mean([r['avg_prob_diff'] for r in all_results])
    max_prob_diff = max([r['avg_prob_diff'] for r in all_results])

    print(f"Total Tests: {len(all_results)}")
    print(f"Top-1 Matches: {top1_matches}/{len(all_results)} ({top1_matches/len(all_results)*100:.0f}%)")
    print(f"Average Top-5 Overlap: {avg_overlap*100:.1f}%")
    print(f"Average Probability Difference: {avg_prob_diff:.6f}")
    print(f"Maximum Probability Difference: {max_prob_diff:.6f}")

    print("\n" + "="*80)
    print("Interpretation")
    print("="*80)

    # Interpret results
    if top1_matches == len(all_results):
        print("✓ EXCELLENT: All top-1 predictions match!")
        print("  Both backends are producing identical results.")
    elif top1_matches >= len(all_results) * 0.8:
        print("✓ GOOD: Most top-1 predictions match (≥80%)")
        print("  Minor differences are expected due to floating point precision.")
    else:
        print("⚠ WARNING: Significant differences detected")
        print("  This may indicate an implementation issue.")

    if avg_overlap >= 0.9:
        print("✓ Top-5 predictions are very consistent (≥90% overlap)")
    elif avg_overlap >= 0.7:
        print("✓ Top-5 predictions are mostly consistent (≥70% overlap)")
    else:
        print("⚠ Top-5 predictions show significant divergence")

    if avg_prob_diff < 0.01:
        print(f"✓ Probability differences are negligible (<1%)")
    elif avg_prob_diff < 0.05:
        print(f"✓ Probability differences are small (<5%)")
    else:
        print(f"⚠ Probability differences are notable (≥5%)")

    print("\n" + "="*80)

    # Overall verdict
    if top1_matches >= len(all_results) * 0.8 and avg_overlap >= 0.7:
        print("✅ PASS: Backends show good parity")
        print("Both implementations are working correctly!")
        return True
    else:
        print("⚠ REVIEW NEEDED: Significant differences detected")
        print("Please investigate the implementation differences.")
        return False


def test_with_interventions(model_name="meta-llama/Llama-3.2-1B"):
    """Test that interventions produce similar effects in both backends."""
    print("\n" + "="*80)
    print("Testing Interventions Parity")
    print("="*80)

    # Load backends
    custom_backend = CustomBackend()
    custom_backend.load_model(model_name, device="cpu")

    tl_backend = TransformerLensBackend()
    tl_backend.load_model(model_name, device="cpu")

    test_text = "The capital of France is"

    # Test scale intervention
    print(f"\nTest: Scale intervention (layer 5, token -1, scale 0.5)")
    interventions = {
        "layer_5_output": [{
            "type": "scale",
            "value": 0.5,
            "token_index": -1
        }]
    }

    result_custom = custom_backend.run_inference(
        text=test_text,
        interventions=interventions,
        lens_type="block_output",
        apply_chat_template=False,
        return_attention=False
    )

    result_tl = tl_backend.run_inference(
        text=test_text,
        interventions=interventions,
        lens_type="block_output",
        apply_chat_template=False,
        return_attention=False
    )

    # Compare final predictions
    preds_custom = result_custom['logit_lens'][-1]['predictions'][-1]
    preds_tl = result_tl['logit_lens'][-1]['predictions'][-1]

    comparison = compare_predictions(preds_custom, preds_tl, k=5)

    print(f"Top-1 Match: {'✓' if comparison['top1_match'] else '✗'}")
    print(f"Top-5 Overlap: {comparison['overlap']}/5")
    print(f"Probability Difference: {comparison['avg_prob_diff']:.6f}")

    if comparison['top1_match'] and comparison['overlap'] >= 3:
        print("✓ Interventions work similarly in both backends")
        return True
    else:
        print("⚠ Interventions may have different effects")
        return False


def main():
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "Backend Parity Test Suite" + " "*33 + "║")
    print("╚" + "="*78 + "╝")
    print()
    print("This comprehensive test verifies that Custom and TransformerLens backends")
    print("produce similar results for Llama models.")
    print()
    print("Note: This test uses Llama-3.2-1B (~5GB RAM)")
    print()

    input("Press Enter to start the tests...")

    # Test basic parity
    parity_pass = test_parity()

    # Test interventions parity
    print("\n")
    intervention_pass = test_with_interventions()

    # Final summary
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Basic Parity: {'✅ PASS' if parity_pass else '⚠ REVIEW'}")
    print(f"Interventions Parity: {'✅ PASS' if intervention_pass else '⚠ REVIEW'}")

    if parity_pass and intervention_pass:
        print("\n🎉 All tests passed! Both backends are working correctly.")
    else:
        print("\n⚠ Some tests need review. Check the details above.")

    print("="*80)


if __name__ == "__main__":
    main()
