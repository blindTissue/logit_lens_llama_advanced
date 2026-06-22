"""Tests for saved session tensor layout and metadata."""
import numpy as np

from session_tensors import (
    extract_states_from_custom_outputs,
    load_session_archive,
    build_tensor_archive,
)


def test_custom_combined_includes_final_output():
    from backends.custom_backend import CustomBackend

    backend = CustomBackend()
    backend.load_model("meta-llama/Llama-3.2-1B", device="cpu")
    result = backend.run_inference("Hi", {}, lens_type="combined")
    names = [x["layer_name"] for x in result["logit_lens"]]
    kinds = list(result["tensors"]["state_kinds"])
    assert names[-1] == "Final Output"
    assert kinds[-1] == "final_output"
    assert "lens_type" in result["tensors"]
    assert str(result["tensors"]["lens_type"].item()) == "combined"
    assert len(names) == result["tensors"]["hidden_states"].shape[0]
    backend.unload_model()


def test_custom_block_output_qwen_naming():
    from backends.custom_backend import CustomBackend

    backend = CustomBackend()
    backend.load_model("Qwen/Qwen3-0.6B", device="cpu")
    result = backend.run_inference("Hi", {}, lens_type="block_output")
    names = [x["layer_name"] for x in result["logit_lens"]]
    kinds = list(result["tensors"]["state_kinds"])
    assert names[0] == "Embeddings"
    assert "Post-Attn" not in names[1]
    assert names[1] == "L0 Block Out"
    assert kinds[1] == "block_output"
    assert names[-1] == "Final Output"
    backend.unload_model()


def test_load_legacy_session_infer_kinds():
    path = "saved_states/20251206_051515_qwen/tensors.npz"
    try:
        data, meta = load_session_archive(path)
    except FileNotFoundError:
        return
    assert meta["lens_type"] == "block_output"
    assert len(meta["state_kinds"]) == data["hidden_states"].shape[0]
