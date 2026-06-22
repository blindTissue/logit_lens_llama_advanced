"""
Build consistent tensor archives for saved sessions (Llama, Qwen3, TransformerLens).

``hidden_states`` in ``tensors.npz`` is the same stack used for logit lens (depends on
``lens_type``). ``state_kinds`` labels each row so notebooks can interpret layouts correctly.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

STATE_KINDS = (
    "embedding",
    "post_attention",
    "block_output",
    "final_output",
)

# Stream intervention suffixes accepted by custom Llama/Qwen models
VALID_STREAM_LOCATIONS = frozenset({"output", "attn_output", "mlp_output"})


def normalize_intervention_key(key: str) -> str:
    """Map legacy UI keys to hook names (e.g. all_layers_attn_output -> expands on apply)."""
    if key == "all_layers_attn_output":
        return "all_layers_attn_output"  # _parse_interventions already splits to layer_*_attn_output
    return key


def extract_states_from_custom_outputs(
    outputs: Dict[str, Any],
    lens_type: str,
) -> Tuple[List[torch.Tensor], List[str], List[str]]:
    """Extract logit-lens states, display names, and per-row kind tags from custom model outputs."""
    states: List[torch.Tensor] = []
    names: List[str] = []
    kinds: List[str] = []

    hidden = outputs["hidden_states"]
    post_attn = outputs["post_attention_states"]
    n_layers = len(post_attn)

    if lens_type == "post_attention":
        states.append(hidden[0])
        names.append("Embeddings")
        kinds.append("embedding")

        for i, state in enumerate(post_attn):
            states.append(state)
            names.append(f"L{i} Post-Attn")
            kinds.append("post_attention")

        states.append(hidden[-1])
        names.append("Final Output")
        kinds.append("final_output")

    elif lens_type == "combined":
        states.append(hidden[0])
        names.append("Embeddings")
        kinds.append("embedding")

        for i in range(n_layers):
            states.append(post_attn[i])
            names.append(f"L{i} Post-Attn")
            kinds.append("post_attention")

            states.append(hidden[i + 1])
            names.append(f"L{i} Block Out")
            kinds.append("block_output")

        states.append(hidden[-1])
        names.append("Final Output")
        kinds.append("final_output")

    else:  # block_output
        for i, state in enumerate(hidden):
            states.append(state)
            if i == 0:
                names.append("Embeddings")
                kinds.append("embedding")
            elif i == len(hidden) - 1:
                names.append("Final Output")
                kinds.append("final_output")
            else:
                names.append(f"L{i - 1} Block Out")
                kinds.append("block_output")

    return states, names, kinds


def extract_states_from_tl_cache(
    cache: Any,
    lens_type: str,
    n_layers: int,
) -> Tuple[List[torch.Tensor], List[str], List[str]]:
    """Extract states from a TransformerLens activation cache."""
    states: List[torch.Tensor] = []
    names: List[str] = []
    kinds: List[str] = []

    if lens_type == "block_output":
        states.append(cache["hook_embed"])
        names.append("Embeddings")
        kinds.append("embedding")

        for i in range(n_layers):
            states.append(cache[f"blocks.{i}.hook_resid_post"])
            if i == n_layers - 1:
                names.append("Final Output")
                kinds.append("final_output")
            else:
                names.append(f"L{i} Block Out")
                kinds.append("block_output")

    elif lens_type == "post_attention":
        states.append(cache["hook_embed"])
        names.append("Embeddings")
        kinds.append("embedding")

        for i in range(n_layers):
            states.append(cache[f"blocks.{i}.hook_resid_mid"])
            names.append(f"L{i} Post-Attn")
            kinds.append("post_attention")

        states.append(cache[f"blocks.{n_layers - 1}.hook_resid_post"])
        names.append("Final Output")
        kinds.append("final_output")

    elif lens_type == "combined":
        states.append(cache["hook_embed"])
        names.append("Embeddings")
        kinds.append("embedding")

        for i in range(n_layers):
            states.append(cache[f"blocks.{i}.hook_resid_mid"])
            names.append(f"L{i} Post-Attn")
            kinds.append("post_attention")

            states.append(cache[f"blocks.{i}.hook_resid_post"])
            names.append(f"L{i} Block Out")
            kinds.append("block_output")

        states.append(cache[f"blocks.{n_layers - 1}.hook_resid_post"])
        names.append("Final Output")
        kinds.append("final_output")

    else:
        raise ValueError(f"Unknown lens_type: {lens_type}")

    return states, names, kinds


def build_tensor_archive(
    *,
    states: Sequence[torch.Tensor],
    layer_names: Sequence[str],
    state_kinds: Sequence[str],
    lens_type: str,
    num_hidden_layers: int,
    logits: torch.Tensor,
    post_attention_states: Sequence[torch.Tensor] | None = None,
    attentions: Sequence[torch.Tensor] | None = None,
) -> Dict[str, np.ndarray]:
    """Build the ``tensors`` dict written to ``tensors.npz``."""
    if len(states) != len(layer_names) or len(states) != len(state_kinds):
        raise ValueError(
            f"State/name/kind length mismatch: {len(states)}, {len(layer_names)}, {len(state_kinds)}"
        )

    archive: Dict[str, np.ndarray] = {
        "hidden_states": np.stack([s.cpu().numpy() for s in states]),
        "logits": logits.cpu().numpy(),
        "layer_names": np.array(list(layer_names)),
        "state_kinds": np.array(list(state_kinds)),
        "lens_type": np.array(lens_type),
        "num_hidden_layers": np.array(num_hidden_layers),
    }

    if post_attention_states:
        archive["post_attention_states"] = np.stack(
            [s.cpu().numpy() for s in post_attention_states]
        )

    if attentions:
        archive["attentions"] = np.stack([a.cpu().numpy() for a in attentions])

    return archive


def load_session_archive(npz_path: str) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Load ``tensors.npz`` and attach inferred metadata for older sessions missing new fields.
    """
    items = np.load(npz_path)
    data = {k: items[k] for k in items.files}

    meta: Dict[str, Any] = {
        "keys": list(items.files),
        "lens_type": str(data["lens_type"].item()) if "lens_type" in data else None,
        "num_hidden_layers": int(data["num_hidden_layers"].item())
        if "num_hidden_layers" in data
        else None,
    }

    if "state_kinds" not in data:
        meta["state_kinds"] = _infer_state_kinds(
            data.get("layer_names"),
            meta.get("lens_type"),
        )
    else:
        meta["state_kinds"] = [str(k) for k in data["state_kinds"]]

    if meta["lens_type"] is None and meta["state_kinds"]:
        meta["lens_type"] = _infer_lens_type(meta["state_kinds"])

    return data, meta


def _infer_lens_type(state_kinds: List[str]) -> str:
    if "post_attention" in state_kinds and "block_output" in state_kinds:
        # Combined interleaves both; block_output-only saves use block_output kind only
        kinds = state_kinds
        has_adjacent_pair = any(
            kinds[i] == "post_attention" and kinds[i + 1] == "block_output"
            for i in range(len(kinds) - 1)
        )
        return "combined" if has_adjacent_pair else "block_output"
    if "post_attention" in state_kinds:
        return "post_attention"
    return "block_output"


def _infer_state_kinds(
    layer_names: np.ndarray | None,
    lens_type: str | None,
) -> List[str]:
    if layer_names is None:
        return []

    names = [str(n) for n in layer_names]
    kinds: List[str] = []
    for name in names:
        if name == "Embeddings":
            kinds.append("embedding")
        elif name == "Final Output":
            kinds.append("final_output")
        elif "Post-Attn" in name:
            kinds.append("post_attention")
        elif "Block Out" in name or name.startswith("Layer "):
            kinds.append("block_output")
        else:
            kinds.append("block_output")
    return kinds


def indices_by_kind(state_kinds: Sequence[str], kind: str) -> List[int]:
    return [i for i, k in enumerate(state_kinds) if k == kind]
