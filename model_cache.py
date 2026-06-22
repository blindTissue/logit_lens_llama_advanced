"""Check whether a Hugging Face model is available in the local cache."""
from __future__ import annotations

import os
from typing import Any, Dict

# config.json is present for any loadable Transformers / TransformerLens model
_CACHE_PROBE_FILES = ("config.json",)


def get_model_cache_status(model_name: str) -> Dict[str, Any]:
    """
    Return whether *model_name* appears to be cached locally (no Hub download needed).

    Does not contact the Hub; only inspects the local Hugging Face cache.
    """
    if os.path.isdir(model_name) or os.path.isfile(model_name):
        return {
            "cached": True,
            "model_name": model_name,
            "source": "local_path",
        }

    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return {
            "cached": False,
            "model_name": model_name,
            "source": "unknown",
            "message": "huggingface_hub not available; cannot inspect cache",
        }

    cached_files: list[str] = []
    for filename in _CACHE_PROBE_FILES:
        path = try_to_load_from_cache(model_name, filename)
        if path is not None and os.path.isfile(path):
            cached_files.append(filename)

    cached = "config.json" in cached_files
    result: Dict[str, Any] = {
        "cached": cached,
        "model_name": model_name,
        "source": "huggingface_hub_cache",
        "cached_files": cached_files,
    }
    if not cached:
        result["message"] = (
            "Model weights are not in the local Hugging Face cache. "
            "Loading will download from the Hub (gated models require HF_TOKEN)."
        )
    return result
