"""
TransformerLens backend implementation.
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from transformers import AutoTokenizer
import gc

try:
    from transformer_lens import HookedTransformer
    TRANSFORMER_LENS_AVAILABLE = True
except ImportError:
    TRANSFORMER_LENS_AVAILABLE = False
    HookedTransformer = None

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from backends.base import BaseBackend


class TransformerLensBackend(BaseBackend):
    """Backend using TransformerLens for model loading and interventions."""

    def __init__(self):
        if not TRANSFORMER_LENS_AVAILABLE:
            raise ImportError(
                "TransformerLens is not installed. "
                "Install it with: pip install transformer-lens"
            )
        self.model = None
        self.tokenizer = None
        self.model_name = None

    def load_model(self, model_name: str, device: str = "cpu") -> Dict[str, Any]:
        """Load a model using TransformerLens."""
        if self.model is not None and self.model_name == model_name:
            return {
                "status": "already_loaded",
                "backend": "transformerlens",
                "model_name": model_name
            }

        # Unload previous model if exists
        if self.model is not None:
            self.unload_model()

        try:
            print(f"Loading {model_name} with TransformerLens...")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            # Load model with TransformerLens
            # TransformerLens handles the conversion automatically
            self.model = HookedTransformer.from_pretrained(
                model_name,
                device=device,
                dtype=torch.float16 if device != "cpu" else torch.float32
            )

            self.model_name = model_name

            return {
                "status": "loaded",
                "backend": "transformerlens",
                "model_name": model_name,
                "num_layers": self.model.cfg.n_layers,
                "hidden_size": self.model.cfg.d_model,
                "num_heads": self.model.cfg.n_heads
            }

        except Exception as e:
            error_msg = str(e)
            # Provide helpful message for unsupported models
            if "not found" in error_msg and "Qwen3" in model_name:
                raise RuntimeError(
                    f"Qwen3 models are not yet supported by TransformerLens. "
                    f"Please switch to the 'Custom' backend to use Qwen3 models. "
                    f"Original error: {error_msg}"
                )
            raise RuntimeError(f"Failed to load model with TransformerLens: {error_msg}")

    def unload_model(self):
        """Unload the model and free memory."""
        if self.model is not None:
            print("Unloading TransformerLens model...")
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            self.model_name = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()

    def is_loaded(self) -> bool:
        """Check if a model is loaded."""
        return self.model is not None

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.is_loaded():
            return {"loaded": False}

        return {
            "loaded": True,
            "backend": "transformerlens",
            "model_name": self.model_name,
            "num_layers": self.model.cfg.n_layers,
            "hidden_size": self.model.cfg.d_model,
            "num_heads": self.model.cfg.n_heads,
            "vocab_size": self.model.cfg.d_vocab
        }

    def get_num_layers(self) -> int:
        """Get number of layers."""
        if not self.is_loaded():
            return 0
        return self.model.cfg.n_layers

    def get_config(self) -> Any:
        """Get model config."""
        if not self.is_loaded():
            return None
        return self.model.cfg

    def run_inference(
        self,
        text: str,
        interventions: Dict[str, List[Any]],
        lens_type: str = "block_output",
        apply_chat_template: bool = False,
        return_attention: bool = False
    ) -> Dict[str, Any]:
        """Run inference with TransformerLens."""
        if not self.is_loaded():
            raise RuntimeError("No model loaded")

        # Tokenize input
        if apply_chat_template:
            messages = [{"role": "user", "content": text}]
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                input_ids = self.tokenizer.encode(
                    prompt, return_tensors="pt", add_special_tokens=False
                ).to(self.model.cfg.device)
            except Exception as e:
                raise RuntimeError(f"Failed to apply chat template: {str(e)}")
        else:
            input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.model.cfg.device)

        seq_len = input_ids.shape[1]

        # Convert interventions to TransformerLens hooks
        hooks = self._create_hooks(interventions, seq_len)

        # Run with hooks
        with torch.no_grad():
            # Run with interventions using context manager
            if hooks:
                # Use run_with_hooks context manager
                with self.model.hooks(fwd_hooks=hooks):
                    logits, cache = self.model.run_with_cache(
                        input_ids,
                        names_filter=lambda name: True  # Cache everything
                    )
            else:
                # No interventions, just run with cache
                logits, cache = self.model.run_with_cache(
                    input_ids,
                    names_filter=lambda name: True  # Cache everything
                )

        # Extract hidden states based on lens_type
        states_to_process, layer_names = self._extract_states(cache, lens_type)

        # Compute logit lens for each state
        lens_data = self._compute_logit_lens(states_to_process, layer_names)

        # Get input tokens
        input_tokens = [self.tokenizer.decode([tid]) for tid in input_ids[0].tolist()]

        # Prepare response
        response = {
            "text": text,
            "input_tokens": input_tokens,
            "logit_lens": lens_data,
            "tensors": self._prepare_tensors(states_to_process, layer_names, cache, logits)
        }

        # Add attention if requested
        if return_attention:
            response["attention"] = self._extract_attention(cache)

        return response

    def _create_hooks(self, interventions: Dict[str, List[Any]], seq_len: int) -> List[Tuple]:
        """
        Convert our intervention format to TransformerLens hooks.

        TransformerLens hook format: (hook_name, hook_function)
        Hook names in TL: "blocks.{layer}.hook_resid_pre", "blocks.{layer}.attn.hook_result", etc.
        """
        hooks = []

        # Map our intervention names to TransformerLens hook points
        # Our format: "layer_{idx}_output", "layer_{idx}_attn_output", "layer_{idx}_mlp_output"
        # TL format: "blocks.{idx}.hook_resid_post", "blocks.{idx}.hook_attn_out", "blocks.{idx}.hook_mlp_out"

        for name, configs in interventions.items():
            if not isinstance(configs, list):
                configs = [configs]

            # Separate attention blocking from stream interventions
            # Handle both dict and Pydantic objects
            stream_configs = []
            attention_configs = []

            for c in configs:
                config_dict = c.dict() if hasattr(c, 'dict') else c
                if config_dict.get("type") == "block_attention":
                    attention_configs.append(c)
                else:
                    stream_configs.append(c)

            # Handle attention blocking interventions
            if attention_configs:
                self._add_attention_blocking_hooks(hooks, name, attention_configs, seq_len)

            if not stream_configs:
                continue

            # Handle different intervention types
            if name == "embeddings":
                hook_name = "hook_embed"
                hooks.append((hook_name, self._make_intervention_hook(stream_configs)))

            elif name.startswith("all_layers"):
                # Apply to all layers
                parts = name.split("_")
                if len(parts) >= 3:
                    location = "_".join(parts[2:])
                    for layer_idx in range(self.model.cfg.n_layers):
                        tl_hook_name = self._map_hook_name(layer_idx, location)
                        if tl_hook_name:
                            hooks.append((tl_hook_name, self._make_intervention_hook(stream_configs)))

            elif name.startswith("layer_"):
                # Single layer
                try:
                    parts = name.split("_")
                    layer_idx = int(parts[1])
                    location = "_".join(parts[2:])
                    tl_hook_name = self._map_hook_name(layer_idx, location)
                    if tl_hook_name:
                        hooks.append((tl_hook_name, self._make_intervention_hook(stream_configs)))
                except (ValueError, IndexError):
                    continue

        return hooks

    def _add_attention_blocking_hooks(
        self, hooks: List[Tuple], name: str, configs: List[Any], seq_len: int
    ):
        """
        Add attention blocking hooks for TransformerLens.

        In TransformerLens, we hook into the attention scores before softmax.
        Hook point: blocks.{layer}.attn.hook_attn_scores
        """
        # Determine which layers to apply to
        if name.startswith("all_layers_attention"):
            layer_indices = range(self.model.cfg.n_layers)
        elif name.startswith("layer_") and "_attention" in name:
            try:
                layer_idx_str = name.split("_")[1]
                layer_indices = [int(layer_idx_str)]
            except (ValueError, IndexError):
                return
        else:
            return

        # Create the blocking hook for each layer
        for layer_idx in layer_indices:
            hook_name = f"blocks.{layer_idx}.attn.hook_attn_scores"
            hook_fn = self._make_attention_blocking_hook(configs, seq_len)
            hooks.append((hook_name, hook_fn))

    def _make_attention_blocking_hook(self, configs: List[Any], seq_len: int):
        """
        Create an attention blocking hook function.

        Attention scores shape: [batch, num_heads, query_pos, key_pos]
        We modify scores to -inf to block attention between specific tokens.
        """
        def hook_fn(attn_scores, hook):
            # attn_scores: [batch, num_heads, query_pos, key_pos]

            for config in configs:
                config_dict = config.dict() if hasattr(config, 'dict') else config

                source_tokens = config_dict.get("source_tokens", [])
                target_tokens = config_dict.get("target_tokens", [])

                if source_tokens and target_tokens:
                    # Block attention from source tokens to target tokens
                    # target_tokens = query positions, source_tokens = key positions
                    for tgt in target_tokens:
                        for src in source_tokens:
                            if 0 <= tgt < seq_len and 0 <= src < seq_len:
                                # Set to -inf across all heads
                                # [batch, num_heads, tgt, src] = -inf
                                attn_scores[:, :, tgt, src] = float("-inf")

            return attn_scores

        return hook_fn

    def _map_hook_name(self, layer_idx: int, location: str) -> Optional[str]:
        """Map our hook location names to TransformerLens hook names."""
        mapping = {
            "output": f"blocks.{layer_idx}.hook_resid_post",
            "attn_output": f"blocks.{layer_idx}.hook_attn_out",
            "mlp_output": f"blocks.{layer_idx}.hook_mlp_out",
        }
        return mapping.get(location)

    def _make_intervention_hook(self, configs: List[Any]):
        """Create a hook function from intervention configs."""
        def hook_fn(tensor, hook):
            # tensor shape: [batch, seq_len, hidden_size]
            for config in configs:
                config_dict = config.dict() if hasattr(config, 'dict') else config

                token_index = config_dict.get("token_index")
                intervention_type = config_dict.get("type")

                if token_index is not None:
                    # Apply to specific token
                    idx = token_index
                    if idx < 0:
                        idx += tensor.shape[1]

                    if 0 <= idx < tensor.shape[1]:
                        if intervention_type == "zero":
                            tensor[:, idx, :] = 0
                        elif intervention_type == "scale":
                            value = config_dict.get("value", 1.0)
                            tensor[:, idx, :] *= value
                else:
                    # Apply to all tokens - use in-place operations
                    if intervention_type == "zero":
                        tensor.zero_()
                    elif intervention_type == "scale":
                        value = config_dict.get("value", 1.0)
                        tensor.mul_(value)

            return tensor

        return hook_fn

    def _extract_states(self, cache, lens_type: str) -> Tuple[List[torch.Tensor], List[str]]:
        """Extract hidden states based on lens_type."""
        states = []
        names = []

        if lens_type == "block_output":
            # Get embedding + all block outputs
            states.append(cache["hook_embed"])
            names.append("Embeddings")

            for i in range(self.model.cfg.n_layers):
                states.append(cache[f"blocks.{i}.hook_resid_post"])
                names.append(f"Layer {i}")

            # Last one is final output
            names[-1] = "Final Output"

        elif lens_type == "post_attention":
            # Get embedding + post-attention states
            states.append(cache["hook_embed"])
            names.append("Embeddings")

            for i in range(self.model.cfg.n_layers):
                # Post-attention = resid_pre + attn_out
                states.append(cache[f"blocks.{i}.hook_resid_mid"])
                names.append(f"L{i} Post-Attn")

            # Add final output
            states.append(cache[f"blocks.{self.model.cfg.n_layers - 1}.hook_resid_post"])
            names.append("Final Output")

        elif lens_type == "combined":
            # Interleave post-attention and block outputs
            states.append(cache["hook_embed"])
            names.append("Embeddings")

            for i in range(self.model.cfg.n_layers):
                # Post-attention
                states.append(cache[f"blocks.{i}.hook_resid_mid"])
                names.append(f"L{i} Post-Attn")

                # Block output
                states.append(cache[f"blocks.{i}.hook_resid_post"])
                names.append(f"L{i} Block Out")

        return states, names

    def _compute_logit_lens(self, states: List[torch.Tensor], layer_names: List[str]) -> List[Dict]:
        """Compute logit lens predictions for each state."""
        lens_data = []

        for i, (state, layer_name) in enumerate(zip(states, layer_names)):
            # Apply layer norm and unembed
            # TransformerLens: ln_final then unembed
            normed = self.model.ln_final(state)
            logits = self.model.unembed(normed)

            # Get top-k predictions
            predictions = self._decode_top_k(logits, k=5)

            lens_data.append({
                "layer_index": i,
                "layer_name": layer_name,
                "predictions": predictions
            })

        return lens_data

    def _decode_top_k(self, logits: torch.Tensor, k: int = 5) -> List[List[Dict]]:
        """Decode logits to top-k tokens."""
        probs = F.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, k, dim=-1)

        top_indices = top_indices.cpu().tolist()
        top_probs = top_probs.cpu().tolist()

        # Return for batch 0 only
        seq_results = []
        for s in range(len(top_indices[0])):
            token_data = []
            for i in range(k):
                token_id = top_indices[0][s][i]
                prob = top_probs[0][s][i]
                token_str = self.tokenizer.decode([token_id])
                token_data.append({"token": token_str, "prob": prob, "id": token_id})
            seq_results.append(token_data)

        return seq_results

    def _prepare_tensors(
        self,
        states: List[torch.Tensor],
        layer_names: List[str],
        cache,
        logits: torch.Tensor
    ) -> Dict[str, np.ndarray]:
        """Prepare tensors for saving."""
        tensors = {
            "hidden_states": np.stack([s.cpu().numpy() for s in states]),
            "logits": logits.cpu().numpy(),
            "layer_names": np.array(layer_names)
        }

        # Extract post-attention states if available
        post_attn_states = []
        for i in range(self.model.cfg.n_layers):
            key = f"blocks.{i}.hook_resid_mid"
            if key in cache:
                post_attn_states.append(cache[key].cpu().numpy())

        if post_attn_states:
            tensors["post_attention_states"] = np.stack(post_attn_states)

        return tensors

    def _extract_attention(self, cache) -> List[List[List[List[float]]]]:
        """Extract attention patterns from cache."""
        attn_data = []

        for i in range(self.model.cfg.n_layers):
            # TransformerLens stores attention patterns at blocks.{i}.attn.hook_pattern
            attn_pattern = cache[f"blocks.{i}.attn.hook_pattern"]
            # Shape: [batch, num_heads, seq, seq]
            # Take batch 0
            attn_data.append(attn_pattern[0].cpu().numpy().tolist())

        return attn_data
