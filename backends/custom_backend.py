"""
Custom backend wrapper for the existing model.py and qwen3_model.py implementations.
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional
from transformers import AutoTokenizer
import gc

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from model import LlamaModel
from qwen3_model import Qwen3Model
from logit_lens import compute_logit_lens, decode_top_k
from backends.base import BaseBackend


def get_model_class(model_name: str):
    """Return the appropriate model class based on model name."""
    model_name_lower = model_name.lower()
    if "qwen3" in model_name_lower:
        return Qwen3Model
    else:
        return LlamaModel


class CustomBackend(BaseBackend):
    """Backend using the custom model implementations."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = None

    def load_model(self, model_name: str, device: str = "cpu") -> Dict[str, Any]:
        """Load a model using custom implementation."""
        if self.model is not None and self.model_name == model_name:
            return {
                "status": "already_loaded",
                "backend": "custom",
                "model_name": model_name,
                "config": str(self.model.config)
            }

        # Unload previous model
        if self.model is not None:
            self.unload_model()

        try:
            print(f"Loading {model_name} with custom implementation...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            ModelClass = get_model_class(model_name)
            print(f"Using model class: {ModelClass.__name__}")
            self.model = ModelClass.from_pretrained(model_name, device=device)

            self.model_name = model_name

            return {
                "status": "loaded",
                "backend": "custom",
                "model_name": model_name,
                "model_class": ModelClass.__name__,
                "config": str(self.model.config)
            }

        except Exception as e:
            raise RuntimeError(f"Failed to load model with custom backend: {str(e)}")

    def unload_model(self):
        """Unload the model and free memory."""
        if self.model is not None:
            print("Unloading custom model...")
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
            "backend": "custom",
            "model_name": self.model_name,
            "config": str(self.model.config)
        }

    def get_num_layers(self) -> int:
        """Get number of layers."""
        if not self.is_loaded():
            return 0
        return self.model.config.num_hidden_layers

    def get_config(self) -> Any:
        """Get model config."""
        if not self.is_loaded():
            return None
        return self.model.config

    def run_inference(
        self,
        text: str,
        interventions: Dict[str, List[Any]],
        lens_type: str = "block_output",
        apply_chat_template: bool = False,
        return_attention: bool = False
    ) -> Dict[str, Any]:
        """Run inference with custom models."""
        if not self.is_loaded():
            raise RuntimeError("No model loaded")

        # Tokenize
        if apply_chat_template:
            messages = [{"role": "user", "content": text}]
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                input_ids = self.tokenizer.encode(
                    prompt, return_tensors="pt", add_special_tokens=False
                ).to(self.model.embed_tokens.weight.device)
            except Exception as e:
                raise RuntimeError(f"Failed to apply chat template: {str(e)}")
        else:
            input_ids = self.tokenizer.encode(text, return_tensors="pt").to(
                self.model.embed_tokens.weight.device
            )

        seq_len = input_ids.shape[1]

        # Parse interventions (reuse existing logic)
        intervention_hooks, attention_masks = self._parse_interventions(
            interventions, seq_len, input_ids.device
        )

        # Run model
        with torch.no_grad():
            outputs = self.model(
                input_ids,
                interventions=intervention_hooks,
                attention_masks=attention_masks,
                output_attentions=True
            )

        # Process states based on lens_type
        states_to_process, layer_names = self._extract_states(outputs, lens_type)

        # Compute logit lens
        # Note: All hidden states are now pre-normalized, so we always apply norm
        lens_data = []
        for i, (state, layer_name) in enumerate(zip(states_to_process, layer_names)):
            logits = compute_logit_lens(state, self.model.lm_head, self.model.norm)
            decoded = decode_top_k(logits, self.tokenizer, k=5)

            lens_data.append({
                "layer_index": i,
                "layer_name": layer_name,
                "predictions": decoded[0]
            })

        # Get input tokens
        input_tokens = [self.tokenizer.decode([tid]) for tid in input_ids[0].tolist()]

        # Prepare tensors
        tensors = {
            "hidden_states": np.stack([h.cpu().numpy() for h in states_to_process]),
            "logits": outputs["logits"].cpu().numpy(),
            "layer_names": np.array(layer_names),
            "post_attention_states": np.stack([h.cpu().numpy() for h in outputs["post_attention_states"]])
        }

        response = {
            "text": text,
            "input_tokens": input_tokens,
            "logit_lens": lens_data,
            "tensors": tensors
        }

        # Add attention if requested
        if return_attention and "attentions" in outputs and outputs["attentions"]:
            attn_data = []
            for layer_attn in outputs["attentions"]:
                attn_data.append(layer_attn[0].cpu().numpy().tolist())
            response["attention"] = attn_data
            tensors["attentions"] = np.stack([a.cpu().numpy() for a in outputs["attentions"]])

        return response

    def _parse_interventions(
        self, interventions: Dict[str, List[Any]], seq_len: int, device
    ) -> tuple:
        """Parse interventions into hooks and attention masks."""
        intervention_hooks = {}
        attention_masks = {}

        for name, configs in interventions.items():
            if not isinstance(configs, list):
                configs = [configs]

            stream_configs = []

            for config in configs:
                config_dict = config.dict() if hasattr(config, 'dict') else config

                if config_dict.get("type") == "block_attention":
                    # Handle attention blocking
                    is_attention_intervention = (
                        (name.startswith("layer_") and "_attention" in name) or
                        name.startswith("all_layers_attention")
                    )

                    if is_attention_intervention:
                        if config_dict.get("all_layers") or name.startswith("all_layers_attention"):
                            layer_indices = range(self.model.config.num_hidden_layers)
                        else:
                            try:
                                layer_idx_str = name.split("_")[1]
                                layer_indices = [int(layer_idx_str)]
                            except:
                                continue

                        for layer_idx in layer_indices:
                            if layer_idx not in attention_masks:
                                attention_masks[layer_idx] = torch.zeros((seq_len, seq_len), device=device)

                            mask = attention_masks[layer_idx]

                            if config_dict.get("source_tokens") and config_dict.get("target_tokens"):
                                for src in config_dict["source_tokens"]:
                                    for tgt in config_dict["target_tokens"]:
                                        if 0 <= src < seq_len and 0 <= tgt < seq_len:
                                            mask[tgt, src] = float("-inf")
                else:
                    stream_configs.append(config)

            if stream_configs:
                # Handle all_layers expansion
                final_hooks_map = {}

                if name.startswith("all_layers") and name != "embeddings":
                    parts = name.split("_")
                    if len(parts) >= 3:
                        location = "_".join(parts[2:])
                        for layer_idx in range(self.model.config.num_hidden_layers):
                            hook_name = f"layer_{layer_idx}_{location}"
                            if hook_name not in final_hooks_map:
                                final_hooks_map[hook_name] = []
                            final_hooks_map[hook_name].extend(stream_configs)
                else:
                    if name not in final_hooks_map:
                        final_hooks_map[name] = []
                    final_hooks_map[name].extend(stream_configs)

                for hook_name, configs_list in final_hooks_map.items():
                    intervention_hooks[hook_name] = self._create_intervention_hook(configs_list)

        return intervention_hooks, attention_masks

    def _create_intervention_hook(self, configs: List[Any]):
        """Create an intervention hook function."""
        def hook(tensor):
            for config in configs:
                config_dict = config.dict() if hasattr(config, 'dict') else config

                token_index = config_dict.get("token_index")
                intervention_type = config_dict.get("type")

                if token_index is not None:
                    idx = token_index
                    if idx < 0:
                        idx += tensor.shape[1]

                    if 0 <= idx < tensor.shape[1]:
                        if intervention_type == "zero":
                            tensor[:, idx, :] = 0
                        elif intervention_type == "scale":
                            tensor[:, idx, :] *= config_dict.get("value", 1.0)
                else:
                    if intervention_type == "zero":
                        tensor = torch.zeros_like(tensor)
                    elif intervention_type == "scale":
                        tensor = tensor * config_dict.get("value", 1.0)

            return tensor

        return hook

    def _extract_states(self, outputs: Dict, lens_type: str) -> tuple:
        """Extract states based on lens_type."""
        states = []
        names = []

        if lens_type == "post_attention":
            states.append(outputs["hidden_states"][0])
            names.append("Embeddings")

            for i, state in enumerate(outputs["post_attention_states"]):
                states.append(state)
                names.append(f"L{i} Post-Attn")

            states.append(outputs["hidden_states"][-1])
            names.append("Final Output")

        elif lens_type == "combined":
            states.append(outputs["hidden_states"][0])
            names.append("Embeddings")

            for i in range(len(outputs["post_attention_states"])):
                states.append(outputs["post_attention_states"][i])
                names.append(f"L{i} Post-Attn")

                states.append(outputs["hidden_states"][i + 1])
                names.append(f"L{i} Block Out")

            # Add final output state (after all layers, before final norm)
            states.append(outputs["hidden_states"][-1])
            names.append("Final Output")

        else:  # block_output
            for i, state in enumerate(outputs["hidden_states"]):
                states.append(state)
                if i == 0:
                    names.append("Embeddings")
                elif i == len(outputs["hidden_states"]) - 1:
                    names.append("Final Output")
                else:
                    names.append(f"Layer {i - 1}")

        return states, names
