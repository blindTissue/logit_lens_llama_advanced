"""
Base interface for LogitLens backends.
All backend implementations should follow this interface.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import torch
import numpy as np


class BaseBackend(ABC):
    """Abstract base class for model backends."""

    @abstractmethod
    def load_model(self, model_name: str, device: str = "cpu") -> Dict[str, Any]:
        """
        Load a model from HuggingFace.

        Returns:
            Dict with status, model info, etc.
        """
        pass

    @abstractmethod
    def unload_model(self):
        """Unload the current model and free memory."""
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        pass

    @abstractmethod
    def run_inference(
        self,
        text: str,
        interventions: Dict[str, List[Any]],
        lens_type: str = "block_output",
        apply_chat_template: bool = False,
        return_attention: bool = False
    ) -> Dict[str, Any]:
        """
        Run inference with interventions and return logit lens data.

        Args:
            text: Input text
            interventions: Dict of intervention configs keyed by hook name
            lens_type: "block_output", "post_attention", or "combined"
            apply_chat_template: Whether to apply chat template
            return_attention: Whether to return attention patterns

        Returns:
            Dict with:
                - text: input text
                - input_tokens: list of token strings
                - logit_lens: list of layer predictions
                - attention: (optional) attention patterns
                - tensors: raw tensors for saving
        """
        pass

    @abstractmethod
    def get_num_layers(self) -> int:
        """Get number of layers in the model."""
        pass

    @abstractmethod
    def get_config(self) -> Any:
        """Get model configuration."""
        pass
