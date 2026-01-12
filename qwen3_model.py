"""
Qwen3 Model Implementation for LogitLens analysis
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import math
from dataclasses import dataclass, field


@dataclass
class Qwen3Config:
    """Configuration for Qwen3 models."""
    vocab_size: int = 151936
    hidden_size: int = 4096
    intermediate_size: int = 22016
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    head_dim: int = 128  # Explicit head dimension in Qwen3
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    rope_theta: float = 10000.0
    attention_bias: bool = False  # Qwen3 default is False
    attention_dropout: float = 0.0
    use_sliding_window: bool = False
    sliding_window: Optional[int] = None


class Qwen3RMSNorm(nn.Module):
    """RMSNorm for Qwen3 - same as Llama."""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Qwen3RotaryEmbedding(nn.Module):
    """Rotary Position Embedding for Qwen3."""
    def __init__(self, dim: int, max_position_embeddings: int = 32768, base: float = 10000.0, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_position_embeddings, device=device, dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len: int, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor for dtype reference
            position_ids: Position indices [batch_size, seq_len]
        Returns:
            cos, sin: [batch_size, seq_len, head_dim]
        """
        seq_len = position_ids.max().item() + 1
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, device=x.device, dtype=x.dtype)

        # Gather cos/sin values for the given position_ids
        cos = self.cos_cached[position_ids].to(dtype=x.dtype)  # [batch, seq_len, dim]
        sin = self.sin_cached[position_ids].to(dtype=x.dtype)  # [batch, seq_len, dim]
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key tensors.

    Args:
        q: Query tensor [batch, num_heads, seq_len, head_dim]
        k: Key tensor [batch, num_kv_heads, seq_len, head_dim]
        cos: Cosine part [batch, seq_len, head_dim]
        sin: Sine part [batch, seq_len, head_dim]
    """
    # Unsqueeze to broadcast over heads: [batch, 1, seq_len, head_dim]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key/value heads to match query heads for GQA.

    Args:
        hidden_states: [batch, num_kv_heads, seq_len, head_dim]
        n_rep: Number of times to repeat
    Returns:
        [batch, num_kv_heads * n_rep, seq_len, head_dim]
    """
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


class Qwen3MLP(nn.Module):
    """MLP for Qwen3 - same structure as Llama."""
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = F.silu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen3Attention(nn.Module):
    """
    Multi-headed attention for Qwen3.

    Key difference from Llama: Q/K normalization before RoPE.
    """
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim ** -0.5

        # Projections with configurable bias
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        # Qwen3-specific: Q/K normalization (applied per head dimension)
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # RoPE
        self.rotary_emb = Qwen3RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        bsz, q_len, _ = hidden_states.size()

        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape to [batch, seq_len, num_heads, head_dim]
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        # Qwen3-specific: Apply Q/K normalization BEFORE transpose and RoPE
        # q_norm and k_norm operate on head_dim (last dimension)
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # Transpose to [batch, num_heads, seq_len, head_dim]
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Handle past key values (for caching)
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None
        kv_seq_len = key_states.shape[2]

        # Expand KV heads for GQA
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

        # Apply causal mask
        if q_len > 1:
            causal_mask = torch.full((q_len, kv_seq_len), float("-inf"), device=hidden_states.device)
            causal_mask = torch.triu(causal_mask, diagonal=kv_seq_len - q_len + 1)
            attn_weights = attn_weights + causal_mask

        # Apply custom attention mask if provided
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            attn_weights = attn_weights + attention_mask

        # Softmax and output
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        if output_attentions:
            return attn_output, past_key_value, attn_weights
        return attn_output, past_key_value, None


class Qwen3DecoderLayer(nn.Module):
    """Transformer decoder layer for Qwen3."""
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = Qwen3Attention(config, layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        interventions: Optional[Dict[str, Any]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor, Optional[torch.Tensor]]:
        # Residual connection 1
        residual = hidden_states

        # Pre-attention normalization
        hidden_states = self.input_layernorm(hidden_states)

        # Self-attention
        attn_outputs, past_key_value, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            attention_mask=attention_mask,
            output_attentions=output_attentions
        )

        # Intervention point: Post-attention (before residual)
        if interventions and f"layer_{self.layer_idx}_attn_output" in interventions:
            attn_outputs = interventions[f"layer_{self.layer_idx}_attn_output"](attn_outputs)

        # Residual add 1
        hidden_states = residual + attn_outputs

        # Post-attention state (for fine-grained LogitLens)
        post_attention_state = hidden_states

        # Residual connection 2
        residual = hidden_states

        # Post-attention normalization
        hidden_states = self.post_attention_layernorm(hidden_states)

        # MLP
        mlp_outputs = self.mlp(hidden_states)

        # Intervention point: Post-MLP (before residual)
        if interventions and f"layer_{self.layer_idx}_mlp_output" in interventions:
            mlp_outputs = interventions[f"layer_{self.layer_idx}_mlp_output"](mlp_outputs)

        # Residual add 2
        hidden_states = residual + mlp_outputs

        # Intervention point: Post-layer (block output)
        if interventions and f"layer_{self.layer_idx}_output" in interventions:
            hidden_states = interventions[f"layer_{self.layer_idx}_output"](hidden_states)

        return hidden_states, past_key_value, post_attention_state, attn_weights


class Qwen3Model(nn.Module):
    """Qwen3 model for LogitLens analysis."""
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.padding_idx = 0
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = True,
        interventions: Optional[Dict[str, Any]] = None,
        attention_masks: Optional[Dict[int, torch.Tensor]] = None,
        output_attentions: bool = False
    ) -> Dict[str, Any]:
        output_hidden_states = True  # Always true for LogitLens

        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)

        # Embeddings
        inputs_embeds = self.embed_tokens(input_ids)

        # Intervention point: Embeddings
        if interventions and "embeddings" in interventions:
            inputs_embeds = interventions["embeddings"](inputs_embeds)

        hidden_states = inputs_embeds

        # Storage for outputs
        all_hidden_states = ()
        all_post_attention_states = ()
        all_attentions = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        # Forward through layers
        for idx, decoder_layer in enumerate(self.layers):
            all_hidden_states += (hidden_states,)

            # Get per-layer attention mask if provided
            layer_attention_mask = attention_masks.get(idx) if attention_masks else None

            layer_outputs = decoder_layer(
                hidden_states,
                position_ids=position_ids,
                past_key_value=past_key_values[idx] if past_key_values is not None else None,
                use_cache=use_cache,
                interventions=interventions,
                attention_mask=layer_attention_mask,
                output_attentions=output_attentions
            )

            hidden_states = layer_outputs[0]
            all_post_attention_states += (layer_outputs[2],)

            if output_attentions:
                all_attentions += (layer_outputs[3],)

            if use_cache:
                next_decoder_cache += (layer_outputs[1],)

        # Final normalization
        hidden_states = self.norm(hidden_states)
        all_hidden_states += (hidden_states,)

        # LM head
        logits = self.lm_head(hidden_states)

        return {
            "logits": logits,
            "past_key_values": next_decoder_cache,
            "hidden_states": all_hidden_states,
            "post_attention_states": all_post_attention_states,
            "attentions": all_attentions
        }

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, device: str = "cpu") -> "Qwen3Model":
        """Load a pretrained Qwen3 model from HuggingFace."""
        from transformers import AutoModelForCausalLM, AutoConfig

        print(f"Loading weights from {model_name_or_path}...")
        hf_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16)
        hf_config = AutoConfig.from_pretrained(model_name_or_path)

        # Build our config from HF config
        config = Qwen3Config(
            vocab_size=hf_config.vocab_size,
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.intermediate_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            num_key_value_heads=hf_config.num_key_value_heads,
            head_dim=getattr(hf_config, 'head_dim', hf_config.hidden_size // hf_config.num_attention_heads),
            max_position_embeddings=hf_config.max_position_embeddings,
            rms_norm_eps=hf_config.rms_norm_eps,
            rope_theta=hf_config.rope_theta,
            attention_bias=getattr(hf_config, 'attention_bias', False),
        )

        model = cls(config)

        # Copy weights from HF model
        state_dict = hf_model.state_dict()
        my_state_dict = model.state_dict()

        # Map HF keys to our keys
        # HF: model.layers.0.self_attn.q_proj.weight -> layers.0.self_attn.q_proj.weight
        # HF: model.embed_tokens.weight -> embed_tokens.weight
        # HF: model.norm.weight -> norm.weight
        # HF: lm_head.weight -> lm_head.weight
        mapped_count = 0
        skipped_keys = []

        for k, v in state_dict.items():
            new_k = k
            if k.startswith("model."):
                new_k = k[6:]  # Remove "model." prefix

            if new_k in my_state_dict:
                if my_state_dict[new_k].shape == v.shape:
                    my_state_dict[new_k].data.copy_(v.data)
                    mapped_count += 1
                else:
                    skipped_keys.append(f"{k} -> {new_k} (shape mismatch: {v.shape} vs {my_state_dict[new_k].shape})")
            else:
                skipped_keys.append(f"{k} -> {new_k} (not found)")

        if skipped_keys:
            print(f"Skipped {len(skipped_keys)} keys:")
            for sk in skipped_keys[:10]:  # Only show first 10
                print(f"  {sk}")
            if len(skipped_keys) > 10:
                print(f"  ... and {len(skipped_keys) - 10} more")

        print(f"Mapped {mapped_count} weights successfully")
        model.to(device)
        return model
