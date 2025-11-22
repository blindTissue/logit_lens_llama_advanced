import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import math
from dataclasses import dataclass

@dataclass
class LlamaConfig:
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    hidden_act: str = "silu"
    max_position_embeddings: int = 2048
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    rope_theta: float = 10000.0

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_position_embeddings, device=device, dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = F.silu

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class LlamaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings, base=self.rope_theta)

    def forward(self, hidden_states, position_ids, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = torch.repeat_interleave(key_states, dim=1, repeats=self.num_key_value_groups)
        value_states = torch.repeat_interleave(value_states, dim=1, repeats=self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        # Causal mask
        if q_len > 1:
             # This is a simplified causal mask for demonstration. 
             # In production, we need to handle padding and proper causal masking for batching.
             # For now, assuming left-padding or single batch inference.
             mask = torch.full((q_len, q_len), float("-inf"), device=hidden_states.device)
             mask = torch.triu(mask, diagonal=1)
             attn_weights = attn_weights + mask
        
        # Apply custom attention mask if provided
        if attention_mask is not None:
            # attention_mask shape: [batch, q_len, kv_seq_len] or [q_len, kv_seq_len]
            # attn_weights shape: [batch, num_heads, q_len, kv_seq_len]
            if attention_mask.dim() == 2:
                # Broadcast to [batch, num_heads, q_len, kv_seq_len]
                attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
            elif attention_mask.dim() == 3:
                # Broadcast to [batch, num_heads, q_len, kv_seq_len]
                attention_mask = attention_mask.unsqueeze(1)
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layer_idx = layer_idx

    def forward(self, hidden_states, position_ids, past_key_value=None, use_cache=False, interventions=None, attention_mask=None):
        # Residual Connection 1
        residual = hidden_states
        
        # Pre-Attention Norm
        hidden_states = self.input_layernorm(hidden_states)
        
        # Attention
        attn_outputs, past_key_value = self.self_attn(
            hidden_states=hidden_states,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            attention_mask=attention_mask,
        )
        
        # Intervention Point: Post-Attention (before adding residual)
        if interventions and f"layer_{self.layer_idx}_attn_output" in interventions:
             attn_outputs = interventions[f"layer_{self.layer_idx}_attn_output"](attn_outputs)

        # Residual Add 1
        hidden_states = residual + attn_outputs

        # Post-Attention State (for fine-grained LogitLens)
        post_attention_state = hidden_states

        # Residual Connection 2
        residual = hidden_states

        # Post-Attention Norm
        hidden_states = self.post_attention_layernorm(hidden_states)

        # MLP
        mlp_outputs = self.mlp(hidden_states)

        # Intervention Point: Post-MLP (before adding residual)
        if interventions and f"layer_{self.layer_idx}_mlp_output" in interventions:
             mlp_outputs = interventions[f"layer_{self.layer_idx}_mlp_output"](mlp_outputs)

        # Residual Add 2
        hidden_states = residual + mlp_outputs
        
        # Intervention Point: Post-Layer (Block Output)
        if interventions and f"layer_{self.layer_idx}_output" in interventions:
             hidden_states = interventions[f"layer_{self.layer_idx}_output"](hidden_states)

        return hidden_states, past_key_value, post_attention_state

class LlamaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.padding_idx = 0 # Default
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, position_ids=None, past_key_values=None, use_cache=True, interventions=None, attention_masks=None):
        output_attentions = False
        output_hidden_states = True # We always want these for LogitLens
        
        if position_ids is None:
             position_ids = torch.arange(0, input_ids.shape[1], device=input_ids.device).unsqueeze(0)

        inputs_embeds = self.embed_tokens(input_ids)
        
        # Intervention Point: Embeddings
        if interventions and "embeddings" in interventions:
            inputs_embeds = interventions["embeddings"](inputs_embeds)

        hidden_states = inputs_embeds
        
        all_hidden_states = () if output_hidden_states else None
        all_post_attention_states = () if output_hidden_states else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            # Get attention mask for this layer if provided
            layer_attention_mask = attention_masks.get(idx) if attention_masks else None

            layer_outputs = decoder_layer(
                hidden_states,
                position_ids=position_ids,
                past_key_value=past_key_values[idx] if past_key_values is not None else None,
                use_cache=use_cache,
                interventions=interventions,
                attention_mask=layer_attention_mask
            )

            hidden_states = layer_outputs[0]
            
            if output_hidden_states:
                all_post_attention_states += (layer_outputs[2],)

            if use_cache:
                next_decoder_cache += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
            
        logits = self.lm_head(hidden_states)

        return {
            "logits": logits,
            "past_key_values": next_decoder_cache,
            "hidden_states": all_hidden_states,
            "post_attention_states": all_post_attention_states
        }

    @classmethod
    def from_pretrained(cls, model_name_or_path, device="cpu"):
        from transformers import AutoModelForCausalLM, AutoConfig
        
        print(f"Loading weights from {model_name_or_path}...")
        hf_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16)
        hf_config = AutoConfig.from_pretrained(model_name_or_path)
        
        config = LlamaConfig(
            vocab_size=hf_config.vocab_size,
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.intermediate_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            num_key_value_heads=hf_config.num_key_value_heads,
            max_position_embeddings=hf_config.max_position_embeddings,
            rms_norm_eps=hf_config.rms_norm_eps,
            rope_theta=hf_config.rope_theta,
        )
        
        model = cls(config)
        
        # Copy weights
        # This is a simplified mapping. Might need adjustment based on exact variable names.
        state_dict = hf_model.state_dict()
        my_state_dict = model.state_dict()
        
        # We need to map HF keys to our keys
        # HF: model.layers.0.self_attn.q_proj.weight
        # Ours: layers.0.self_attn.q_proj.weight
        # HF: model.embed_tokens.weight -> embed_tokens.weight
        # HF: model.norm.weight -> norm.weight
        # HF: lm_head.weight -> lm_head.weight
        
        mapped_keys = {}
        for k, v in state_dict.items():
            new_k = k
            if k.startswith("model."):
                new_k = k[6:] # remove "model."
            
            if new_k in my_state_dict:
                my_state_dict[new_k].data.copy_(v.data)
                mapped_keys[new_k] = True
            else:
                print(f"Skipping {k} -> {new_k} (not found in custom model)")

        model.to(device)
        return model

