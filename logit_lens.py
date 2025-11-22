import torch
import torch.nn.functional as F

def compute_logit_lens(hidden_states: torch.Tensor, lm_head: torch.nn.Linear, norm: torch.nn.Module = None):
    """
    Projects hidden states to vocabulary space.
    
    Args:
        hidden_states: [batch, seq_len, hidden_size]
        lm_head: The unembedding layer (Linear)
        norm: Optional normalization layer (RMSNorm) to apply before projection.
              Llama usually applies RMSNorm before the head.
    
    Returns:
        logits: [batch, seq_len, vocab_size]
    """
    if norm:
        hidden_states = norm(hidden_states)
    
    logits = lm_head(hidden_states)
    return logits

def decode_top_k(logits: torch.Tensor, tokenizer, k: int = 5):
    """
    Decodes logits to top-k tokens.
    
    Args:
        logits: [batch, seq_len, vocab_size]
        tokenizer: HF tokenizer
        k: Number of top tokens to return
        
    Returns:
        List of lists of dicts: [batch_idx][seq_idx] -> [{"token": str, "prob": float}, ...]
    """
    probs = F.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, k, dim=-1)
    
    # Convert to CPU for decoding
    top_indices = top_indices.cpu().tolist()
    top_probs = top_probs.cpu().tolist()
    
    batch_results = []
    for b in range(len(top_indices)):
        seq_results = []
        for s in range(len(top_indices[b])):
            token_data = []
            for i in range(k):
                token_id = top_indices[b][s][i]
                prob = top_probs[b][s][i]
                token_str = tokenizer.decode([token_id])
                token_data.append({"token": token_str, "prob": prob, "id": token_id})
            seq_results.append(token_data)
        batch_results.append(seq_results)
        
    return batch_results
