from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import torch
import numpy as np
from transformers import AutoTokenizer
from model import LlamaModel
from logit_lens import compute_logit_lens, decode_top_k
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
model = None
tokenizer = None
model_config = None

class LoadModelRequest(BaseModel):
    model_name: str = "meta-llama/Llama-3.2-3B"

class InterventionConfig(BaseModel):
    type: str # "scale", "zero", "add", "block_attention"
    value: Optional[float] = None # For scale
    vector: Optional[List[float]] = None # For add
    token_index: Optional[int] = None # Optional: target specific token
    source_tokens: Optional[List[int]] = None # For block_attention: tokens to block FROM
    target_tokens: Optional[List[int]] = None # For block_attention: tokens to block TO

class InferenceRequest(BaseModel):
    text: str
    interventions: Dict[str, InterventionConfig] = {} # key: hook_name (e.g. "layer_0_output")
    lens_type: str = "block_output" # "block_output" or "post_attention"

class SaveStateRequest(BaseModel):
    filename: str
    data: Dict[str, Any] # This might be too large to pass back and forth. 
                         # Better: The server keeps the last state and saves it on request.

# Store last run state for saving
last_run_state = {}

import threading

model_lock = threading.Lock()

@app.post("/load_model")
def load_model(req: LoadModelRequest):
    global model, tokenizer, model_config
    
    # Quick check before lock
    if model is not None and model_config == req.model_name:
         return {"status": "already_loaded", "config": str(model.config)}

    with model_lock:
        # Check again inside lock
        if model is not None and model_config == req.model_name:
             return {"status": "already_loaded", "config": str(model.config)}
             
        try:
            print(f"Loading {req.model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(req.model_name)
            model = LlamaModel.from_pretrained(req.model_name, device="cpu") # Use CPU for safety/compatibility first
            # model.to("mps") # Uncomment if on Mac with MPS support
            model_config = req.model_name # Store the model name in model_config global variable
            return {"status": "loaded", "config": str(model.config)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/model_status")
def get_model_status():
    global model, model_config
    return {"loaded": model is not None, "model_name": model_config if model else None}

def create_intervention_hook(config: InterventionConfig):
    def hook(tensor):
        # tensor shape: [batch, seq_len, hidden_size]
        
        if config.token_index is not None:
            # Apply only to specific token
            idx = config.token_index
            if idx < 0: idx += tensor.shape[1] # Handle negative indices
            
            if 0 <= idx < tensor.shape[1]:
                if config.type == "zero":
                    tensor[:, idx, :] = 0
                elif config.type == "scale":
                    tensor[:, idx, :] *= config.value
            return tensor
        
        # Apply to all tokens if no index specified
        if config.type == "zero":
            return torch.zeros_like(tensor)
        elif config.type == "scale":
            return tensor * config.value
        # Add more complex ones later
        return tensor
    return hook

@app.post("/inference")
def inference(req: InferenceRequest):
    global model, tokenizer, last_run_state
    if model is None:
        raise HTTPException(status_code=400, detail="Model not loaded")

    input_ids = tokenizer.encode(req.text, return_tensors="pt").to(model.embed_tokens.weight.device)
    seq_len = input_ids.shape[1]
    
    # Parse interventions
    intervention_hooks = {}
    attention_masks = {}  # Dict[layer_idx, mask_tensor]
    
    for name, config in req.interventions.items():
        if config.type == "block_attention":
            # Extract layer index from intervention name (e.g., "layer_5_attention" -> 5)
            if name.startswith("layer_") and "_attention" in name:
                layer_idx = int(name.split("_")[1])
                
                # Create attention mask for this layer
                # Mask shape: [seq_len, seq_len] with -inf where attention should be blocked
                mask = torch.zeros((seq_len, seq_len), device=input_ids.device)
                
                if config.source_tokens and config.target_tokens:
                    for src in config.source_tokens:
                        for tgt in config.target_tokens:
                            # Block attention from src to tgt
                            if 0 <= src < seq_len and 0 <= tgt < seq_len:
                                mask[tgt, src] = float("-inf")
                
                attention_masks[layer_idx] = mask
        else:
            # Regular intervention (scale, zero, etc.)
            intervention_hooks[name] = create_intervention_hook(config)

    with torch.no_grad():
        outputs = model(input_ids, interventions=intervention_hooks, attention_masks=attention_masks)
    
    # Process LogitLens
    try:
        if req.lens_type == "post_attention":
            # embeddings + post_attention_states + final_norm (not really applicable for final norm but we can keep it)
            # actually post_attention_states has N items.
            # We should probably include embeddings as the "start".
            states_to_process = [outputs["hidden_states"][0]] + list(outputs["post_attention_states"])
            # Note: post_attention_states doesn't include the final output of the model, 
            # but usually logit lens includes the final output.
            # Let's append the final hidden state as well to complete the picture
            states_to_process.append(outputs["hidden_states"][-1])
        elif req.lens_type == "combined":
            # Interleave post-attention and block output states
            states_to_process = [outputs["hidden_states"][0]]  # Start with embeddings
            for i in range(len(outputs["post_attention_states"])):
                states_to_process.append(outputs["post_attention_states"][i])  # Post-attn
                states_to_process.append(outputs["hidden_states"][i + 1])  # Block output
        else:
            states_to_process = outputs["hidden_states"] # Tuple of (batch, seq, hidden)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing states: {str(e)}")
    
    lens_data = []
    
    for i, h in enumerate(states_to_process):
        # Project
        logits = compute_logit_lens(h, model.lm_head, model.norm)
        decoded = decode_top_k(logits, tokenizer, k=5)
        
        if req.lens_type == "post_attention":
             if i == 0: layer_name = "Embeddings"
             elif i == len(states_to_process) - 1: layer_name = "Final Output"
             else: layer_name = f"L{i-1} Post-Attn"
        elif req.lens_type == "combined":
            if i == 0:
                layer_name = "Embeddings"
            else:
                # Calculate which layer we're in
                layer_num = (i - 1) // 2
                is_post_attn = (i - 1) % 2 == 0
                if is_post_attn:
                    layer_name = f"L{layer_num} Post-Attn"
                else:
                    layer_name = f"L{layer_num} Block Out"
        else:
            layer_name = f"Layer {i-1}" if i > 0 else "Embeddings"
            if i == len(states_to_process) - 1:
                layer_name = "Final Output"
            
        # decoded[0] is List[List[Dict]] (seq_len -> top_k)
        # The frontend expects a flat list of tokens (one per position), so we take top-1.
        top1_predictions = [preds[0] for preds in decoded[0]]

        lens_data.append({
            "layer_index": i,
            "layer_name": layer_name,
            "predictions": top1_predictions 
        })

    # Store for saving
    last_run_state["hidden_states"] = [h.cpu().numpy() for h in states_to_process]
    last_run_state["logits"] = outputs["logits"].cpu().numpy()
    last_run_state["text"] = req.text

    return {
        "text": req.text,
        "logit_lens": lens_data
    }

@app.post("/save_state")
def save_state(filename: str = "state.npz"):
    if not last_run_state:
        raise HTTPException(status_code=400, detail="No state to save")
    
    path = os.path.join(os.getcwd(), filename)
    np.savez(path, **last_run_state)
    return {"status": "saved", "path": path}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
