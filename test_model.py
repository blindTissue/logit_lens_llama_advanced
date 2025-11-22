import torch
from model import LlamaModel
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_model():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"Loading {model_name}...")
    
    # Load my model
    my_model = LlamaModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    text = "The capital of France is"
    input_ids = tokenizer.encode(text, return_tensors="pt")
    
    print("Running inference...")
    with torch.no_grad():
        outputs = my_model(input_ids)
    
    logits = outputs["logits"]
    print(f"Logits shape: {logits.shape}")
    
    next_token = torch.argmax(logits[:, -1, :], dim=-1)
    print(f"Predicted token ID: {next_token.item()}")
    print(f"Predicted token: {tokenizer.decode(next_token)}")
    
    # Compare with HF model
    print("Comparing with HF model...")
    hf_model = AutoModelForCausalLM.from_pretrained(model_name)
    with torch.no_grad():
        hf_outputs = hf_model(input_ids)
    
    hf_logits = hf_outputs.logits
    diff = (logits - hf_logits).abs().max()
    print(f"Max difference in logits: {diff.item()}")
    
    if diff < 1e-3:
        print("SUCCESS: Models match!")
    else:
        print("WARNING: Large difference detected.")

if __name__ == "__main__":
    test_model()
