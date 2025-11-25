from transformers import AutoTokenizer

model_name = "meta-llama/Llama-3.2-1B-Instruct"
print(f"Loading tokenizer for {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

text = "Hello"
messages = [{"role": "user", "content": text}]

print(f"\n--- Test: apply_chat_template + encode (FIXED) ---")
# This is what server.py currently does
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(f"Prompt string: {repr(prompt)}")

# This is the FIX: add_special_tokens=False
input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
print(f"Input IDs: {input_ids}")
print(f"Tokens: {tokenizer.convert_ids_to_tokens(input_ids[0])}")

# Check for double BOS
# Llama 3 usually uses <|begin_of_text|> (128000)
bos_token_id = tokenizer.bos_token_id
print(f"BOS token ID: {bos_token_id}")

if input_ids[0][0] == bos_token_id and input_ids[0][1] == bos_token_id:
    print("\nFAIL: Double BOS detected!")
elif input_ids[0][0] == bos_token_id:
    print("\nINFO: Single BOS detected. This is CORRECT.")
else:
    print("\nINFO: No BOS at start? (Might be okay if template handles it differently, but usually we expect one)")
