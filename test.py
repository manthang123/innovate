from transformers import GPT2Tokenizer, PreTrainedTokenizerFast
import os

# Path to your local tokenizer files
tokenizer_path = "./gpt2_tokenizer"

# Try to load the tokenizer from local files
try:
    tokenizer = GPT2Tokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True
    )
    print("Successfully loaded GPT-2 tokenizer from local files")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    # Fallback to a simple version if the full loading fails
    from transformers import GPT2TokenizerFast
    
    class SimpleTokenizer:
        def __init__(self):
            self.eos_token_id = 50256
            
        def encode(self, text, return_tensors=None):
            # Very simplified encoding - just for testing
            import torch
            return torch.tensor([[464, 466, 551, 50256]])  # Simplified tokens
            
        def decode(self, token_ids, skip_special_tokens=False):
            # Very simplified decoding - just for testing
            return "This is a placeholder response."
    
    tokenizer = SimpleTokenizer()
    print("Using simplified tokenizer as fallback")
