"""
Simple script to load a NoBackdrop model and test its next token prediction capabilities.
"""

import argparse
import os
import sys
import torch
from transformers import AutoTokenizer

# Add the parent directory to the Python path so we can import no_backdrop
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from no_backdrop.model.hebbian_lm import HebbianLM


def main(args):
    """Main function to load model and run interactive prediction loop."""
    print("=== NoBackdrop Next Token Predictor ===")
    
    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    
    # Ensure the tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Determine device
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    # Load model from checkpoint
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        print(f"Loading model from checkpoint: {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
        
        # Extract model configuration from the checkpoint
        state_dict = checkpoint['model_state_dict']
        
        # Print a few keys to debug
        print(f"Sample keys in state_dict: {list(state_dict.keys())[:5]}")
        
        # Extract hidden size from embedding weight
        hidden_size = state_dict['embeddings.token_embeddings.weight'].shape[1]
        print(f"Detected hidden_size: {hidden_size}")
        
        # Extract number of layers by counting attention layers
        num_layers = 0
        for key in state_dict.keys():
            if 'layers.' in key and '.attention.attention.q_proj.weight' in key:
                layer_num = int(key.split('.')[1])
                num_layers = max(num_layers, layer_num + 1)
        print(f"Detected num_layers: {num_layers}")
        
        # Extract intermediate size from feed forward layer
        for key in state_dict.keys():
            if 'layers.0.feed_forward.fc1.weight' in key:
                intermediate_size = state_dict[key].shape[0]
                break
        print(f"Detected intermediate_size: {intermediate_size}")
        
        # Extract number of attention heads
        q_proj_shape = state_dict['layers.0.attention.attention.q_proj.weight'].shape
        k_proj_shape = state_dict['layers.0.attention.attention.k_proj.weight'].shape
        num_heads = q_proj_shape[0] // k_proj_shape[1]
        print(f"Detected num_heads: {num_heads}")
        
        # Create a new model with the detected configuration
        vocab_size = state_dict['embeddings.token_embeddings.weight'].shape[0]
        position_size = state_dict['embeddings.position_embeddings.weight'].shape[0]
        
        # Create model with the detected configuration
        model = HebbianLM(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=position_size,
            window_size=args.window_size,
            dropout=args.dropout,
            update_rate=args.update_rate,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        
        # Load the state dict
        try:
            model.load_state_dict(state_dict)
            print("Successfully loaded model weights")
        except Exception as e:
            print(f"Warning: Error loading model weights: {e}")
            print("Continuing with a fresh model...")
        
        # Load token contexts if available
        if 'token_contexts' in checkpoint:
            model.token_contexts = checkpoint['token_contexts']
            print("Loaded token contexts from checkpoint")
    else:
        print("No checkpoint provided. Creating a new model...")
        model = HebbianLM(
            vocab_size=tokenizer.vocab_size,
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_layers,
            num_attention_heads=args.num_heads,
            window_size=args.window_size,
            dropout=args.dropout,
            max_position_embeddings=args.max_length,
            update_rate=args.update_rate,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Move model to device
    model.to(device)
    model.eval()  # Set model to evaluation mode
    
    # Interactive prompt loop
    print("\n=== Next Token Prediction ===")
    print("Enter prompts to test the model's next token prediction (type 'exit' to quit):\n")
    
    while True:
        user_prompt = input("Your prompt> ")
        
        if user_prompt.lower() in ['exit', 'quit', 'q']:
            break
            
        if not user_prompt.strip():
            continue
            
        # Tokenize prompt
        tokenized = tokenizer(user_prompt, return_tensors="pt")
        input_ids = tokenized["input_ids"].to(device)
        
        # Get next token predictions (top-k)
        with torch.no_grad():
            # Forward pass to get logits
            outputs = model(input_ids=input_ids, update_model=False)
            next_token_logits = outputs["logits"][0, -1, :]
            
            # Try different temperature values
            for temp_name, temperature in [("Low", 0.3), ("Medium", 0.7), ("High", 1.0)]:
                print(f"\n{temp_name} Temperature (T={temperature}):")
                
                # Apply temperature scaling
                scaled_logits = next_token_logits / temperature
                
                # Get top-k predictions
                top_k = 10
                topk_values, topk_indices = torch.topk(scaled_logits, top_k)
                
                # Convert logits to probabilities
                topk_probs = torch.softmax(topk_values, dim=0)
                
                # Print top-k predictions
                for i, (token_id, prob) in enumerate(zip(topk_indices.tolist(), topk_probs.tolist())):
                    token_text = tokenizer.decode([token_id])
                    print(f"{i+1}. '{token_text}' ({prob:.4f})")
                
                # Use the top prediction to complete the sentence
                top_token_id = topk_indices[0].unsqueeze(0).unsqueeze(0)
                completed_ids = torch.cat([input_ids, top_token_id], dim=1)
                completed_text = tokenizer.decode(completed_ids[0], skip_special_tokens=True)
                print(f"\nCompleted: {completed_text}")
        
        print("\n" + "-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test next token prediction with NoBackdrop model")
    
    # Model arguments
    parser.add_argument("--tokenizer_name", type=str, default="gpt2", help="Tokenizer name")
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden size")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--window_size", type=int, default=64, help="Attention window size")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum sequence length")
    parser.add_argument("--update_rate", type=float, default=0.05, help="Fast weight update rate")
    
    # Checkpoint arguments
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to model checkpoint")
    
    # Device arguments
    parser.add_argument("--device", type=str, default=None, help="Device to use (cpu or cuda)")
    
    args = parser.parse_args()
    
    main(args)
