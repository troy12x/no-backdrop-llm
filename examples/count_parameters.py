"""
Script to load a NoBackdrop model and count its parameters.
"""

import argparse
import os
import sys
import torch
from transformers import AutoTokenizer

# Add the parent directory to the Python path so we can import no_backdrop
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from no_backdrop.model.hebbian_lm import HebbianLM


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_by_layer(model):
    """Count parameters by layer/component."""
    result = {}
    
    # Count embeddings
    embedding_params = sum(p.numel() for name, p in model.named_parameters() 
                          if 'embeddings' in name and p.requires_grad)
    result['embeddings'] = embedding_params
    
    # Count by layer
    layer_params = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
            
        if 'layers.' in name:
            parts = name.split('.')
            layer_num = int(parts[1])
            if layer_num not in layer_params:
                layer_params[layer_num] = 0
            layer_params[layer_num] += p.numel()
    
    for layer_num, params in sorted(layer_params.items()):
        result[f'layer_{layer_num}'] = params
    
    # Count output layer
    output_params = sum(p.numel() for name, p in model.named_parameters() 
                       if 'output' in name and p.requires_grad)
    result['output'] = output_params
    
    return result


def estimate_model_size(vocab_size, hidden_size, num_layers, num_heads, intermediate_size, max_position_embeddings):
    """Estimate the number of parameters in a transformer model."""
    # Embeddings
    token_embedding_params = vocab_size * hidden_size
    position_embedding_params = max_position_embeddings * hidden_size
    embedding_layer_norm_params = 2 * hidden_size  # gamma and beta
    
    # Each transformer layer
    head_dim = hidden_size // num_heads
    
    # Attention parameters per layer
    qkv_params = 3 * (hidden_size * hidden_size)  # Query, Key, Value projections
    output_proj_params = hidden_size * hidden_size  # Output projection
    attention_layer_norm_params = 2 * hidden_size  # gamma and beta
    
    # Feed-forward parameters per layer
    ff1_params = hidden_size * intermediate_size  # First linear layer
    ff2_params = intermediate_size * hidden_size  # Second linear layer
    ff_layer_norm_params = 2 * hidden_size  # gamma and beta
    
    # Parameters per layer
    params_per_layer = qkv_params + output_proj_params + attention_layer_norm_params + ff1_params + ff2_params + ff_layer_norm_params
    
    # Total transformer layers
    transformer_params = num_layers * params_per_layer
    
    # Output layer (LM head)
    output_params = hidden_size * vocab_size
    
    # Total parameters
    total_params = token_embedding_params + position_embedding_params + embedding_layer_norm_params + transformer_params + output_params
    
    return {
        'token_embeddings': token_embedding_params,
        'position_embeddings': position_embedding_params,
        'embedding_layer_norm': embedding_layer_norm_params,
        'transformer_layers': transformer_params,
        'transformer_params_per_layer': params_per_layer,
        'output_layer': output_params,
        'total': total_params
    }


def main(args):
    """Main function to load model and count parameters."""
    print("=== NoBackdrop Model Parameter Counter ===")
    
    # First, estimate parameters for the configuration in simple_training.py
    print("\n=== Estimating Parameters for New Configuration ===")
    vocab_size = 50257  # GPT-2 vocab size
    hidden_size = 768
    num_layers = 6  # Increased from 4 to 6
    num_heads = 12
    intermediate_size = hidden_size * 4
    max_position_embeddings = 1024  # Increased to avoid position embedding errors
    
    estimated_params = estimate_model_size(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings
    )
    
    print(f"Estimated parameters for new configuration:")
    print(f"  Token embeddings: {estimated_params['token_embeddings']:,}")
    print(f"  Position embeddings: {estimated_params['position_embeddings']:,}")
    print(f"  Embedding layer norm: {estimated_params['embedding_layer_norm']:,}")
    print(f"  Transformer layers: {estimated_params['transformer_layers']:,} ({num_layers} layers)")
    print(f"  Parameters per layer: {estimated_params['transformer_params_per_layer']:,}")
    print(f"  Output layer: {estimated_params['output_layer']:,}")
    print(f"  Total parameters: {estimated_params['total']:,} ({estimated_params['total']/1_000_000:.2f}M)")
    print("\n=== Checking Actual Model ===")
    
    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Ensure the tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Determine device
    device = "cpu"  # Use CPU for parameter counting
    
    # Load model from checkpoint
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        print(f"Loading model from checkpoint: {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
        
        # Extract model configuration from the checkpoint
        state_dict = checkpoint['model_state_dict']
        
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
    else:
        print("No checkpoint provided. Creating a new model...")
        # Use the same configuration as in our estimation
        model = HebbianLM(
            vocab_size=vocab_size,  # Use the Qwen vocab size
            hidden_size=hidden_size,  # 2048
            num_hidden_layers=num_layers,  # 24
            num_attention_heads=num_heads,  # 16
            intermediate_size=intermediate_size,  # 8192
            window_size=args.window_size,
            dropout=args.dropout,
            max_position_embeddings=max_position_embeddings,  # 512
            update_rate=args.update_rate,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Count parameters
    total_params = count_parameters(model)
    params_by_layer = count_parameters_by_layer(model)
    
    # Print parameter counts
    print("\n=== Model Parameters ===")
    print(f"Total trainable parameters: {total_params:,}")
    
    print("\nParameters by component:")
    for component, params in params_by_layer.items():
        print(f"  {component}: {params:,} ({params/total_params*100:.2f}%)")
    
    # Print model architecture summary
    print("\n=== Model Architecture ===")
    print(f"Vocabulary size: {vocab_size:,}")
    print(f"Hidden size: {hidden_size}")
    print(f"Number of layers: {num_layers}")
    print(f"Number of attention heads: {num_heads}")
    print(f"Intermediate size: {intermediate_size}")
    
    # Only print position_size if it was detected from checkpoint
    if 'position_size' in locals():
        print(f"Position embedding size: {position_size}")
    
    # Compare to standard models
    print("\n=== Comparison to Standard Models ===")
    print(f"GPT-2 Small: 124M parameters")
    print(f"GPT-2 Medium: 355M parameters")
    print(f"GPT-2 Large: 774M parameters")
    print(f"GPT-2 XL: 1.5B parameters")
    print(f"This model: {total_params/1_000_000:.2f}M parameters")
    
    # Memory usage estimate (rough approximation)
    memory_usage_fp32 = total_params * 4 / (1024 * 1024)  # MB for FP32
    memory_usage_fp16 = total_params * 2 / (1024 * 1024)  # MB for FP16
    
    print("\n=== Estimated Memory Usage ===")
    print(f"FP32 weights: {memory_usage_fp32:.2f} MB")
    print(f"FP16 weights: {memory_usage_fp16:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count parameters in NoBackdrop model")
    
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
    
    args = parser.parse_args()
    
    main(args)
