"""
Demonstration of the evaluation and visualization utilities for NoBackdrop models.

This script shows how to:
1. Load a pre-trained HebbianLM model
2. Evaluate its performance metrics (memory usage, inference speed, training speed)
3. Test its adaptation capabilities
4. Visualize training metrics and adaptation results
"""

import os
import sys
import torch
import argparse
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import numpy as np
import time

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from no_backdrop.model.hebbian_lm import HebbianLM
from no_backdrop.training.trainer import Trainer
from no_backdrop.training.data_utils import prepare_dataloaders, TextDataset
from no_backdrop.utils.evaluation import PerformanceMetrics, AdaptationEvaluator, evaluate_model_capabilities
from no_backdrop.utils.visualization import TrainingVisualizer, TokenStatisticsVisualizer, AttentionVisualizer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation demo for NoBackdrop models")
    parser.add_argument("--model_path", type=str, default=None, help="Path to pre-trained model checkpoint")
    parser.add_argument("--dataset", type=str, default="wikitext", help="Dataset to use for evaluation")
    parser.add_argument("--dataset_subset", type=str, default="wikitext-2-raw-v1", help="Dataset subset")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results", help="Directory to save results")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cpu or cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def create_model(vocab_size, device):
    """Create a new HebbianLM model if no pre-trained model is provided."""
    print("Creating a new HebbianLM model...")
    
    # Create model with explicit parameters
    model = HebbianLM(
        vocab_size=vocab_size,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=512,
        window_size=64,  # Local attention window size
        dropout=0.1,
        max_position_embeddings=512,
        update_rate=0.01,  # Fast weight update rate
        use_fast_weights=True,
        pad_token_id=0,
        bos_token_id=None,
        eos_token_id=None,
        use_frequency_stats=True,
        use_context_modulation=True
    )
    
    # Move model to device
    model.to(device)
    
    return model


def evaluate_performance(model, tokenizer, device, output_dir):
    """Evaluate model performance metrics."""
    print("\n=== Evaluating Model Performance ===")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate random input for benchmarking
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, model.vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids)
    
    # Evaluate memory usage
    print("Measuring memory usage...")
    memory_usage = PerformanceMetrics.compute_memory_usage(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        update_model=True,
    )
    print(f"Memory usage: {memory_usage:.2f} MB")
    
    # Evaluate inference speed
    print("Measuring inference speed...")
    inference_speed = PerformanceMetrics.compute_inference_speed(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        num_runs=5,
    )
    print(f"Inference speed: {inference_speed:.2f} tokens/sec")
    
    # Evaluate training speed
    print("Measuring training speed...")
    training_speed = PerformanceMetrics.compute_training_speed(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        num_runs=5,
    )
    print(f"Training speed: {training_speed:.2f} tokens/sec")
    
    # Plot performance metrics
    metrics = {
        "Memory Usage (MB)": memory_usage,
        "Inference Speed (tokens/sec)": inference_speed,
        "Training Speed (tokens/sec)": training_speed,
    }
    
    plt.figure(figsize=(10, 6))
    plt.bar(metrics.keys(), metrics.values())
    plt.title("NoBackdrop Model Performance Metrics")
    plt.ylabel("Value")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "performance_metrics.png"))
    
    return metrics


def evaluate_adaptation(model, tokenizer, device, output_dir):
    """Evaluate model adaptation capabilities."""
    print("\n=== Evaluating Model Adaptation Capabilities ===")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create adaptation evaluator
    adaptation_evaluator = AdaptationEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=device,
    )
    
    # Test 1: Simple adaptation
    print("Testing simple adaptation...")
    adaptation_text = "The capital of France is Paris. The capital of Italy is Rome. The capital of Spain is Madrid."
    test_prompts = ["The capital of France is", "The capital of Italy is", "The capital of Spain is"]
    
    simple_results = adaptation_evaluator.evaluate_adaptation(
        adaptation_text=adaptation_text,
        test_prompts=test_prompts,
        max_length=30,
        temperature=0.7,
    )
    
    # Print results
    print("\nSimple Adaptation Results:")
    for i, prompt in enumerate(test_prompts):
        print(f"Prompt: {prompt}")
        print(f"  Before: {simple_results['initial_generations'][i]}")
        print(f"  After:  {simple_results['adapted_generations'][i]}")
    
    # Plot results
    adaptation_evaluator.plot_adaptation_results(
        results=simple_results,
        output_path=os.path.join(output_dir, "simple_adaptation.png"),
    )
    
    # Test 2: Continuous adaptation
    print("\nTesting continuous adaptation...")
    adaptation_texts = [
        "The largest planet in our solar system is Jupiter.",
        "The closest planet to the Sun is Mercury.",
        "The planet with rings is Saturn.",
    ]
    test_prompts = [
        "The largest planet in our solar system is",
        "The closest planet to the Sun is",
        "The planet with rings is",
    ]
    
    continuous_results = adaptation_evaluator.evaluate_continuous_adaptation(
        adaptation_texts=adaptation_texts,
        test_prompts=test_prompts,
        max_length=30,
        temperature=0.7,
    )
    
    # Print results
    print("\nContinuous Adaptation Results:")
    for i, prompt in enumerate(test_prompts):
        print(f"Prompt: {prompt}")
        print(f"  Initial: {continuous_results['initial_generations'][i]}")
        for j, adaptation_text in enumerate(adaptation_texts):
            print(f"  After '{adaptation_text[:20]}...': {continuous_results['all_adapted_generations'][j][i]}")
    
    # Plot results
    adaptation_evaluator.plot_adaptation_results(
        results=continuous_results,
        output_path=os.path.join(output_dir, "continuous_adaptation.png"),
    )
    
    return {
        "simple_adaptation": simple_results,
        "continuous_adaptation": continuous_results,
    }


def simulate_training(model, tokenizer, device, output_dir):
    """Simulate a short training run to demonstrate visualization utilities."""
    print("\n=== Simulating Training for Visualization Demo ===")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create training visualizer
    visualizer = TrainingVisualizer(
        output_dir=output_dir,
        experiment_name="demo_training",
    )
    
    # Generate synthetic training data
    print("Generating synthetic training data...")
    num_steps = 100
    
    # Simulate training metrics
    train_loss = np.linspace(5.0, 2.0, num_steps) + np.random.normal(0, 0.2, num_steps)
    train_perplexity = np.exp(train_loss)
    
    # Simulate evaluation metrics (every 10 steps)
    eval_steps = list(range(0, num_steps, 10))
    eval_loss = np.linspace(5.2, 2.2, len(eval_steps)) + np.random.normal(0, 0.1, len(eval_steps))
    eval_perplexity = np.exp(eval_loss)
    
    # Log metrics
    print("Logging training metrics...")
    for i in range(num_steps):
        train_metrics = {
            "loss": float(train_loss[i]),
            "perplexity": float(train_perplexity[i]),
        }
        visualizer.log_metrics(train_metrics, step=i)
        
        # Log evaluation metrics every 10 steps
        if i in eval_steps:
            eval_idx = eval_steps.index(i)
            eval_metrics = {
                "loss": float(eval_loss[eval_idx]),
                "perplexity": float(eval_perplexity[eval_idx]),
            }
            visualizer.log_metrics(eval_metrics, step=i, is_eval=True)
    
    # Plot metrics
    print("Plotting training metrics...")
    visualizer.plot_loss_curve(show=False)
    visualizer.plot_perplexity_curve(show=False)
    visualizer.plot_all_metrics(show=False)
    
    # Save metrics
    visualizer.save_metrics()
    
    print(f"Training visualization saved to {os.path.join(output_dir, 'demo_training')}")
    
    return visualizer


def visualize_token_statistics(model, tokenizer, device, output_dir):
    """Visualize token statistics and embeddings."""
    print("\n=== Visualizing Token Statistics ===")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate synthetic token counts
    print("Generating synthetic token statistics...")
    vocab_size = model.vocab_size
    token_ids = list(range(100))  # Use first 100 tokens for visualization
    
    # Simulate token counts (power law distribution)
    token_counts = {}
    for i, token_id in enumerate(token_ids):
        token_counts[token_id] = int(10000 * np.power(0.9, i))
    
    # Simulate token update rates
    token_update_rates = {}
    for token_id in token_ids:
        token_update_rates[token_id] = np.random.beta(2, 5)  # Beta distribution
    
    # Plot token frequency
    print("Plotting token frequency...")
    TokenStatisticsVisualizer.plot_token_frequency(
        token_counts=token_counts,
        tokenizer=tokenizer,
        top_k=30,
        save_path=os.path.join(output_dir, "token_frequency.png"),
        show=False,
    )
    
    # Plot token update rates
    print("Plotting token update rates...")
    TokenStatisticsVisualizer.plot_token_update_rates(
        token_update_rates=token_update_rates,
        tokenizer=tokenizer,
        top_k=30,
        save_path=os.path.join(output_dir, "token_update_rates.png"),
        show=False,
    )
    
    # Plot embedding PCA (if model has embeddings)
    try:
        print("Plotting token embeddings PCA...")
        embeddings = model.get_input_embeddings().weight.data
        TokenStatisticsVisualizer.plot_embedding_pca(
            embeddings=embeddings,
            token_ids=token_ids,
            tokenizer=tokenizer,
            top_k=50,
            save_path=os.path.join(output_dir, "token_embeddings_pca.png"),
            show=False,
        )
    except Exception as e:
        print(f"Could not plot embeddings PCA: {e}")
    
    print(f"Token statistics visualizations saved to {output_dir}")


def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Determine device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Load or create model
    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading pre-trained model from {args.model_path}...")
        
        # Load the checkpoint to extract configuration
        checkpoint = torch.load(args.model_path, map_location=device)
        
        # Extract model configuration from the first layer parameters
        state_dict = checkpoint['model_state_dict']
        
        # Print some keys to debug
        print("Checkpoint keys (first 10):")
        for i, key in enumerate(list(state_dict.keys())[:10]):
            print(f"  {key}")
        
        # Determine model dimensions from the checkpoint
        position_emb_shape = state_dict['embeddings.position_embeddings.weight'].shape
        max_position_embeddings = position_emb_shape[0]
        hidden_size = position_emb_shape[1]
        
        # Get intermediate size from feed forward layer
        intermediate_size = state_dict['layers.0.feed_forward.fc1.bias'].shape[0]
        
        # Count number of layers
        num_layers = 0
        while f'layers.{num_layers}.feed_forward.fc2.bias' in state_dict:
            num_layers += 1
        
        # Get number of attention heads from the q_proj weight shape
        attention_weight_shape = state_dict['layers.0.attention.attention.q_proj.weight'].shape
        num_heads = 8  # Default value
        if attention_weight_shape[0] % hidden_size == 0:
            num_heads = attention_weight_shape[0] // hidden_size
        
        print(f"Detected model config: hidden_size={hidden_size}, layers={num_layers}, heads={num_heads}, intermediate_size={intermediate_size}")
        
        # Create a model with matching architecture
        model = HebbianLM(
            vocab_size=tokenizer.vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=intermediate_size,
            window_size=64,  # This should match the training config
            dropout=0.1,
            max_position_embeddings=max_position_embeddings,
            update_rate=0.01,  # Fast weight update rate
            use_fast_weights=True,
            pad_token_id=0,
            bos_token_id=None,
            eos_token_id=None,
            use_frequency_stats=True,
            use_context_modulation=True
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
        
        # Move model to device
        model.to(device)
    else:
        model = create_model(tokenizer.vocab_size, device)
    
    # Evaluate model performance
    performance_metrics = evaluate_performance(
        model=model,
        tokenizer=tokenizer,
        device=device,
        output_dir=os.path.join(args.output_dir, "performance"),
    )
    
    # Evaluate adaptation capabilities
    adaptation_results = evaluate_adaptation(
        model=model,
        tokenizer=tokenizer,
        device=device,
        output_dir=os.path.join(args.output_dir, "adaptation"),
    )
    
    # Simulate training for visualization demo
    training_visualizer = simulate_training(
        model=model,
        tokenizer=tokenizer,
        device=device,
        output_dir=os.path.join(args.output_dir, "training"),
    )
    
    # Visualize token statistics
    visualize_token_statistics(
        model=model,
        tokenizer=tokenizer,
        device=device,
        output_dir=os.path.join(args.output_dir, "tokens"),
    )
    
    print(f"\nEvaluation complete! Results saved to {args.output_dir}")
    print("To view the results, check the PNG files in the output directory.")


if __name__ == "__main__":
    main()
