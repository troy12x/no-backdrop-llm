"""
Real-time adaptation demonstration for NoBackdrop models.

This script demonstrates the NoBackdrop model's ability to adapt to streaming text
in real-time without backpropagation. It shows how the model can:

1. Learn from new information as it arrives
2. Incorporate this information into its predictions immediately
3. Adapt to changing contexts over time
4. Maintain performance with minimal memory footprint

The demo simulates a streaming scenario where new facts arrive sequentially,
and the model must adapt its knowledge and predictions accordingly.
"""

import os
import sys
import torch
import argparse
import time
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from no_backdrop.model.hebbian_lm import HebbianLM
from no_backdrop.utils.evaluation import AdaptationEvaluator
from no_backdrop.utils.visualization import TrainingVisualizer


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time adaptation demo for NoBackdrop models")
    parser.add_argument("--model_path", type=str, default=None, help="Path to pre-trained model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./real_time_results", help="Directory to save results")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cpu or cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    return parser.parse_args()


def create_model(vocab_size, device):
    """Create a new HebbianLM model if no pre-trained model is provided."""
    print("Creating a new HebbianLM model...")
    
    # Model configuration
    config = {
        "vocab_size": vocab_size,
        "hidden_size": 256,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "intermediate_size": 512,
        "hidden_act": "gelu",
        "max_position_embeddings": 512,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-12,
        "pad_token_id": 0,
        "fast_update_rate": 0.1,
        "slow_update_rate": 1e-4,
        "token_context_window": 100,
    }
    
    # Create model
    model = HebbianLM(config)
    model.to(device)
    
    return model


def simulate_streaming_scenario(model, tokenizer, device, output_dir):
    """
    Simulate a streaming scenario where new facts arrive sequentially.
    
    This demonstrates the model's ability to adapt to new information in real-time.
    """
    print("\n=== Simulating Streaming Scenario ===")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create adaptation evaluator
    evaluator = AdaptationEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=device,
    )
    
    # Define streaming facts (arriving sequentially)
    streaming_facts = [
        "The capital of France is Paris.",
        "The Eiffel Tower is located in Paris, France.",
        "Paris is known as the City of Light.",
        "The Seine River flows through Paris.",
        "The Louvre Museum in Paris houses the Mona Lisa painting.",
        "The Notre-Dame Cathedral is a famous landmark in Paris.",
        "Paris hosted the Summer Olympics in 1900 and 1924.",
        "Paris will host the Summer Olympics again in 2024.",
        "The Paris Metro is one of the oldest subway systems in the world.",
        "Paris has a population of over 2 million people."
    ]
    
    # Define test prompts to evaluate adaptation
    test_prompts = [
        "The capital of France is",
        "Paris is known as",
        "The Eiffel Tower is located in",
        "The Seine River flows through",
        "The Louvre Museum houses",
        "Paris will host the Olympics in",
    ]
    
    # Track adaptation progress
    adaptation_progress = {
        "facts": streaming_facts,
        "test_prompts": test_prompts,
        "generations": []
    }
    
    # Initial generation (before any adaptation)
    print("\nInitial generations (before adaptation):")
    initial_generations = []
    for prompt in test_prompts:
        # Generate with model
        tokenized = tokenizer(prompt, return_tensors="pt")
        input_ids = tokenized["input_ids"].to(device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                max_length=30,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                update_model=False,
            )
        
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        initial_generations.append(generated_text)
        print(f"  Prompt: {prompt}")
        print(f"  Generated: {generated_text}")
    
    adaptation_progress["generations"].append(initial_generations)
    
    # Process streaming facts
    print("\nProcessing streaming facts...")
    for i, fact in enumerate(tqdm(streaming_facts)):
        print(f"\nFact {i+1}: {fact}")
        
        # Tokenize fact
        tokenized = tokenizer(fact, return_tensors="pt")
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = torch.ones_like(input_ids)
        
        # Update model with new fact
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                update_model=True,
                compute_loss=False,
            )
        
        # Generate after adaptation
        print("\nGenerations after this fact:")
        adapted_generations = []
        for prompt in test_prompts:
            # Generate with model
            tokenized = tokenizer(prompt, return_tensors="pt")
            input_ids = tokenized["input_ids"].to(device)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=input_ids,
                    max_length=30,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.9,
                    update_model=False,
                )
            
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            adapted_generations.append(generated_text)
            print(f"  Prompt: {prompt}")
            print(f"  Generated: {generated_text}")
        
        adaptation_progress["generations"].append(adapted_generations)
        
        # Pause to simulate streaming
        time.sleep(0.5)
    
    # Visualize adaptation progress
    visualize_adaptation_progress(adaptation_progress, test_prompts, output_dir)
    
    return adaptation_progress


def visualize_adaptation_progress(adaptation_progress, test_prompts, output_dir):
    """Visualize how the model's generations evolve as it adapts to new information."""
    print("\n=== Visualizing Adaptation Progress ===")
    
    facts = adaptation_progress["facts"]
    generations = adaptation_progress["generations"]
    
    # Create a figure for each test prompt
    for i, prompt in enumerate(test_prompts):
        plt.figure(figsize=(12, 8))
        
        # Extract generations for this prompt at each step
        prompt_generations = [gen_set[i] for gen_set in generations]
        
        # Plot as a heatmap-like visualization
        plt.imshow([[j] for j in range(len(prompt_generations))], 
                   aspect='auto', cmap='viridis', alpha=0.1)
        
        # Add text for each generation
        for j, gen in enumerate(prompt_generations):
            # Truncate generation for display
            display_text = gen[:50] + "..." if len(gen) > 50 else gen
            
            # Add fact that was just processed (except for initial generation)
            if j > 0:
                fact_text = f"Fact: {facts[j-1]}"
                plt.text(0.5, j, fact_text, fontsize=10, ha='center', va='center', 
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            
            # Add generation
            plt.text(1.5, j, display_text, fontsize=10, ha='left', va='center')
        
        plt.title(f"Adaptation Progress for Prompt: '{prompt}'")
        plt.xlabel("Generated Text")
        plt.ylabel("Adaptation Step")
        plt.yticks(range(len(prompt_generations)), 
                   ["Initial"] + [f"After Fact {j+1}" for j in range(len(facts))])
        plt.xticks([])
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(output_dir, f"adaptation_progress_prompt_{i}.png"))
        plt.close()
    
    # Create a summary visualization
    plt.figure(figsize=(15, 10))
    
    # For each prompt, show initial and final generation
    for i, prompt in enumerate(test_prompts):
        initial_gen = generations[0][i]
        final_gen = generations[-1][i]
        
        # Truncate for display
        initial_display = initial_gen[:50] + "..." if len(initial_gen) > 50 else initial_gen
        final_display = final_gen[:50] + "..." if len(final_gen) > 50 else final_gen
        
        # Plot
        plt.text(0.05, 0.95 - i*0.15, f"Prompt: {prompt}", fontsize=12, ha='left', va='top')
        plt.text(0.1, 0.91 - i*0.15, f"Initial: {initial_display}", fontsize=10, ha='left', va='top')
        plt.text(0.1, 0.87 - i*0.15, f"Final: {final_display}", fontsize=10, ha='left', va='top', 
                 color='green')
    
    plt.title("Summary of Adaptation Progress")
    plt.axis('off')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, "adaptation_summary.png"))
    plt.close()


def interactive_mode(model, tokenizer, device):
    """
    Run in interactive mode where the user can input facts and test prompts.
    
    This allows for direct experimentation with the model's adaptation capabilities.
    """
    print("\n=== Interactive Mode ===")
    print("Enter facts to teach the model, and test how it adapts.")
    print("Type 'test' to test the model, 'reset' to reset the model, or 'exit' to quit.")
    
    # Track facts and prompts
    facts = []
    
    while True:
        print("\nOptions:")
        print("1. Enter a fact to teach the model")
        print("2. Test the model with a prompt")
        print("3. Reset the model")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == "1":
            # Get fact from user
            fact = input("\nEnter a fact to teach the model: ")
            facts.append(fact)
            
            # Tokenize fact
            tokenized = tokenizer(fact, return_tensors="pt")
            input_ids = tokenized["input_ids"].to(device)
            attention_mask = torch.ones_like(input_ids)
            
            # Update model with new fact
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    update_model=True,
                    compute_loss=False,
                )
            
            print(f"Model has learned: '{fact}'")
            
        elif choice == "2":
            # Get prompt from user
            prompt = input("\nEnter a prompt to test the model: ")
            
            # Generate with model
            tokenized = tokenizer(prompt, return_tensors="pt")
            input_ids = tokenized["input_ids"].to(device)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=input_ids,
                    max_length=50,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.9,
                    update_model=False,
                )
            
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            print(f"\nGenerated: {generated_text}")
            
        elif choice == "3":
            # Reset model
            print("\nResetting model...")
            model.reset_fast_weights()
            facts = []
            print("Model reset complete.")
            
        elif choice == "4":
            # Exit
            print("\nExiting interactive mode.")
            break
            
        else:
            print("\nInvalid choice. Please enter a number between 1 and 4.")


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
        model = HebbianLM.from_pretrained(args.model_path)
        model.to(device)
    else:
        model = create_model(tokenizer.vocab_size, device)
    
    # Run in interactive mode if requested
    if args.interactive:
        interactive_mode(model, tokenizer, device)
    else:
        # Run streaming scenario simulation
        adaptation_progress = simulate_streaming_scenario(
            model=model,
            tokenizer=tokenizer,
            device=device,
            output_dir=args.output_dir,
        )
    
    print(f"\nDemo complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
