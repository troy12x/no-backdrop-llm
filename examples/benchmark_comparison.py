"""
Benchmark comparison between NoBackdrop and traditional backpropagation models.
"""

import os
import argparse
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from no_backdrop.model.hebbian_lm import HebbianLM
from no_backdrop.training.trainer import Trainer
from no_backdrop.training.data_utils import prepare_dataloaders


class PerformanceBenchmark:
    """
    Benchmark for comparing NoBackdrop with traditional backpropagation models.
    """
    
    def __init__(
        self,
        no_backdrop_model: HebbianLM,
        baseline_model: torch.nn.Module,
        tokenizer: AutoTokenizer,
        device: str = None,
    ):
        """
        Initialize the benchmark.
        
        Args:
            no_backdrop_model: NoBackdrop model
            baseline_model: Baseline model for comparison
            tokenizer: Tokenizer to use
            device: Device to use (cpu or cuda)
        """
        self.no_backdrop_model = no_backdrop_model
        self.baseline_model = baseline_model
        self.tokenizer = tokenizer
        
        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Move models to device
        self.no_backdrop_model.to(self.device)
        self.baseline_model.to(self.device)
        
        # Create trainers
        self.no_backdrop_trainer = Trainer(
            model=no_backdrop_model,
            device=self.device,
        )
    
    def benchmark_memory_usage(self) -> dict:
        """
        Benchmark memory usage of both models.
        
        Returns:
            results: Dictionary of memory usage results
        """
        # Measure memory usage for NoBackdrop model
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        # Generate random input
        input_ids = torch.randint(0, self.no_backdrop_model.vocab_size, (1, 512), device=self.device)
        attention_mask = torch.ones_like(input_ids)
        
        # Forward pass with NoBackdrop model
        self.no_backdrop_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            update_model=True,
            compute_loss=True,
        )
        
        no_backdrop_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        
        # Measure memory usage for baseline model
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        # Forward pass with baseline model
        self.baseline_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        baseline_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        
        # Calculate memory ratio
        memory_ratio = no_backdrop_memory / baseline_memory if baseline_memory > 0 else float('inf')
        
        # Collect results
        results = {
            "no_backdrop_memory_mb": no_backdrop_memory,
            "baseline_memory_mb": baseline_memory,
            "memory_ratio": memory_ratio,
        }
        
        return results
    
    def benchmark_training_speed(self, num_steps: int = 10) -> dict:
        """
        Benchmark training speed of both models.
        
        Args:
            num_steps: Number of steps to benchmark
            
        Returns:
            results: Dictionary of training speed results
        """
        # Generate random input
        batch_size = 4
        seq_len = 128
        input_ids = torch.randint(0, self.no_backdrop_model.vocab_size, (batch_size, seq_len), device=self.device)
        attention_mask = torch.ones_like(input_ids)
        
        # Prepare batch
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        
        # Benchmark NoBackdrop model
        self.no_backdrop_model.train()
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(num_steps):
            self.no_backdrop_trainer.train_step(batch)
        
        torch.cuda.synchronize()
        no_backdrop_time = time.time() - start_time
        
        # Benchmark baseline model
        self.baseline_model.train()
        optimizer = torch.optim.AdamW(self.baseline_model.parameters(), lr=5e-5)
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(num_steps):
            # Forward pass
            outputs = self.baseline_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            
            # Backward pass
            loss = outputs.loss
            loss.backward()
            
            # Update parameters
            optimizer.step()
            optimizer.zero_grad()
        
        torch.cuda.synchronize()
        baseline_time = time.time() - start_time
        
        # Calculate speed ratio
        speed_ratio = baseline_time / no_backdrop_time if no_backdrop_time > 0 else float('inf')
        
        # Collect results
        results = {
            "no_backdrop_time_sec": no_backdrop_time,
            "baseline_time_sec": baseline_time,
            "steps_per_second_no_backdrop": num_steps / no_backdrop_time,
            "steps_per_second_baseline": num_steps / baseline_time,
            "speed_ratio": speed_ratio,
        }
        
        return results
    
    def benchmark_single_batch_learning(self, text: str) -> dict:
        """
        Benchmark learning from a single batch of data.
        
        Args:
            text: Text to learn from
            
        Returns:
            results: Dictionary of learning results
        """
        # Tokenize text
        tokenized = self.tokenizer(text, return_tensors="pt")
        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = torch.ones_like(input_ids)
        
        # Prepare batch
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        
        # Measure initial perplexity for NoBackdrop model
        self.no_backdrop_model.eval()
        with torch.no_grad():
            outputs = self.no_backdrop_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                update_model=False,
                compute_loss=True,
            )
        
        initial_loss_no_backdrop = outputs["loss"].item()
        initial_perplexity_no_backdrop = self.no_backdrop_trainer._compute_perplexity(outputs["loss"])
        
        # Measure initial perplexity for baseline model
        self.baseline_model.eval()
        with torch.no_grad():
            outputs = self.baseline_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
        
        initial_loss_baseline = outputs.loss.item()
        initial_perplexity_baseline = torch.exp(outputs.loss).item()
        
        # Train NoBackdrop model on a single batch
        self.no_backdrop_model.train()
        metrics = self.no_backdrop_trainer.train_step(batch)
        
        # Train baseline model on a single batch
        self.baseline_model.train()
        optimizer = torch.optim.AdamW(self.baseline_model.parameters(), lr=5e-5)
        
        outputs = self.baseline_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )
        
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Measure final perplexity for NoBackdrop model
        self.no_backdrop_model.eval()
        with torch.no_grad():
            outputs = self.no_backdrop_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                update_model=False,
                compute_loss=True,
            )
        
        final_loss_no_backdrop = outputs["loss"].item()
        final_perplexity_no_backdrop = self.no_backdrop_trainer._compute_perplexity(outputs["loss"])
        
        # Measure final perplexity for baseline model
        self.baseline_model.eval()
        with torch.no_grad():
            outputs = self.baseline_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
        
        final_loss_baseline = outputs.loss.item()
        final_perplexity_baseline = torch.exp(outputs.loss).item()
        
        # Calculate improvement
        improvement_no_backdrop = initial_perplexity_no_backdrop - final_perplexity_no_backdrop
        improvement_baseline = initial_perplexity_baseline - final_perplexity_baseline
        
        # Collect results
        results = {
            "initial_loss_no_backdrop": initial_loss_no_backdrop,
            "initial_perplexity_no_backdrop": initial_perplexity_no_backdrop,
            "final_loss_no_backdrop": final_loss_no_backdrop,
            "final_perplexity_no_backdrop": final_perplexity_no_backdrop,
            "improvement_no_backdrop": improvement_no_backdrop,
            "initial_loss_baseline": initial_loss_baseline,
            "initial_perplexity_baseline": initial_perplexity_baseline,
            "final_loss_baseline": final_loss_baseline,
            "final_perplexity_baseline": final_perplexity_baseline,
            "improvement_baseline": improvement_baseline,
        }
        
        return results
    
    def benchmark_adaptation(self, adaptation_text: str, test_prompts: list) -> dict:
        """
        Benchmark adaptation to new information.
        
        Args:
            adaptation_text: Text to adapt to
            test_prompts: List of prompts to test adaptation
            
        Returns:
            results: Dictionary of adaptation results
        """
        # Tokenize adaptation text
        tokenized = self.tokenizer(adaptation_text, return_tensors="pt")
        adaptation_ids = tokenized["input_ids"].to(self.device)
        attention_mask = torch.ones_like(adaptation_ids)
        
        # Record initial generations
        initial_generations_no_backdrop = []
        initial_generations_baseline = []
        
        for prompt in test_prompts:
            # Generate with NoBackdrop model
            tokenized = self.tokenizer(prompt, return_tensors="pt")
            input_ids = tokenized["input_ids"].to(self.device)
            
            self.no_backdrop_model.eval()
            with torch.no_grad():
                generated_ids = self.no_backdrop_model.generate(
                    input_ids=input_ids,
                    max_length=50,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.9,
                    update_model=False,
                )
            
            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            initial_generations_no_backdrop.append(generated_text)
            
            # Generate with baseline model
            self.baseline_model.eval()
            with torch.no_grad():
                generated_ids = self.baseline_model.generate(
                    input_ids=input_ids,
                    max_length=50,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.9,
                )
            
            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            initial_generations_baseline.append(generated_text)
        
        # Adapt NoBackdrop model
        self.no_backdrop_model.eval()
        with torch.no_grad():
            outputs = self.no_backdrop_model(
                input_ids=adaptation_ids,
                attention_mask=attention_mask,
                update_model=True,
                compute_loss=False,
            )
        
        # Adapt baseline model (fine-tuning)
        self.baseline_model.train()
        optimizer = torch.optim.AdamW(self.baseline_model.parameters(), lr=5e-5)
        
        outputs = self.baseline_model(
            input_ids=adaptation_ids,
            attention_mask=attention_mask,
            labels=adaptation_ids,
        )
        
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Record adapted generations
        adapted_generations_no_backdrop = []
        adapted_generations_baseline = []
        
        for prompt in test_prompts:
            # Generate with NoBackdrop model
            tokenized = self.tokenizer(prompt, return_tensors="pt")
            input_ids = tokenized["input_ids"].to(self.device)
            
            self.no_backdrop_model.eval()
            with torch.no_grad():
                generated_ids = self.no_backdrop_model.generate(
                    input_ids=input_ids,
                    max_length=50,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.9,
                    update_model=False,
                )
            
            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            adapted_generations_no_backdrop.append(generated_text)
            
            # Generate with baseline model
            self.baseline_model.eval()
            with torch.no_grad():
                generated_ids = self.baseline_model.generate(
                    input_ids=input_ids,
                    max_length=50,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.9,
                )
            
            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            adapted_generations_baseline.append(generated_text)
        
        # Collect results
        results = {
            "initial_generations_no_backdrop": initial_generations_no_backdrop,
            "initial_generations_baseline": initial_generations_baseline,
            "adapted_generations_no_backdrop": adapted_generations_no_backdrop,
            "adapted_generations_baseline": adapted_generations_baseline,
        }
        
        return results
    
    def plot_results(self, results: dict, output_dir: str = "./results"):
        """
        Plot benchmark results.
        
        Args:
            results: Dictionary of benchmark results
            output_dir: Directory to save plots
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot memory usage
        if "memory_usage" in results:
            memory_usage = results["memory_usage"]
            
            plt.figure(figsize=(10, 6))
            plt.bar(["NoBackdrop", "Baseline"], 
                   [memory_usage["no_backdrop_memory_mb"], memory_usage["baseline_memory_mb"]])
            plt.title("Memory Usage Comparison")
            plt.ylabel("Memory Usage (MB)")
            plt.savefig(f"{output_dir}/memory_usage.png")
            plt.close()
        
        # Plot training speed
        if "training_speed" in results:
            training_speed = results["training_speed"]
            
            plt.figure(figsize=(10, 6))
            plt.bar(["NoBackdrop", "Baseline"], 
                   [training_speed["steps_per_second_no_backdrop"], 
                    training_speed["steps_per_second_baseline"]])
            plt.title("Training Speed Comparison")
            plt.ylabel("Steps per Second")
            plt.savefig(f"{output_dir}/training_speed.png")
            plt.close()
        
        # Plot single batch learning
        if "single_batch_learning" in results:
            learning = results["single_batch_learning"]
            
            plt.figure(figsize=(10, 6))
            plt.bar(["NoBackdrop Initial", "NoBackdrop Final", "Baseline Initial", "Baseline Final"], 
                   [learning["initial_perplexity_no_backdrop"], learning["final_perplexity_no_backdrop"],
                    learning["initial_perplexity_baseline"], learning["final_perplexity_baseline"]])
            plt.title("Single Batch Learning Comparison")
            plt.ylabel("Perplexity")
            plt.savefig(f"{output_dir}/single_batch_learning.png")
            plt.close()


def create_baseline_model(model_name: str, vocab_size: int = None):
    """
    Create a baseline model for comparison.
    
    Args:
        model_name: Model name or path
        vocab_size: Vocabulary size
        
    Returns:
        model: Baseline model
    """
    try:
        # Try to load from Hugging Face
        model = AutoModelForCausalLM.from_pretrained(model_name)
        return model
    except:
        # Create a simple GPT-2 model
        from transformers import GPT2Config, GPT2LMHeadModel
        
        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=512,
            n_ctx=512,
            n_embd=256,
            n_layer=4,
            n_head=8,
        )
        
        model = GPT2LMHeadModel(config)
        return model


def main(args):
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    
    # Ensure the tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create NoBackdrop model
    no_backdrop_model = HebbianLM(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        window_size=args.window_size,
        dropout=args.dropout,
        max_position_embeddings=args.max_length,
        update_rate=args.update_rate,
        use_fast_weights=True,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id,
        eos_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id,
    )
    
    # Create baseline model
    baseline_model = create_baseline_model(args.baseline_model, vocab_size=len(tokenizer))
    
    # Load checkpoint if provided
    if args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
        no_backdrop_model.load_state_dict(checkpoint["model_state_dict"])
        if "token_contexts" in checkpoint:
            no_backdrop_model.token_contexts = checkpoint["token_contexts"]
        print(f"Loaded checkpoint from {args.checkpoint_path}")
    
    # Create benchmark
    benchmark = PerformanceBenchmark(
        no_backdrop_model=no_backdrop_model,
        baseline_model=baseline_model,
        tokenizer=tokenizer,
        device=args.device,
    )
    
    # Run benchmarks
    results = {}
    
    # Memory usage benchmark
    if args.benchmark_memory:
        print("Running memory usage benchmark...")
        memory_results = benchmark.benchmark_memory_usage()
        results["memory_usage"] = memory_results
        
        print(f"NoBackdrop Memory: {memory_results['no_backdrop_memory_mb']:.2f} MB")
        print(f"Baseline Memory: {memory_results['baseline_memory_mb']:.2f} MB")
        print(f"Memory Ratio: {memory_results['memory_ratio']:.2f}x")
    
    # Training speed benchmark
    if args.benchmark_speed:
        print("Running training speed benchmark...")
        speed_results = benchmark.benchmark_training_speed(num_steps=args.num_steps)
        results["training_speed"] = speed_results
        
        print(f"NoBackdrop Training Speed: {speed_results['steps_per_second_no_backdrop']:.2f} steps/sec")
        print(f"Baseline Training Speed: {speed_results['steps_per_second_baseline']:.2f} steps/sec")
        print(f"Speed Ratio: {speed_results['speed_ratio']:.2f}x")
    
    # Single batch learning benchmark
    if args.benchmark_learning:
        print("Running single batch learning benchmark...")
        
        # Load sample text
        if args.dataset_name:
            dataset = load_dataset(args.dataset_name, args.dataset_config)
            text = dataset["train"]["text"][0]
        else:
            text = "The quick brown fox jumps over the lazy dog. This is a sample text for testing single batch learning."
        
        learning_results = benchmark.benchmark_single_batch_learning(text)
        results["single_batch_learning"] = learning_results
        
        print(f"NoBackdrop Initial Perplexity: {learning_results['initial_perplexity_no_backdrop']:.2f}")
        print(f"NoBackdrop Final Perplexity: {learning_results['final_perplexity_no_backdrop']:.2f}")
        print(f"NoBackdrop Improvement: {learning_results['improvement_no_backdrop']:.2f}")
        print(f"Baseline Initial Perplexity: {learning_results['initial_perplexity_baseline']:.2f}")
        print(f"Baseline Final Perplexity: {learning_results['final_perplexity_baseline']:.2f}")
        print(f"Baseline Improvement: {learning_results['improvement_baseline']:.2f}")
    
    # Adaptation benchmark
    if args.benchmark_adaptation:
        print("Running adaptation benchmark...")
        
        adaptation_text = "The capital of France is Paris. The capital of Italy is Rome. The capital of Spain is Madrid."
        test_prompts = [
            "The capital of France is",
            "The capital of Italy is",
            "The capital of Spain is",
        ]
        
        adaptation_results = benchmark.benchmark_adaptation(adaptation_text, test_prompts)
        results["adaptation"] = adaptation_results
        
        for i, prompt in enumerate(test_prompts):
            print(f"\nPrompt: {prompt}")
            print(f"NoBackdrop Initial: {adaptation_results['initial_generations_no_backdrop'][i]}")
            print(f"NoBackdrop Adapted: {adaptation_results['adapted_generations_no_backdrop'][i]}")
            print(f"Baseline Initial: {adaptation_results['initial_generations_baseline'][i]}")
            print(f"Baseline Adapted: {adaptation_results['adapted_generations_baseline'][i]}")
    
    # Plot results
    if args.plot_results:
        print("Plotting results...")
        benchmark.plot_results(results, output_dir=args.output_dir)
    
    print("Benchmark complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark NoBackdrop against traditional models")
    
    # Model arguments
    parser.add_argument("--tokenizer_name", type=str, default="gpt2", help="Tokenizer name")
    parser.add_argument("--baseline_model", type=str, default="gpt2", help="Baseline model name")
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden size")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--window_size", type=int, default=64, help="Attention window size")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--update_rate", type=float, default=0.01, help="Fast weight update rate")
    
    # Benchmark arguments
    parser.add_argument("--benchmark_memory", action="store_true", help="Benchmark memory usage")
    parser.add_argument("--benchmark_speed", action="store_true", help="Benchmark training speed")
    parser.add_argument("--benchmark_learning", action="store_true", help="Benchmark single batch learning")
    parser.add_argument("--benchmark_adaptation", action="store_true", help="Benchmark adaptation")
    parser.add_argument("--num_steps", type=int, default=10, help="Number of steps for speed benchmark")
    parser.add_argument("--dataset_name", type=str, default=None, help="Dataset name for learning benchmark")
    parser.add_argument("--dataset_config", type=str, default=None, help="Dataset configuration")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./benchmark_results", help="Output directory")
    parser.add_argument("--plot_results", action="store_true", help="Plot benchmark results")
    
    # Checkpoint arguments
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to model checkpoint")
    
    # Device arguments
    parser.add_argument("--device", type=str, default=None, help="Device to use (cpu or cuda)")
    
    args = parser.parse_args()
    
    # Enable all benchmarks if none specified
    if not (args.benchmark_memory or args.benchmark_speed or 
            args.benchmark_learning or args.benchmark_adaptation):
        args.benchmark_memory = True
        args.benchmark_speed = True
        args.benchmark_learning = True
        args.benchmark_adaptation = True
    
    main(args)
