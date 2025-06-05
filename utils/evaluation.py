"""
Evaluation utilities for NoBackdrop models.
"""

import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union, Any
from transformers import PreTrainedTokenizer

from ..model.hebbian_lm import HebbianLM


class PerformanceMetrics:
    """
    Class for computing and tracking performance metrics for NoBackdrop models.
    """
    
    @staticmethod
    def compute_perplexity(loss: torch.Tensor) -> float:
        """
        Compute perplexity from loss.
        
        Args:
            loss: Loss tensor
            
        Returns:
            perplexity: Computed perplexity
        """
        try:
            # Handle NaN or infinite loss
            if torch.isnan(loss) or torch.isinf(loss):
                return 1000.0  # Return a large but finite value
            
            # Get loss as a Python float
            loss_value = loss.item()
            
            # Apply clipping to prevent overflow
            loss_value = max(1.0, min(loss_value, 15.0))
            
            # Calculate perplexity
            perplexity = np.exp(loss_value)
            
            # Cap perplexity at a reasonable maximum value
            if perplexity > 10000 or np.isnan(perplexity) or np.isinf(perplexity):
                return 10000.0
                
            return perplexity
        except Exception as e:
            # Fallback for any calculation errors
            return 1000.0
    
    @staticmethod
    def compute_token_accuracy(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> float:
        """
        Compute token prediction accuracy.
        
        Args:
            logits: Logits tensor of shape (batch_size, seq_len, vocab_size)
            labels: Labels tensor of shape (batch_size, seq_len)
            ignore_index: Index to ignore in accuracy calculation
            
        Returns:
            accuracy: Token prediction accuracy
        """
        # Get predictions
        preds = torch.argmax(logits, dim=-1)
        
        # Create mask for valid tokens
        mask = (labels != ignore_index)
        
        # Compute accuracy
        correct = (preds == labels) & mask
        accuracy = correct.sum().float() / mask.sum().float()
        
        return accuracy.item()
    
    @staticmethod
    def compute_memory_usage(model: torch.nn.Module, input_ids: torch.Tensor, 
                            attention_mask: Optional[torch.Tensor] = None,
                            update_model: bool = True) -> float:
        """
        Compute memory usage for a forward pass.
        
        Args:
            model: Model to evaluate
            input_ids: Input token IDs
            attention_mask: Attention mask
            update_model: Whether to update the model during forward pass
            
        Returns:
            memory_mb: Peak memory usage in MB
        """
        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        # Forward pass
        if isinstance(model, HebbianLM):
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                update_model=update_model,
                compute_loss=True,
            )
        else:
            # Assume it's a standard transformer model
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
        
        # Get peak memory usage
        memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        
        return memory_mb
    
    @staticmethod
    def compute_inference_speed(model: torch.nn.Module, input_ids: torch.Tensor, 
                              attention_mask: Optional[torch.Tensor] = None,
                              num_runs: int = 10) -> float:
        """
        Compute inference speed in tokens per second.
        
        Args:
            model: Model to evaluate
            input_ids: Input token IDs
            attention_mask: Attention mask
            num_runs: Number of runs to average over
            
        Returns:
            tokens_per_second: Inference speed in tokens per second
        """
        # Set model to evaluation mode
        model.eval()
        
        # Get total number of tokens
        batch_size, seq_len = input_ids.size()
        total_tokens = batch_size * seq_len * num_runs
        
        # Warm-up run
        with torch.no_grad():
            if isinstance(model, HebbianLM):
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    update_model=False,
                    compute_loss=False,
                )
            else:
                # Assume it's a standard transformer model
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
        
        # Measure inference time
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                if isinstance(model, HebbianLM):
                    model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        update_model=False,
                        compute_loss=False,
                    )
                else:
                    # Assume it's a standard transformer model
                    model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        # Compute tokens per second
        elapsed_time = end_time - start_time
        tokens_per_second = total_tokens / elapsed_time
        
        return tokens_per_second
    
    @staticmethod
    def compute_training_speed(model: torch.nn.Module, input_ids: torch.Tensor, 
                             attention_mask: Optional[torch.Tensor] = None,
                             optimizer: Optional[torch.optim.Optimizer] = None,
                             num_runs: int = 10) -> float:
        """
        Compute training speed in tokens per second.
        
        Args:
            model: Model to evaluate
            input_ids: Input token IDs
            attention_mask: Attention mask
            optimizer: Optimizer for standard models (not needed for HebbianLM)
            num_runs: Number of runs to average over
            
        Returns:
            tokens_per_second: Training speed in tokens per second
        """
        # Set model to training mode
        model.train()
        
        # Get total number of tokens
        batch_size, seq_len = input_ids.size()
        total_tokens = batch_size * seq_len * num_runs
        
        # Create optimizer if not provided
        if optimizer is None and not isinstance(model, HebbianLM):
            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        
        # Warm-up run
        if isinstance(model, HebbianLM):
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                update_model=True,
                compute_loss=True,
            )
        else:
            # Assume it's a standard transformer model
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # Measure training time
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(num_runs):
            if isinstance(model, HebbianLM):
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    update_model=True,
                    compute_loss=True,
                )
            else:
                # Assume it's a standard transformer model
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids,
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        # Compute tokens per second
        elapsed_time = end_time - start_time
        tokens_per_second = total_tokens / elapsed_time
        
        return tokens_per_second


class AdaptationEvaluator:
    """
    Evaluator for measuring model adaptation capabilities.
    """
    
    def __init__(
        self,
        model: HebbianLM,
        tokenizer: PreTrainedTokenizer,
        device: torch.device = None,
    ):
        """
        Initialize the evaluator.
        
        Args:
            model: HebbianLM model
            tokenizer: Tokenizer to use
            device: Device to use (cpu or cuda)
        """
        self.model = model
        self.tokenizer = tokenizer
        
        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Move model to device
        self.model.to(self.device)
    
    def evaluate_adaptation(
        self,
        adaptation_text: str,
        test_prompts: List[str],
        max_length: int = 30,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> Dict[str, Any]:
        """
        Evaluate adaptation to new information.
        
        Args:
            adaptation_text: Text to adapt to
            test_prompts: List of prompts to test adaptation
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_k: Number of highest probability tokens to keep
            top_p: Cumulative probability threshold for nucleus sampling
            
        Returns:
            results: Dictionary of adaptation results
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Record initial generations
        initial_generations = []
        
        for prompt in test_prompts:
            # Generate with model
            tokenized = self.tokenizer(prompt, return_tensors="pt")
            input_ids = tokenized["input_ids"].to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    update_model=False,
                )
            
            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            initial_generations.append(generated_text)
        
        # Adapt model
        tokenized = self.tokenizer(adaptation_text, return_tensors="pt")
        adaptation_ids = tokenized["input_ids"].to(self.device)
        attention_mask = torch.ones_like(adaptation_ids)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=adaptation_ids,
                attention_mask=attention_mask,
                update_model=True,
                compute_loss=False,
            )
        
        # Record adapted generations
        adapted_generations = []
        
        for prompt in test_prompts:
            # Generate with model
            tokenized = self.tokenizer(prompt, return_tensors="pt")
            input_ids = tokenized["input_ids"].to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    update_model=False,
                )
            
            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            adapted_generations.append(generated_text)
        
        # Collect results
        results = {
            "adaptation_text": adaptation_text,
            "test_prompts": test_prompts,
            "initial_generations": initial_generations,
            "adapted_generations": adapted_generations,
        }
        
        return results
    
    def evaluate_continuous_adaptation(
        self,
        adaptation_texts: List[str],
        test_prompts: List[str],
        max_length: int = 30,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> Dict[str, Any]:
        """
        Evaluate continuous adaptation to a sequence of texts.
        
        Args:
            adaptation_texts: List of texts to adapt to in sequence
            test_prompts: List of prompts to test adaptation
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_k: Number of highest probability tokens to keep
            top_p: Cumulative probability threshold for nucleus sampling
            
        Returns:
            results: Dictionary of adaptation results
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Record initial generations
        initial_generations = []
        
        for prompt in test_prompts:
            # Generate with model
            tokenized = self.tokenizer(prompt, return_tensors="pt")
            input_ids = tokenized["input_ids"].to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    update_model=False,
                )
            
            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            initial_generations.append(generated_text)
        
        # Track generations after each adaptation
        all_adapted_generations = []
        
        # Adapt model to each text in sequence
        for adaptation_text in adaptation_texts:
            tokenized = self.tokenizer(adaptation_text, return_tensors="pt")
            adaptation_ids = tokenized["input_ids"].to(self.device)
            attention_mask = torch.ones_like(adaptation_ids)
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=adaptation_ids,
                    attention_mask=attention_mask,
                    update_model=True,
                    compute_loss=False,
                )
            
            # Record generations after this adaptation
            adapted_generations = []
            
            for prompt in test_prompts:
                # Generate with model
                tokenized = self.tokenizer(prompt, return_tensors="pt")
                input_ids = tokenized["input_ids"].to(self.device)
                
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        input_ids=input_ids,
                        max_length=max_length,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        update_model=False,
                    )
                
                generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                adapted_generations.append(generated_text)
            
            all_adapted_generations.append(adapted_generations)
        
        # Collect results
        results = {
            "adaptation_texts": adaptation_texts,
            "test_prompts": test_prompts,
            "initial_generations": initial_generations,
            "all_adapted_generations": all_adapted_generations,
        }
        
        return results
    
    def plot_adaptation_results(
        self,
        results: Dict[str, Any],
        output_path: Optional[str] = None,
    ) -> None:
        """
        Plot adaptation results.
        
        Args:
            results: Dictionary of adaptation results
            output_path: Path to save the plot
        """
        # Extract data
        test_prompts = results["test_prompts"]
        initial_generations = results["initial_generations"]
        
        if "adapted_generations" in results:
            # Single adaptation
            adapted_generations = results["adapted_generations"]
            
            # Create figure
            fig, axes = plt.subplots(len(test_prompts), 1, figsize=(12, 4 * len(test_prompts)))
            if len(test_prompts) == 1:
                axes = [axes]
            
            for i, (prompt, initial, adapted) in enumerate(zip(test_prompts, initial_generations, adapted_generations)):
                # Truncate generations for display
                initial_display = initial[:100] + "..." if len(initial) > 100 else initial
                adapted_display = adapted[:100] + "..." if len(adapted) > 100 else adapted
                
                # Plot
                axes[i].text(0.01, 0.7, f"Initial: {initial_display}", wrap=True, fontsize=10)
                axes[i].text(0.01, 0.3, f"Adapted: {adapted_display}", wrap=True, fontsize=10)
                axes[i].set_title(f"Prompt: {prompt}")
                axes[i].axis("off")
            
            plt.tight_layout()
            
            # Save or show
            if output_path:
                plt.savefig(output_path)
            else:
                plt.show()
            
            plt.close()
        
        elif "all_adapted_generations" in results:
            # Continuous adaptation
            all_adapted_generations = results["all_adapted_generations"]
            adaptation_texts = results["adaptation_texts"]
            
            # Create figure for each prompt
            for i, prompt in enumerate(test_prompts):
                plt.figure(figsize=(12, 6))
                
                # Plot initial generation
                plt.text(0.01, 0.95, f"Initial: {initial_generations[i][:100]}...", wrap=True, fontsize=10)
                
                # Plot generations after each adaptation
                for j, (adapted_generations, adaptation_text) in enumerate(zip(all_adapted_generations, adaptation_texts)):
                    y_pos = 0.85 - (j + 1) * 0.15
                    plt.text(0.01, y_pos, f"After '{adaptation_text[:30]}...': {adapted_generations[i][:100]}...", 
                             wrap=True, fontsize=10)
                
                plt.title(f"Adaptation Progress for Prompt: {prompt}")
                plt.axis("off")
                
                # Save or show
                if output_path:
                    plt.savefig(f"{output_path.split('.')[0]}_{i}.png")
                else:
                    plt.show()
                
                plt.close()


def evaluate_model_capabilities(
    model: HebbianLM,
    tokenizer: PreTrainedTokenizer,
    device: torch.device = None,
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of model capabilities.
    
    Args:
        model: HebbianLM model
        tokenizer: Tokenizer to use
        device: Device to use (cpu or cuda)
        
    Returns:
        results: Dictionary of evaluation results
    """
    # Determine device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move model to device
    model.to(device)
    
    # Initialize results
    results = {}
    
    # Generate random input
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, model.vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids)
    
    # Evaluate memory usage
    memory_usage = PerformanceMetrics.compute_memory_usage(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        update_model=True,
    )
    results["memory_usage_mb"] = memory_usage
    
    # Evaluate inference speed
    inference_speed = PerformanceMetrics.compute_inference_speed(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        num_runs=5,
    )
    results["inference_speed_tokens_per_second"] = inference_speed
    
    # Evaluate training speed
    training_speed = PerformanceMetrics.compute_training_speed(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        num_runs=5,
    )
    results["training_speed_tokens_per_second"] = training_speed
    
    # Evaluate adaptation capabilities
    adaptation_evaluator = AdaptationEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=device,
    )
    
    # Simple adaptation test
    adaptation_results = adaptation_evaluator.evaluate_adaptation(
        adaptation_text="The capital of France is Paris. The capital of Italy is Rome.",
        test_prompts=["The capital of France is", "The capital of Italy is"],
    )
    results["adaptation_test"] = adaptation_results
    
    # Continuous adaptation test
    continuous_adaptation_results = adaptation_evaluator.evaluate_continuous_adaptation(
        adaptation_texts=[
            "The largest planet in our solar system is Jupiter.",
            "The closest planet to the Sun is Mercury.",
            "The planet with rings is Saturn.",
        ],
        test_prompts=[
            "The largest planet in our solar system is",
            "The closest planet to the Sun is",
            "The planet with rings is",
        ],
    )
    results["continuous_adaptation_test"] = continuous_adaptation_results
    
    return results
