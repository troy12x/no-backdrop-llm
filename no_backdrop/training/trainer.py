"""
Trainer for NoBackdrop models with forward-only learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import logging
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

from ..model.hebbian_lm import HebbianLM


class Trainer:
    """
    Trainer for NoBackdrop models with forward-only learning.
    
    This trainer doesn't use backpropagation, instead relying on the model's
    ability to update its parameters during the forward pass.
    """
    
    def __init__(
        self,
        model: HebbianLM,
        learning_rate: float = 0.01,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        device: str = None,
        log_interval: int = 10,
        eval_interval: int = 100,
        save_interval: int = 1000,
        checkpoint_dir: str = "./checkpoints",
        use_wandb: bool = False,
        save_best_only: bool = False,
    ):
        """
        Initialize the trainer.
        
        Args:
            model: HebbianLM model
            learning_rate: Learning rate for optimizer (used only for slow weights)
            weight_decay: Weight decay for optimizer
            max_grad_norm: Maximum gradient norm for clipping
            device: Device to use (cpu or cuda)
            log_interval: Interval for logging training metrics
            eval_interval: Interval for evaluation
            save_interval: Interval for saving checkpoints
            checkpoint_dir: Directory to save checkpoints
            use_wandb: Whether to use Weights & Biases for logging
            save_best_only: Whether to save only the best model (based on eval loss)
        """
        self.model = model
        
        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Move model to device
        self.model.to(self.device)
        
        # Create optimizer for slow weights (traditional parameters)
        # Fast weights are updated during the forward pass
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # Training configuration
        self.max_grad_norm = max_grad_norm
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.checkpoint_dir = checkpoint_dir
        self.use_wandb = use_wandb
        self.save_best_only = save_best_only
        
        # Initialize wandb if requested
        if use_wandb:
            try:
                import wandb
                self.wandb = wandb
            except ImportError:
                print("Weights & Biases not installed. Running without wandb.")
                self.use_wandb = False
        
        # Training state
        self.global_step = 0
        self.best_eval_loss = float("inf")
    
    def _compute_loss(self, model_outputs: Dict[str, torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for the model.
        
        Args:
            model_outputs: Model outputs dictionary
            labels: Target labels
            
        Returns:
            loss: Computed loss
        """
        # Get loss from model outputs
        if "loss" in model_outputs:
            return model_outputs["loss"]
        
        # Compute loss manually if not provided
        logits = model_outputs["logits"]
        
        # Shift logits and labels for next token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        # Compute cross-entropy loss
        loss_fct = nn.CrossEntropyLoss(ignore_index=self.model.pad_token_id)
        loss = loss_fct(shift_logits.view(-1, self.model.vocab_size), shift_labels.view(-1))
        
        return loss
    
    def _compute_perplexity(self, loss: torch.Tensor) -> float:
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
            perplexity = math.exp(loss_value)
            
            # Cap perplexity at a reasonable maximum value
            if perplexity > 10000 or math.isnan(perplexity) or math.isinf(perplexity):
                return 10000.0
                
            return perplexity
        except Exception as e:
            # Fallback for any calculation errors
            return 1000.0
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            batch: Batch of data
            
        Returns:
            metrics: Dictionary of training metrics
        """
        # Set model to training mode
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass with parameter updates
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            update_model=True,  # Enable fast weight updates
            compute_loss=True,  # Compute loss during forward pass
        )
        
        # Get loss
        loss = self._compute_loss(outputs, batch["input_ids"])
        
        # Backward pass (only for slow weights)
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        
        # Update slow weights
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Compute perplexity
        perplexity = self._compute_perplexity(loss)
        
        # Collect metrics
        metrics = {
            "loss": loss.item(),
            "perplexity": perplexity,
        }
        
        # Add local loss if available
        if "local_loss" in outputs:
            metrics["local_loss"] = outputs["local_loss"].item()
        
        return metrics
    
    def eval_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform a single evaluation step.
        
        Args:
            batch: Batch of data
            
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass without parameter updates
        with torch.no_grad():
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                update_model=False,  # Disable fast weight updates
                compute_loss=True,  # Compute loss during forward pass
            )
        
        # Get loss
        loss = self._compute_loss(outputs, batch["input_ids"])
        
        # Compute perplexity
        perplexity = self._compute_perplexity(loss)
        
        # Collect metrics
        metrics = {
            "loss": loss.item(),
            "perplexity": perplexity,
        }
        
        # Add local loss if available
        if "local_loss" in outputs:
            metrics["local_loss"] = outputs["local_loss"].item()
        
        return metrics
    
    def train(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        eval_dataloader: Optional[torch.utils.data.DataLoader] = None,
        num_epochs: int = 1,
        max_steps: Optional[int] = None,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_dataloader: DataLoader for training
            eval_dataloader: DataLoader for evaluation
            num_epochs: Number of epochs to train
            max_steps: Maximum number of steps to train
            callback: Callback function called after each step
            
        Returns:
            history: Dictionary of training history
        """
        # Initialize history
        history = {
            "train_loss": [],
            "train_perplexity": [],
            "eval_loss": [],
            "eval_perplexity": [],
        }
        
        # Initialize wandb if requested
        if self.use_wandb and self.global_step == 0:
            self.wandb.init(
                project="no-backdrop",
                config={
                    "model_type": "HebbianLM",
                    "vocab_size": self.model.vocab_size,
                    "hidden_size": self.model.hidden_size,
                    "num_hidden_layers": self.model.num_hidden_layers,
                    "num_attention_heads": self.model.num_attention_heads,
                    "window_size": self.model.window_size,
                    "use_fast_weights": self.model.use_fast_weights,
                    "update_rate": self.model.update_rate,
                },
            )
        
        # Training loop
        steps_per_epoch = len(train_dataloader)
        total_steps = num_epochs * steps_per_epoch
        if max_steps is not None:
            total_steps = min(total_steps, max_steps)
        
        print(f"Starting training for {total_steps} steps")
        
        # Create progress bar
        progress_bar = tqdm(total=total_steps, desc="Training")
        
        # Training loop
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Training
            for batch_idx, batch in enumerate(train_dataloader):
                # Check if we've reached the maximum number of steps
                if max_steps is not None and self.global_step >= max_steps:
                    break
                
                # Perform training step
                metrics = self.train_step(batch)
                
                # Update history
                history["train_loss"].append(metrics["loss"])
                history["train_perplexity"].append(metrics["perplexity"])
                
                # Log metrics
                if self.global_step % self.log_interval == 0:
                    log_metrics = {
                        "train/loss": metrics["loss"],
                        "train/perplexity": metrics["perplexity"],
                        "train/step": self.global_step,
                        "train/epoch": epoch + batch_idx / steps_per_epoch,
                    }
                    
                    # Add local loss if available
                    if "local_loss" in metrics:
                        log_metrics["train/local_loss"] = metrics["local_loss"]
                    
                    # Log to wandb if requested
                    if self.use_wandb:
                        self.wandb.log(log_metrics)
                    
                    # Print metrics
                    print(f"Step {self.global_step}: {log_metrics}")
                
                # Evaluation
                if eval_dataloader is not None and self.global_step % self.eval_interval == 0:
                    eval_metrics = self.evaluate(eval_dataloader)
                    
                    # Update history
                    history["eval_loss"].append(eval_metrics["loss"])
                    history["eval_perplexity"].append(eval_metrics["perplexity"])
                    
                    # Log metrics
                    log_metrics = {
                        "eval/loss": eval_metrics["loss"],
                        "eval/perplexity": eval_metrics["perplexity"],
                        "eval/step": self.global_step,
                    }
                    
                    # Add local loss if available
                    if "local_loss" in eval_metrics:
                        log_metrics["eval/local_loss"] = eval_metrics["local_loss"]
                    
                    # Log to wandb if requested
                    if self.use_wandb:
                        self.wandb.log(log_metrics)
                    
                    # Print metrics
                    print(f"Evaluation at step {self.global_step}: {log_metrics}")
                    
                    # Save best model if this is the best evaluation loss
                    if eval_metrics["loss"] < self.best_eval_loss:
                        self.best_eval_loss = eval_metrics["loss"]
                        try:
                            self.save_checkpoint(f"{self.checkpoint_dir}/best_model.pt")
                            print(f"Checkpoint saved to {self.checkpoint_dir}/best_model.pt")
                        except Exception as e:
                            print(f"Warning: Failed to save best model: {e}")
                
                # Save checkpoint at regular intervals (if not save_best_only)
                if self.global_step % self.save_interval == 0 and not self.save_best_only:
                    try:
                        self.save_checkpoint(f"{self.checkpoint_dir}/step_{self.global_step}.pt")
                    except Exception as e:
                        print(f"Warning: Failed to save checkpoint at step {self.global_step}: {e}")
                
                # Call callback if provided
                if callback is not None:
                    callback({
                        "step": self.global_step,
                        "epoch": epoch,
                        "batch_idx": batch_idx,
                        "metrics": metrics,
                    })
                
                # Update progress bar
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "loss": f"{metrics['loss']:.4f}",
                    "ppl": f"{metrics['perplexity']:.2f}",
                })
                
                # Increment global step
                self.global_step += 1
            
            # End of epoch
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
        
        # Close progress bar
        progress_bar.close()
        
        # Final evaluation
        if eval_dataloader is not None:
            eval_metrics = self.evaluate(eval_dataloader)
            print(f"Final evaluation: {eval_metrics}")
        
        # Save final model
        self.save_checkpoint(f"{self.checkpoint_dir}/final_model.pt")
        
        # Close wandb if requested
        if self.use_wandb:
            self.wandb.finish()
        
        return history
    
    def evaluate(self, eval_dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            eval_dataloader: DataLoader for evaluation
            
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize metrics
        total_loss = 0.0
        total_samples = 0
        
        # Evaluation loop
        for batch in eval_dataloader:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass without parameter updates
            with torch.no_grad():
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    update_model=False,  # Disable fast weight updates
                    compute_loss=True,  # Compute loss during forward pass
                )
            
            # Get loss
            loss = self._compute_loss(outputs, batch["input_ids"])
            
            # Update metrics
            batch_size = batch["input_ids"].size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
        
        # Compute average metrics
        if total_samples > 0:
            avg_loss = total_loss / total_samples
            perplexity = self._compute_perplexity(torch.tensor(avg_loss))
        else:
            # Handle case of empty evaluation dataset
            avg_loss = 0.0
            perplexity = 0.0
            print("Warning: No samples in evaluation dataset")
        
        # Collect metrics
        metrics = {
            "loss": avg_loss,
            "perplexity": perplexity,
        }
        
        return metrics
    
    def generate_text(
        self,
        prompt: str,
        tokenizer: Any,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        update_model: bool = True,
        num_return_sequences: int = 1,
    ) -> List[str]:
        """
        Generate text using the model.
        
        Args:
            prompt: Text prompt
            tokenizer: Tokenizer to use
            max_length: Maximum length of generated sequence
            temperature: Sampling temperature
            top_k: Number of highest probability tokens to keep
            top_p: Cumulative probability threshold for nucleus sampling
            repetition_penalty: Penalty for repeating tokens
            update_model: Whether to update fast weights during generation
            num_return_sequences: Number of sequences to generate
            
        Returns:
            generated_texts: List of generated text strings
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Tokenize prompt
        tokenized = tokenizer(prompt, return_tensors="pt")
        input_ids = tokenized["input_ids"].to(self.device)
        
        # Repeat input for multiple sequences
        if num_return_sequences > 1:
            input_ids = input_ids.repeat(num_return_sequences, 1)
        
        # Generate text
        generated_ids = self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            update_model=update_model,
        )
        
        # Decode generated text
        generated_texts = []
        for ids in generated_ids:
            text = tokenizer.decode(ids, skip_special_tokens=True)
            generated_texts.append(text)
        
        return generated_texts
    
    def save_checkpoint(self, path: str) -> None:
        """
        Save a checkpoint of the model and trainer state.
        
        Args:
            path: Path to save checkpoint
        """
        # Create checkpoint
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "best_eval_loss": self.best_eval_loss,
            "token_contexts": self.model.token_contexts,
        }
        
        # Save checkpoint
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """
        Load a checkpoint of the model and trainer state.
        
        Args:
            path: Path to load checkpoint from
        """
        # Load checkpoint
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Load trainer state
        self.global_step = checkpoint["global_step"]
        self.best_eval_loss = checkpoint["best_eval_loss"]
        
        # Load token contexts
        if "token_contexts" in checkpoint:
            self.model.token_contexts = checkpoint["token_contexts"]
        
        print(f"Checkpoint loaded from {path}")
