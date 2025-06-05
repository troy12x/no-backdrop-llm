"""
Visualization utilities for NoBackdrop models.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Dict, List, Optional, Tuple, Union, Any
import time
from datetime import datetime


class TrainingVisualizer:
    """
    Visualizer for training metrics and model performance.
    """
    
    def __init__(
        self,
        output_dir: str = "./visualizations",
        experiment_name: Optional[str] = None,
    ):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualizations
            experiment_name: Name of the experiment
        """
        self.output_dir = output_dir
        
        # Create experiment name if not provided
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"experiment_{timestamp}"
        
        self.experiment_name = experiment_name
        self.experiment_dir = os.path.join(output_dir, experiment_name)
        
        # Create directories
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Initialize metrics
        self.metrics = {
            "train_loss": [],
            "train_perplexity": [],
            "eval_loss": [],
            "eval_perplexity": [],
            "local_loss": [],
            "steps": [],
            "eval_steps": [],
        }
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        is_eval: bool = False,
    ) -> None:
        """
        Log training or evaluation metrics.
        
        Args:
            metrics: Dictionary of metrics
            step: Current step
            is_eval: Whether these are evaluation metrics
        """
        # Log step
        if not is_eval and step not in self.metrics["steps"]:
            self.metrics["steps"].append(step)
        
        if is_eval and step not in self.metrics["eval_steps"]:
            self.metrics["eval_steps"].append(step)
        
        # Log metrics
        for key, value in metrics.items():
            if key == "loss":
                if is_eval:
                    self.metrics["eval_loss"].append(value)
                else:
                    self.metrics["train_loss"].append(value)
            elif key == "perplexity":
                if is_eval:
                    self.metrics["eval_perplexity"].append(value)
                else:
                    self.metrics["train_perplexity"].append(value)
            elif key == "local_loss":
                self.metrics["local_loss"].append(value)
            else:
                # Add other metrics as needed
                if key not in self.metrics:
                    self.metrics[key] = []
                self.metrics[key].append(value)
    
    def plot_loss_curve(
        self,
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> None:
        """
        Plot training and evaluation loss curves.
        
        Args:
            save_path: Path to save the plot
            show: Whether to show the plot
        """
        plt.figure(figsize=(10, 6))
        
        # Plot training loss
        if len(self.metrics["train_loss"]) > 0:
            plt.plot(self.metrics["steps"], self.metrics["train_loss"], label="Train Loss")
        
        # Plot evaluation loss
        if len(self.metrics["eval_loss"]) > 0:
            plt.plot(self.metrics["eval_steps"], self.metrics["eval_loss"], label="Eval Loss", marker="o")
        
        # Plot local loss if available
        if len(self.metrics["local_loss"]) > 0:
            plt.plot(self.metrics["steps"][:len(self.metrics["local_loss"])], 
                    self.metrics["local_loss"], label="Local Loss", linestyle="--")
        
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Training and Evaluation Loss")
        plt.legend()
        plt.grid(True)
        
        # Save or show
        if save_path is None:
            save_path = os.path.join(self.experiment_dir, "loss_curve.png")
        
        plt.savefig(save_path)
        
        if show:
            plt.show()
        
        plt.close()
    
    def plot_perplexity_curve(
        self,
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> None:
        """
        Plot training and evaluation perplexity curves.
        
        Args:
            save_path: Path to save the plot
            show: Whether to show the plot
        """
        plt.figure(figsize=(10, 6))
        
        # Plot training perplexity
        if len(self.metrics["train_perplexity"]) > 0:
            plt.plot(self.metrics["steps"], self.metrics["train_perplexity"], label="Train Perplexity")
        
        # Plot evaluation perplexity
        if len(self.metrics["eval_perplexity"]) > 0:
            plt.plot(self.metrics["eval_steps"], self.metrics["eval_perplexity"], 
                    label="Eval Perplexity", marker="o")
        
        plt.xlabel("Steps")
        plt.ylabel("Perplexity")
        plt.title("Training and Evaluation Perplexity")
        plt.legend()
        plt.grid(True)
        
        # Save or show
        if save_path is None:
            save_path = os.path.join(self.experiment_dir, "perplexity_curve.png")
        
        plt.savefig(save_path)
        
        if show:
            plt.show()
        
        plt.close()
    
    def plot_all_metrics(
        self,
        show: bool = False,
    ) -> None:
        """
        Plot all available metrics.
        
        Args:
            show: Whether to show the plots
        """
        # Plot loss curve
        self.plot_loss_curve(show=show)
        
        # Plot perplexity curve
        self.plot_perplexity_curve(show=show)
        
        # Plot other metrics as needed
        for key in self.metrics:
            if key not in ["train_loss", "train_perplexity", "eval_loss", "eval_perplexity", 
                          "local_loss", "steps", "eval_steps"] and len(self.metrics[key]) > 0:
                plt.figure(figsize=(10, 6))
                plt.plot(self.metrics["steps"][:len(self.metrics[key])], self.metrics[key], label=key)
                plt.xlabel("Steps")
                plt.ylabel(key)
                plt.title(f"{key} over Training")
                plt.legend()
                plt.grid(True)
                
                # Save
                save_path = os.path.join(self.experiment_dir, f"{key}_curve.png")
                plt.savefig(save_path)
                
                if show:
                    plt.show()
                
                plt.close()
    
    def save_metrics(self) -> None:
        """
        Save metrics to a file.
        """
        import json
        
        # Convert metrics to JSON-serializable format
        serializable_metrics = {}
        for key, value in self.metrics.items():
            serializable_metrics[key] = [float(v) if isinstance(v, (np.float32, np.float64)) else v for v in value]
        
        # Save metrics
        save_path = os.path.join(self.experiment_dir, "metrics.json")
        with open(save_path, "w") as f:
            json.dump(serializable_metrics, f, indent=2)
    
    def load_metrics(self, path: str) -> None:
        """
        Load metrics from a file.
        
        Args:
            path: Path to the metrics file
        """
        import json
        
        # Load metrics
        with open(path, "r") as f:
            self.metrics = json.load(f)


class TokenStatisticsVisualizer:
    """
    Visualizer for token statistics and embeddings.
    """
    
    @staticmethod
    def plot_token_frequency(
        token_counts: Dict[int, int],
        tokenizer: Any,
        top_k: int = 50,
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> None:
        """
        Plot token frequency distribution.
        
        Args:
            token_counts: Dictionary mapping token IDs to counts
            tokenizer: Tokenizer to decode token IDs
            top_k: Number of top tokens to show
            save_path: Path to save the plot
            show: Whether to show the plot
        """
        # Sort tokens by frequency
        sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
        top_tokens = sorted_tokens[:top_k]
        
        # Decode token IDs
        token_texts = []
        for token_id, _ in top_tokens:
            try:
                token_text = tokenizer.decode([token_id])
                # Clean up token text for display
                token_text = token_text.replace(" ", "·").replace("\n", "\\n")
                if len(token_text) > 10:
                    token_text = token_text[:10] + "..."
                token_texts.append(token_text)
            except:
                token_texts.append(f"ID:{token_id}")
        
        # Get counts
        counts = [count for _, count in top_tokens]
        
        # Plot
        plt.figure(figsize=(12, 8))
        plt.bar(range(len(counts)), counts)
        plt.xticks(range(len(counts)), token_texts, rotation=90)
        plt.xlabel("Tokens")
        plt.ylabel("Frequency")
        plt.title(f"Top {top_k} Token Frequencies")
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path)
        
        if show:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_token_update_rates(
        token_update_rates: Dict[int, float],
        tokenizer: Any,
        top_k: int = 50,
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> None:
        """
        Plot token update rates.
        
        Args:
            token_update_rates: Dictionary mapping token IDs to update rates
            tokenizer: Tokenizer to decode token IDs
            top_k: Number of top tokens to show
            save_path: Path to save the plot
            show: Whether to show the plot
        """
        # Sort tokens by update rate
        sorted_tokens = sorted(token_update_rates.items(), key=lambda x: x[1], reverse=True)
        top_tokens = sorted_tokens[:top_k]
        
        # Decode token IDs
        token_texts = []
        for token_id, _ in top_tokens:
            try:
                token_text = tokenizer.decode([token_id])
                # Clean up token text for display
                token_text = token_text.replace(" ", "·").replace("\n", "\\n")
                if len(token_text) > 10:
                    token_text = token_text[:10] + "..."
                token_texts.append(token_text)
            except:
                token_texts.append(f"ID:{token_id}")
        
        # Get update rates
        rates = [rate for _, rate in top_tokens]
        
        # Plot
        plt.figure(figsize=(12, 8))
        plt.bar(range(len(rates)), rates)
        plt.xticks(range(len(rates)), token_texts, rotation=90)
        plt.xlabel("Tokens")
        plt.ylabel("Update Rate")
        plt.title(f"Top {top_k} Token Update Rates")
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path)
        
        if show:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_embedding_pca(
        embeddings: torch.Tensor,
        token_ids: List[int],
        tokenizer: Any,
        top_k: int = 100,
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> None:
        """
        Plot PCA of token embeddings.
        
        Args:
            embeddings: Token embeddings tensor
            token_ids: List of token IDs
            tokenizer: Tokenizer to decode token IDs
            top_k: Number of top tokens to show
            save_path: Path to save the plot
            show: Whether to show the plot
        """
        from sklearn.decomposition import PCA
        
        # Convert embeddings to numpy
        embeddings_np = embeddings.detach().cpu().numpy()
        
        # Apply PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings_np)
        
        # Select top_k tokens
        if len(token_ids) > top_k:
            token_ids = token_ids[:top_k]
            embeddings_2d = embeddings_2d[:top_k]
        
        # Decode token IDs
        token_texts = []
        for token_id in token_ids:
            try:
                token_text = tokenizer.decode([token_id])
                # Clean up token text for display
                token_text = token_text.replace(" ", "·").replace("\n", "\\n")
                if len(token_text) > 10:
                    token_text = token_text[:10] + "..."
                token_texts.append(token_text)
            except:
                token_texts.append(f"ID:{token_id}")
        
        # Plot
        plt.figure(figsize=(12, 10))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5)
        
        # Add token labels
        for i, txt in enumerate(token_texts):
            plt.annotate(txt, (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=8)
        
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
        plt.title("PCA of Token Embeddings")
        plt.grid(True)
        
        # Save or show
        if save_path:
            plt.savefig(save_path)
        
        if show:
            plt.show()
        
        plt.close()


class AttentionVisualizer:
    """
    Visualizer for attention patterns.
    """
    
    @staticmethod
    def plot_attention_heatmap(
        attention_weights: torch.Tensor,
        tokens: List[str],
        layer_idx: int = 0,
        head_idx: int = 0,
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> None:
        """
        Plot attention weights as a heatmap.
        
        Args:
            attention_weights: Attention weights tensor of shape [batch_size, num_heads, seq_len, seq_len]
            tokens: List of token strings
            layer_idx: Layer index
            head_idx: Attention head index
            save_path: Path to save the plot
            show: Whether to show the plot
        """
        # Extract attention weights for the specified layer and head
        if attention_weights.dim() == 4:
            # [batch_size, num_heads, seq_len, seq_len]
            weights = attention_weights[0, head_idx].detach().cpu().numpy()
        else:
            # [num_heads, seq_len, seq_len]
            weights = attention_weights[head_idx].detach().cpu().numpy()
        
        # Truncate tokens if necessary
        seq_len = weights.shape[0]
        if len(tokens) > seq_len:
            tokens = tokens[:seq_len]
        elif len(tokens) < seq_len:
            tokens = tokens + [""] * (seq_len - len(tokens))
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.imshow(weights, cmap="viridis")
        plt.colorbar()
        
        # Add labels
        plt.xticks(range(len(tokens)), tokens, rotation=90, fontsize=8)
        plt.yticks(range(len(tokens)), tokens, fontsize=8)
        
        plt.xlabel("Target Tokens")
        plt.ylabel("Source Tokens")
        plt.title(f"Attention Weights (Layer {layer_idx}, Head {head_idx})")
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path)
        
        if show:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_local_attention_pattern(
        attention_weights: torch.Tensor,
        tokens: List[str],
        window_size: int,
        layer_idx: int = 0,
        head_idx: int = 0,
        token_idx: int = None,
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> None:
        """
        Plot local attention pattern for a specific token.
        
        Args:
            attention_weights: Attention weights tensor of shape [batch_size, num_heads, seq_len, seq_len]
            tokens: List of token strings
            window_size: Size of the attention window
            layer_idx: Layer index
            head_idx: Attention head index
            token_idx: Index of the token to visualize (if None, use the middle token)
            save_path: Path to save the plot
            show: Whether to show the plot
        """
        # Extract attention weights for the specified layer and head
        if attention_weights.dim() == 4:
            # [batch_size, num_heads, seq_len, seq_len]
            weights = attention_weights[0, head_idx].detach().cpu().numpy()
        else:
            # [num_heads, seq_len, seq_len]
            weights = attention_weights[head_idx].detach().cpu().numpy()
        
        # Determine token index
        seq_len = weights.shape[0]
        if token_idx is None:
            token_idx = seq_len // 2
        
        # Determine window boundaries
        window_start = max(0, token_idx - window_size // 2)
        window_end = min(seq_len, token_idx + window_size // 2 + 1)
        
        # Extract local attention weights
        local_weights = weights[token_idx, window_start:window_end]
        local_tokens = tokens[window_start:window_end]
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(local_weights)), local_weights)
        plt.xticks(range(len(local_weights)), local_tokens, rotation=90)
        
        plt.xlabel("Tokens")
        plt.ylabel("Attention Weight")
        plt.title(f"Local Attention Pattern (Layer {layer_idx}, Head {head_idx}, Token: {tokens[token_idx]})")
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path)
        
        if show:
            plt.show()
        
        plt.close()


def create_training_animation(
    metrics_dir: str,
    output_path: str,
    metric_name: str = "train_loss",
    fps: int = 10,
) -> None:
    """
    Create an animation of training progress.
    
    Args:
        metrics_dir: Directory containing metric files
        output_path: Path to save the animation
        metric_name: Name of the metric to animate
        fps: Frames per second
    """
    import glob
    import json
    import matplotlib.animation as animation
    
    # Find all metric files
    metric_files = sorted(glob.glob(os.path.join(metrics_dir, "*.json")))
    
    if len(metric_files) == 0:
        print(f"No metric files found in {metrics_dir}")
        return
    
    # Load metrics
    all_metrics = []
    for file_path in metric_files:
        with open(file_path, "r") as f:
            metrics = json.load(f)
            all_metrics.append(metrics)
    
    # Create animation
    fig, ax = plt.subplots(figsize=(10, 6))
    
    def update(frame):
        ax.clear()
        
        # Get metrics up to the current frame
        current_metrics = all_metrics[frame]
        
        # Plot metric
        if metric_name in current_metrics:
            steps = current_metrics.get("steps", list(range(len(current_metrics[metric_name]))))
            ax.plot(steps, current_metrics[metric_name])
            
            ax.set_xlabel("Steps")
            ax.set_ylabel(metric_name)
            ax.set_title(f"{metric_name} over Training (Step {steps[-1] if steps else 0})")
            ax.grid(True)
    
    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=len(all_metrics), interval=1000/fps)
    
    # Save animation
    ani.save(output_path, writer="pillow", fps=fps)
    plt.close()
