"""
Feed-forward network with memory-efficient implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """
    Feed-forward network with GELU activation and optional fast-weight adaptation.
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_fast_weights: bool = False,
        update_rate: float = 0.01,
    ):
        super().__init__()
        
        # Set intermediate size to 4x hidden size if not specified
        if intermediate_size is None:
            intermediate_size = hidden_size * 4
        
        # Main feed-forward layers
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        
        # Fast weights for dynamic adaptation
        self.use_fast_weights = use_fast_weights
        self.update_rate = update_rate
        
        if use_fast_weights:
            # Initialize fast weights as zeros
            self.register_buffer("fast_fc1_weight", torch.zeros_like(self.fc1.weight))
            self.register_buffer("fast_fc2_weight", torch.zeros_like(self.fc2.weight))
        
        # Activation function
        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for stability."""
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        update_weights: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass with optional fast weight updates.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            update_weights: Whether to update fast weights during this forward pass
            
        Returns:
            output: Processed tensor of shape [batch_size, seq_len, hidden_size]
        """
        # Apply layer normalization first (pre-norm formulation)
        normalized_states = self.layer_norm(hidden_states)
        
        # First linear layer with fast weights if enabled
        if self.use_fast_weights:
            # Combine slow and fast weights
            combined_weight = self.fc1.weight + self.fast_fc1_weight
            hidden = F.linear(normalized_states, combined_weight, self.fc1.bias)
        else:
            hidden = self.fc1(normalized_states)
        
        # Apply activation
        hidden = self.activation(hidden)
        
        # Second linear layer with fast weights if enabled
        if self.use_fast_weights:
            # Combine slow and fast weights
            combined_weight = self.fc2.weight + self.fast_fc2_weight
            output = F.linear(hidden, combined_weight, self.fc2.bias)
        else:
            output = self.fc2(hidden)
        
        # Apply dropout
        output = self.dropout(output)
        
        # Update fast weights if enabled and in update mode
        if self.use_fast_weights and update_weights:
            # Simple Hebbian update for fast weights
            # Strengthen connections that are active together
            batch_size, seq_len, _ = hidden_states.size()
            
            # Sample a subset of positions for efficiency
            sample_size = min(seq_len, 16)
            if seq_len > sample_size:
                indices = torch.randperm(seq_len)[:sample_size]
                sample_input = normalized_states[:, indices]
                sample_hidden = hidden[:, indices]
            else:
                sample_input = normalized_states
                sample_hidden = hidden
            
            # Flatten batch and sequence dimensions
            flat_input = sample_input.reshape(-1, sample_input.size(-1))
            flat_hidden = sample_hidden.reshape(-1, sample_hidden.size(-1))
            
            # Compute outer products for Hebbian updates
            # For FC1: input -> hidden
            update_fc1 = torch.einsum('bi,bj->ij', flat_hidden, flat_input)
            update_fc1 = update_fc1 / (flat_input.size(0) + 1e-6)
            
            # For FC2: hidden -> output
            # We use the pre-activation values from FC2 for this update
            if self.use_fast_weights:
                combined_weight = self.fc2.weight + self.fast_fc2_weight
                pre_output = F.linear(sample_hidden, combined_weight, bias=None)
            else:
                pre_output = F.linear(sample_hidden, self.fc2.weight, bias=None)
            
            flat_pre_output = pre_output.reshape(-1, pre_output.size(-1))
            update_fc2 = torch.einsum('bi,bj->ij', flat_pre_output, flat_hidden)
            update_fc2 = update_fc2 / (flat_hidden.size(0) + 1e-6)
            
            # Apply updates with stability checks
            if not torch.isnan(update_fc1).any() and not torch.isinf(update_fc1).any():
                # Clip updates to prevent extreme values
                update_fc1 = torch.clamp(update_fc1, min=-0.1, max=0.1)
                self.fast_fc1_weight += self.update_rate * update_fc1
            
            if not torch.isnan(update_fc2).any() and not torch.isinf(update_fc2).any():
                # Clip updates to prevent extreme values
                update_fc2 = torch.clamp(update_fc2, min=-0.1, max=0.1)
                self.fast_fc2_weight += self.update_rate * update_fc2
        
        # Apply residual connection
        output = hidden_states + output
        
        return output
