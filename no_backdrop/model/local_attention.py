"""
Local attention mechanism for efficient processing with limited context.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class LocalAttention(nn.Module):
    """
    Attention mechanism that only operates on a local window of tokens.
    
    This reduces computational complexity from O(n²) to O(n*w) where w is the window size.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        window_size: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = hidden_size // num_heads
        
        # Check that hidden size is divisible by number of heads
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for stability."""
        nn.init.normal_(self.q_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.k_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.v_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02)
        
        nn.init.zeros_(self.q_proj.bias)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.zeros_(self.out_proj.bias)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for local attention.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Mask tensor of shape [batch_size, seq_len]
            
        Returns:
            output: Attended tensor of shape [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.size()
        
        # Project inputs to queries, keys, and values
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores with local windowing
        # This is the key optimization: each token only attends to nearby tokens
        attention_scores = torch.zeros(
            batch_size, self.num_heads, seq_len, seq_len, device=hidden_states.device
        )
        
        # Fill in attention scores only for tokens within the window
        for i in range(seq_len):
            # Define local window
            window_start = max(0, i - self.window_size // 2)
            window_end = min(seq_len, i + self.window_size // 2 + 1)
            
            # Compute attention scores only for tokens in the window
            local_scores = torch.matmul(
                q[:, :, i:i+1], 
                k[:, :, window_start:window_end].transpose(-1, -2)
            ) / math.sqrt(self.head_dim)
            
            # Place scores in the right position in the full attention matrix
            attention_scores[:, :, i:i+1, window_start:window_end] = local_scores
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Convert mask (batch_size, seq_len) to attention mask shape (batch_size, 1, 1, seq_len)
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            
            # Set masked positions to a large negative value
            attention_scores = attention_scores.masked_fill(mask == 0, -10000.0)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        context = torch.matmul(attention_weights, v)
        
        # Reshape back to original dimensions
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        # Final projection
        output = self.out_proj(context)
        
        return output


class LocalSelfAttention(nn.Module):
    """
    Self-attention layer with local attention and layer normalization.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        window_size: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.attention = LocalAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            window_size=window_size,
            dropout=dropout,
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for local self-attention with residual connection and layer norm.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Mask tensor of shape [batch_size, seq_len]
            
        Returns:
            output: Attended tensor of shape [batch_size, seq_len, hidden_size]
        """
        # Apply layer normalization first (pre-norm formulation)
        normalized_states = self.layer_norm(hidden_states)
        
        # Apply attention
        attention_output = self.attention(normalized_states, attention_mask)
        
        # Apply dropout and residual connection
        output = hidden_states + self.dropout(attention_output)
        
        return output
