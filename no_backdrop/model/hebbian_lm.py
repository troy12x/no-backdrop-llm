"""
Main model architecture for NoBackdrop - a forward-only learning language model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

from .token_embeddings import HebbianEmbedding
from .local_attention import LocalSelfAttention
from .feed_forward import FeedForward


class HebbianLayer(nn.Module):
    """
    A layer combining local attention and feed-forward networks with fast weights.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        window_size: int = 16,
        intermediate_size: int = None,
        dropout: float = 0.1,
        use_fast_weights: bool = True,
        update_rate: float = 0.01,
    ):
        super().__init__()
        
        # Local attention mechanism
        self.attention = LocalSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            window_size=window_size,
            dropout=dropout,
        )
        
        # Feed-forward network
        self.feed_forward = FeedForward(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dropout=dropout,
            use_fast_weights=use_fast_weights,
            update_rate=update_rate,
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        update_weights: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through the layer.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Mask tensor of shape [batch_size, seq_len]
            update_weights: Whether to update fast weights during this forward pass
            
        Returns:
            output: Processed tensor of shape [batch_size, seq_len, hidden_size]
        """
        # Apply attention
        attention_output = self.attention(hidden_states, attention_mask)
        
        # Apply feed-forward network
        output = self.feed_forward(attention_output, update_weights)
        
        return output


class HebbianLM(nn.Module):
    """
    Hebbian Language Model with forward-only learning capabilities.
    
    This model uses a combination of traditional parameters (slow weights) and
    rapidly adaptable parameters (fast weights) to enable learning without
    backpropagation through Hebbian-inspired updates.
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 768,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 12,
        window_size: int = 64,
        intermediate_size: int = None,
        dropout: float = 0.1,
        max_position_embeddings: int = 1024,
        update_rate: float = 0.01,
        use_fast_weights: bool = True,
        use_frequency_stats: bool = True,
        use_context_modulation: bool = True,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
    ):
        super().__init__()
        
        # Model configuration
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.window_size = window_size
        self.intermediate_size = intermediate_size or hidden_size * 4
        self.dropout = dropout
        self.max_position_embeddings = max_position_embeddings
        self.update_rate = update_rate
        self.use_fast_weights = use_fast_weights
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        
        # Token embeddings with fast weights
        self.embeddings = HebbianEmbedding(
            vocab_size=vocab_size,
            embedding_dim=hidden_size,
            max_position_embeddings=max_position_embeddings,
            dropout=dropout,
            update_rate=update_rate,
            use_fast_weights=use_fast_weights,
            use_frequency_stats=use_frequency_stats,
            use_context_modulation=use_context_modulation,
        )
        
        # Transformer layers with local attention and fast weights
        self.layers = nn.ModuleList([
            HebbianLayer(
                hidden_size=hidden_size,
                num_heads=num_attention_heads,
                window_size=window_size,
                intermediate_size=intermediate_size,
                dropout=dropout,
                use_fast_weights=use_fast_weights,
                update_rate=update_rate,
            )
            for _ in range(num_hidden_layers)
        ])
        
        # Final layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Output projection to vocabulary
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Tie weights between input embeddings and output projection
        self.lm_head.weight = self.embeddings.token_embeddings.weight
        
        # Initialize weights
        self._init_weights()
        
        # Token context memory
        self.token_contexts = {}
    
    def _init_weights(self):
        """Initialize weights with small values for stability."""
        # Embeddings and layers are initialized in their respective classes
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
    
    def get_input_embeddings(self):
        """Get the input embeddings layer."""
        return self.embeddings.token_embeddings
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        update_model: bool = False,
        compute_loss: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional fast weight updates.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Mask tensor of shape [batch_size, seq_len]
            position_ids: Position IDs [batch_size, seq_len]
            update_model: Whether to update fast weights during this forward pass
            compute_loss: Whether to compute and return loss
            
        Returns:
            outputs: Dictionary containing model outputs
        """
        batch_size, seq_len = input_ids.size()
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Get embeddings with fast weight updates if enabled
        embeddings, updated_contexts = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_contexts=self.token_contexts,
            update_embeddings=update_model,
        )
        
        # Update token contexts if in update mode
        if update_model:
            self.token_contexts = updated_contexts
        
        # Process through layers
        hidden_states = embeddings
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                update_weights=update_model,
            )
        
        # Apply final layer normalization
        hidden_states = self.layer_norm(hidden_states)
        
        # Compute logits
        logits = self.lm_head(hidden_states)
        
        # Prepare outputs
        outputs = {
            "hidden_states": hidden_states,
            "logits": logits,
        }
        
        # Compute loss if requested
        if compute_loss:
            # Shift logits and labels for next token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            
            # Compute cross-entropy loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
            loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))
            
            # Compute local contrastive loss if using fast weights
            if self.use_fast_weights:
                # Get predicted next token embeddings
                pred_token_embeds = F.softmax(shift_logits, dim=-1) @ self.lm_head.weight
                
                # Get actual next token embeddings
                actual_token_embeds = self.embeddings.token_embeddings(shift_labels)
                
                # Compute cosine similarity loss
                similarity = F.cosine_similarity(
                    pred_token_embeds.view(-1, self.hidden_size),
                    actual_token_embeds.view(-1, self.hidden_size),
                    dim=1
                )
                
                # Convert to loss (1 - similarity)
                local_loss = torch.mean(1.0 - similarity)
                
                # Add to outputs
                outputs["local_loss"] = local_loss
            
            # Add loss to outputs
            outputs["loss"] = loss
        
        return outputs
    
    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        update_model: bool = True,
    ) -> torch.LongTensor:
        """
        Generate text using the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask for input tokens
            max_length: Maximum length of generated sequence
            temperature: Temperature for sampling
            top_k: Number of highest probability tokens to keep
            top_p: Cumulative probability threshold for nucleus sampling
            repetition_penalty: Penalty for repeating tokens
            update_model: Whether to update the model during generation
            
        Returns:
            Generated token IDs
        """
        self.eval()
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Set attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Initialize generated sequence with input
        generated_ids = input_ids.clone()
        
        # Generate tokens up to max_length
        for _ in range(max_length - input_ids.shape[1]):
            # Get model outputs for the current sequence
            outputs = self(
                input_ids=generated_ids,
                attention_mask=torch.ones_like(generated_ids),
                update_model=update_model,
            )
            
            # Get logits for the next token (last position)
            next_token_logits = outputs["logits"][:, -1, :]
            
            # Apply temperature scaling
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            # Apply repetition penalty
            if repetition_penalty > 1.0:
                for i in range(batch_size):
                    for token_id in set(generated_ids[i].tolist()):
                        next_token_logits[i, token_id] /= repetition_penalty
            
            # Apply top-k filtering
            if top_k > 0:
                # Get top-k indices
                topk_values, topk_indices = torch.topk(next_token_logits, top_k, dim=-1)
                # Get minimum values for each batch
                min_values = topk_values[:, -1].unsqueeze(-1)
                # Create filter mask
                filter_mask = next_token_logits < min_values
                # Set filtered logits to -inf
                next_token_logits.masked_fill_(filter_mask, float('-inf'))
            
            # Apply top-p (nucleus) filtering
            if 0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Create filter mask for tokens above threshold
                filter_mask = cumulative_probs > top_p
                
                # Shift filter mask to exclude first token above threshold
                filter_mask = torch.cat([
                    torch.zeros_like(filter_mask[:, :1]),
                    filter_mask[:, :-1]
                ], dim=-1)
                
                # Get indices of tokens to keep
                indices_to_keep = torch.zeros_like(next_token_logits, dtype=torch.bool)
                for i in range(batch_size):
                    indices_to_keep[i, sorted_indices[i, ~filter_mask[i]]] = True
                
                # Filter logits
                next_token_logits = torch.where(
                    indices_to_keep, 
                    next_token_logits, 
                    torch.full_like(next_token_logits, float('-inf'))
                )
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append next token to generated sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            # Safe EOS token checking
            if self.eos_token_id is not None:
                # Check if any sequence contains EOS token
                eos_found = False
                for i in range(batch_size):
                    # Check if EOS token is in this sequence
                    if self.eos_token_id in generated_ids[i]:
                        eos_found = True
                        break
                        
                if eos_found:
                    break
        
        return generated_ids
