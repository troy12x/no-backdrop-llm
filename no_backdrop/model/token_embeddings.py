"""
Hebbian-inspired token embeddings with fast weights for forward-only learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class HebbianEmbedding(nn.Module):
    """
    Token embeddings with fast weights that update during the forward pass.
    
    This embedding layer maintains both slow weights (traditional embeddings) and
    fast weights (rapidly adaptable during inference) to enable learning without
    backpropagation through Hebbian-inspired updates.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        max_position_embeddings: int = 1024,
        dropout: float = 0.1,
        update_rate: float = 0.01,
        use_fast_weights: bool = True,
        use_frequency_stats: bool = True,
        use_context_modulation: bool = True,
    ):
        super().__init__()
        
        # Traditional embedding tables (slow weights)
        self.token_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.position_embeddings = nn.Embedding(max_position_embeddings, embedding_dim)
        
        # Fast weights for dynamic adaptation
        self.use_fast_weights = use_fast_weights
        if use_fast_weights:
            # Initialize fast weights as zeros (will be populated during forward pass)
            self.register_buffer("fast_token_weights", torch.zeros(vocab_size, embedding_dim))
            
        # Token statistics tracking
        self.use_frequency_stats = use_frequency_stats
        if use_frequency_stats:
            self.register_buffer("token_frequencies", torch.zeros(vocab_size))
            self.register_buffer("token_recency", torch.zeros(vocab_size))
        
        # Context modulation
        self.use_context_modulation = use_context_modulation
        if use_context_modulation:
            self.context_modulation = nn.Linear(embedding_dim, embedding_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.embedding_dim = embedding_dim
        self.update_rate = update_rate
        self.vocab_size = vocab_size
        
        # Initialize embeddings with small values
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for stability."""
        nn.init.normal_(self.token_embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.02)
        if self.use_context_modulation:
            nn.init.normal_(self.context_modulation.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.context_modulation.bias)
    
    def forward(
        self, 
        input_ids: torch.LongTensor,
        position_ids: Optional[torch.LongTensor] = None,
        token_contexts: Optional[Dict[int, torch.Tensor]] = None,
        update_embeddings: bool = True,
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """
        Forward pass with dynamic embedding updates.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            position_ids: Position IDs [batch_size, seq_len]
            token_contexts: Dictionary mapping token IDs to context vectors
            update_embeddings: Whether to update embeddings during this forward pass
            
        Returns:
            embeddings: Combined token and position embeddings
            updated_contexts: Updated token contexts dictionary
        """
        batch_size, seq_len = input_ids.size()
        
        # Generate position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Get base embeddings from slow weights
        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        
        # Combine with fast weights if enabled
        if self.use_fast_weights:
            # Gather fast weights for the current tokens
            fast_embeds = F.embedding(input_ids, self.fast_token_weights)
            
            # Combine slow and fast embeddings
            # Fast weights are additive to allow for incremental learning
            token_embeds = token_embeds + fast_embeds
        
        # Apply position embeddings
        embeddings = token_embeds + position_embeds
        
        # Apply context modulation if enabled
        if self.use_context_modulation:
            # Modulate embeddings based on their context
            context_mod = self.context_modulation(embeddings)
            embeddings = embeddings + context_mod
        
        # Update token statistics if enabled
        if self.use_frequency_stats and update_embeddings:
            # Get unique tokens in the batch
            unique_tokens = torch.unique(input_ids)
            
            # Update frequency counts
            self.token_frequencies.index_add_(0, unique_tokens, 
                                             torch.ones_like(unique_tokens, dtype=torch.float))
            
            # Update recency (decay old values and set current tokens to 1.0)
            self.token_recency *= 0.99  # Decay factor
            self.token_recency.index_fill_(0, unique_tokens, 1.0)
        
        # Update fast weights if enabled and in update mode
        if self.use_fast_weights and update_embeddings:
            # Process tokens in batches for efficiency
            # Get unique tokens in the batch to avoid redundant updates
            unique_tokens = torch.unique(input_ids)
            
            # Skip special tokens (assuming they're at the beginning of the vocab)
            unique_tokens = unique_tokens[unique_tokens >= 5]
            
            # Create a context embedding matrix for all tokens in the batch
            token_to_context = {}
            
            # First pass: collect context for each token
            for i in range(batch_size):
                # Use a sliding window approach for efficiency
                for j in range(seq_len):
                    token_id = input_ids[i, j].item()
                    
                    # Skip special tokens
                    if token_id < 5:
                        continue
                        
                    # Define context window (tokens before and after current token)
                    window_size = 4  # Use a fixed window size for stability
                    context_start = max(0, j - window_size)
                    context_end = min(seq_len, j + window_size + 1)
                    
                    # Get context tokens, excluding the current token
                    context_indices = torch.cat([input_ids[i, context_start:j], 
                                               input_ids[i, j+1:context_end]])
                    
                    # Skip if no context tokens
                    if len(context_indices) == 0:
                        continue
                        
                    # Filter out special tokens from context
                    context_indices = context_indices[context_indices >= 5]
                    
                    # Skip if no valid context tokens remain
                    if len(context_indices) == 0:
                        continue
                        
                    # Get embeddings for context tokens
                    with torch.no_grad():
                        context_embeds = self.token_embeddings(context_indices)
                        
                        # Check for NaN values and replace with zeros
                        if torch.isnan(context_embeds).any():
                            context_embeds = torch.nan_to_num(context_embeds, nan=0.0)
                        
                        # Average context embeddings
                        context_avg = torch.mean(context_embeds, dim=0)
                        
                        # Skip if context is all zeros or NaN
                        if torch.all(context_avg == 0) or torch.isnan(context_avg).any():
                            continue
                            
                        # Normalize for stability
                        context_norm = F.normalize(context_avg, p=2, dim=0)
                        
                        # Store normalized context for this token
                        if token_id not in token_to_context:
                            token_to_context[token_id] = []
                        token_to_context[token_id].append(context_norm)
            
            # Second pass: update fast weights for each unique token
            for token_id in token_to_context:
                # Average all contexts for this token
                contexts = token_to_context[token_id]
                if not contexts:
                    continue
                    
                # Stack and average all context vectors for this token
                all_contexts = torch.stack(contexts)
                avg_context = torch.mean(all_contexts, dim=0)
                
                # Normalize again for stability
                avg_context = F.normalize(avg_context, p=2, dim=0)
                
                # Get current token embedding
                token_embed = self.token_embeddings.weight[token_id]
                
                # Compute cosine similarity to determine update magnitude
                similarity = F.cosine_similarity(
                    token_embed.unsqueeze(0),
                    avg_context.unsqueeze(0),
                    dim=1
                ).item()
                
                # Use adaptive update rate based on similarity
                # Less update for tokens already similar to their context
                adaptive_rate = self.update_rate * (1.0 - max(0, min(0.9, similarity)))
                
                # Scale down update rate for very frequent tokens
                if self.use_frequency_stats and self.token_frequencies[token_id] > 100:
                    freq_factor = 1.0 / (1.0 + torch.log(self.token_frequencies[token_id] / 100))
                    adaptive_rate *= freq_factor
                
                # Compute update vector
                update = adaptive_rate * avg_context
                
                # Clip update to prevent extreme values
                update = torch.clamp(update, min=-0.05, max=0.05)
                
                # Apply update with stability check
                if not torch.isnan(update).any() and not torch.isinf(update).any():
                    self.fast_token_weights[token_id] += update
                    
                    # Apply L2 normalization to fast weights periodically
                    if torch.rand(1).item() < 0.01:  # 1% chance to normalize
                        norm = torch.norm(self.fast_token_weights[token_id], p=2)
                        if norm > 1.0:
                            self.fast_token_weights[token_id] = F.normalize(
                                self.fast_token_weights[token_id], p=2, dim=0)
        
        # Create updated contexts dictionary
        updated_contexts = token_contexts.copy() if token_contexts else {}
        
        # Update contexts with current token representations
        for i in range(batch_size):
            for j in range(seq_len):
                token_id = input_ids[i, j].item()
                # Skip special tokens
                if token_id < 5:
                    continue
                # Store token's contextual representation
                updated_contexts[token_id] = embeddings[i, j].detach().clone()
        
        # Apply dropout
        embeddings = self.dropout(embeddings)
        
        return embeddings, updated_contexts

        embeddings = self.dropout(embeddings)
        
        return embeddings, updated_contexts
