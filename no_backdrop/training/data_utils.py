"""
Data utilities for training NoBackdrop models.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from transformers import PreTrainedTokenizer


class TextDataset(Dataset):
    """
    Dataset for text data with efficient tokenization and chunking.
    """
    
    def __init__(
        self,
        texts: List[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        stride: int = 128,
    ):
        """
        Initialize the dataset.
        
        Args:
            texts: List of text strings
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
            stride: Stride for overlapping chunks
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        
        # Tokenize and chunk texts
        self.examples = []
        for text in texts:
            self.examples.extend(self._tokenize_and_chunk(text))
    
    def _tokenize_and_chunk(self, text: str) -> List[Dict[str, torch.Tensor]]:
        """
        Tokenize text and split into overlapping chunks.
        
        Args:
            text: Input text string
            
        Returns:
            chunks: List of tokenized chunks
        """
        # Tokenize text
        tokenized = self.tokenizer(text, return_tensors="pt", add_special_tokens=True)
        input_ids = tokenized["input_ids"][0]
        attention_mask = tokenized["attention_mask"][0]
        
        # Handle short texts by padding them if necessary
        if len(input_ids) <= 2:  # Just special tokens
            # Create a minimal example with padding
            padded_input_ids = torch.cat([input_ids, torch.full((4,), self.tokenizer.pad_token_id, dtype=torch.long)])
            padded_attention_mask = torch.cat([attention_mask, torch.zeros(4, dtype=torch.long)])
            return [{
                "input_ids": padded_input_ids,
                "attention_mask": padded_attention_mask,
            }]
        
        # Split into chunks
        chunks = []
        for i in range(0, len(input_ids) - 2, self.stride):
            # Get chunk of appropriate length
            end = min(i + self.max_length, len(input_ids))
            
            # For short chunks, pad them instead of skipping
            if end - i < self.max_length // 4:
                # Only do this for the last chunk
                if i + self.stride >= len(input_ids) - 2:
                    # Pad the last chunk if it's too short
                    chunk_input_ids = input_ids[i:end].clone()
                    chunk_attention_mask = attention_mask[i:end].clone()
                    
                    # Pad to at least 1/4 of max_length
                    min_length = self.max_length // 4
                    pad_length = max(0, min_length - len(chunk_input_ids))
                    
                    if pad_length > 0:
                        chunk_input_ids = torch.cat([chunk_input_ids, torch.full((pad_length,), self.tokenizer.pad_token_id, dtype=torch.long)])
                        chunk_attention_mask = torch.cat([chunk_attention_mask, torch.zeros(pad_length, dtype=torch.long)])
                    
                    chunks.append({
                        "input_ids": chunk_input_ids,
                        "attention_mask": chunk_attention_mask,
                    })
                continue
            
            # Extract chunk
            chunk_input_ids = input_ids[i:end].clone()
            chunk_attention_mask = attention_mask[i:end].clone()
            
            # Create example
            example = {
                "input_ids": chunk_input_ids,
                "attention_mask": chunk_attention_mask,
            }
            
            chunks.append(example)
            
            # Stop if we've reached the end
            if end == len(input_ids):
                break
        
        return chunks
    
    def __len__(self) -> int:
        """Get the number of examples in the dataset."""
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get an example from the dataset."""
        return self.examples[idx]


def collate_batch(examples: List[Dict[str, torch.Tensor]], pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    """
    Collate function for batching examples with padding.
    
    Args:
        examples: List of examples
        pad_token_id: Token ID to use for padding
        
    Returns:
        batch: Batched examples with padding
    """
    # Get maximum sequence length in the batch
    max_length = max(len(example["input_ids"]) for example in examples)
    
    # Initialize tensors
    batch_size = len(examples)
    input_ids = torch.full((batch_size, max_length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
    
    # Fill tensors with data
    for i, example in enumerate(examples):
        seq_length = len(example["input_ids"])
        input_ids[i, :seq_length] = example["input_ids"]
        attention_mask[i, :seq_length] = example["attention_mask"]
    
    # Create batch
    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    
    return batch


def prepare_dataloaders(
    train_texts: List[str],
    eval_texts: List[str],
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 8,
    max_length: int = 512,
    stride: int = 128,
    num_workers: int = 4,
    pad_token_id: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Prepare training and evaluation dataloaders.
    
    Args:
        train_texts: List of training text strings
        eval_texts: List of evaluation text strings
        tokenizer: Tokenizer to use
        batch_size: Batch size
        max_length: Maximum sequence length
        stride: Stride for overlapping chunks
        num_workers: Number of workers for data loading
        pad_token_id: Token ID to use for padding
        
    Returns:
        train_dataloader: DataLoader for training
        eval_dataloader: DataLoader for evaluation
    """
    # Create datasets
    train_dataset = TextDataset(
        texts=train_texts,
        tokenizer=tokenizer,
        max_length=max_length,
        stride=stride,
    )
    
    eval_dataset = TextDataset(
        texts=eval_texts,
        tokenizer=tokenizer,
        max_length=max_length,
        stride=stride,
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda examples: collate_batch(examples, pad_token_id),
        pin_memory=True,
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda examples: collate_batch(examples, pad_token_id),
        pin_memory=True,
    )
    
    return train_dataloader, eval_dataloader
