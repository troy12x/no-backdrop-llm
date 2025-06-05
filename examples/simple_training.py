"""
Simple example of training a NoBackdrop model on a text dataset.
"""

import os
import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

# Add the project root to the path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  

from no_backdrop.model.hebbian_lm import HebbianLM
from no_backdrop.training.trainer import Trainer
from no_backdrop.training.data_utils import prepare_dataloaders


def main(args):
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    
    # Ensure the tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    dataset = load_dataset(args.dataset)
    
    # Process dataset to use only the 'input' column
    def process_dataset(examples):
        return {"text": examples["prompt"]}
    
    dataset = dataset.map(process_dataset, batched=True)
    
    # Extract texts and ensure they're long enough for training
    raw_texts = dataset["train"]["text"][:args.max_samples]
    
    # Filter out texts that are too short and ensure they're long enough for chunking
    min_length = 100  # Minimum character length to be useful for training
    filtered_texts = [text for text in raw_texts if len(text) >= min_length]
    
    if len(filtered_texts) < 10:
        # If we don't have enough long texts, concatenate shorter ones
        print("Warning: Not enough long texts found. Concatenating shorter texts.")
        combined_text = " ".join(raw_texts)
        # Split into chunks of reasonable size
        chunk_size = 500
        filtered_texts = [combined_text[i:i+chunk_size] for i in range(0, len(combined_text), chunk_size)]
    
    # Repeat texts to ensure we have enough data for multiple training steps
    train_texts = []
    for _ in range(10):  # Repeat 10 times to ensure multiple batches
        train_texts.extend(filtered_texts)
    
    # Create evaluation texts (use the same texts to ensure they're processed)
    eval_texts = filtered_texts[:20]  # Use first 20 filtered texts for evaluation
    
    print(f"Using {len(filtered_texts)} unique texts, repeated to create {len(train_texts)} training examples")
    
    print(f"Loaded {len(train_texts)} training examples and {len(eval_texts)} validation examples")
    
    # Prepare dataloaders
    train_dataloader, eval_dataloader = prepare_dataloaders(
        train_texts=train_texts,
        eval_texts=eval_texts,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        stride=args.stride,
        num_workers=0,  # Set to 0 to avoid multiprocessing pickling errors
        pad_token_id=tokenizer.pad_token_id,
    )
    
    # Create model
    # Set intermediate size to 4x hidden size for a typical transformer architecture
    intermediate_size = args.hidden_size * 4
    
    model = HebbianLM(
        vocab_size=tokenizer.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=intermediate_size,  # Added explicit intermediate size
        window_size=args.window_size,
        dropout=args.dropout,
        max_position_embeddings=args.max_length,
        update_rate=args.update_rate,
        use_fast_weights=args.use_fast_weights,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id,
        eos_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        device=args.device,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        checkpoint_dir=args.output_dir,
        use_wandb=args.use_wandb,
        save_best_only=args.save_best_only,
    )
    
    # Train model
    trainer.train(
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        num_epochs=args.num_epochs,
    )
    
    # Generate some example text
    prompts = [
        "Imagine you are an experienced",
        "Using",
    ]
    
    print("\nGenerating example texts:")
    for prompt in prompts:
        generated_texts = trainer.generate_text(
            prompt=prompt,
            tokenizer=tokenizer,
            max_length=30,  # Shorter length for more natural responses
            temperature=0.8,  # Slightly higher temperature for more diversity
            top_k=10,
            top_p=0.9,
            update_model=True,
        )
        
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated_texts[0]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a NoBackdrop model")
    
    # Data arguments
    parser.add_argument("--dataset", type=str, default="fka/awesome-chatgpt-prompts", help="Dataset to use")
    parser.add_argument("--tokenizer_name", type=str, default="gpt2", help="Tokenizer name")
    parser.add_argument("--max_samples", type=int, default=1000, help="Maximum number of samples to use")
    
    # Model arguments
    parser.add_argument("--hidden_size", type=int, default=384, help="Hidden size")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--num_heads", type=int, default=6, help="Number of attention heads")
    parser.add_argument("--window_size", type=int, default=1024, help="Attention window size")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--update_rate", type=float, default=0.01, help="Fast weight update rate")
    parser.add_argument("--use_fast_weights", action="store_true", help="Use fast weights")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--stride", type=int, default=512, help="Stride for sliding window")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs")
    
    # Logging and saving arguments
    parser.add_argument("--log_interval", type=int, default=10, help="Logging interval")
    parser.add_argument("--eval_interval", type=int, default=100, help="Evaluation interval")
    parser.add_argument("--save_interval", type=int, default=5000, help="Saving interval")
    parser.add_argument("--save_best_only", action="store_true", help="Only save the best model")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    
    # Device arguments
    parser.add_argument("--device", type=str, default=None, help="Device to use (cpu or cuda)")
    
    args = parser.parse_args()
    
    main(args)
