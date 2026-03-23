#!/usr/bin/env python3
"""
Hello World SLM Finetuning
Fine-tunes DistilGPT2 on a small greeting dataset.
"""

import json
import os
import time
import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)


class GreetingDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        with open(data_path) as f:
            for line in f:
                item = json.loads(line.strip())
                encoded = tokenizer(
                    item["text"],
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                    return_tensors="pt",
                )
                self.examples.append({
                    "input_ids": encoded["input_ids"].squeeze(),
                    "attention_mask": encoded["attention_mask"].squeeze(),
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def main():
    parser = argparse.ArgumentParser(description="Finetune DistilGPT2 on greeting data")
    parser.add_argument("--data", default="data/hello_world.jsonl", help="Path to training data")
    parser.add_argument("--output", default="output/model", help="Output directory")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    os.makedirs("output/logs", exist_ok=True)

    model_name = "distilgpt2"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.eos_token_id

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    print(f"Loading dataset from {args.data}...")
    dataset = GreetingDataset(args.data, tokenizer)
    print(f"Dataset size: {len(dataset)} examples")

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=10,
        logging_dir="output/logs",
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
        no_cuda=not torch.cuda.is_available(),
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    print("Starting training...")
    start = time.time()
    train_result = trainer.train()
    elapsed = time.time() - start
    print(f"Training complete in {elapsed:.1f}s")

    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)

    # Save training summary
    metrics = train_result.metrics
    summary = {
        "model_name": model_name,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "dataset_size": len(dataset),
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "training_time_seconds": round(elapsed, 2),
        "final_loss": round(metrics.get("train_loss", 0), 4),
        "train_samples_per_second": round(metrics.get("train_samples_per_second", 0), 2),
    }

    with open("output/training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Training summary saved to output/training_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
