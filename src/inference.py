#!/usr/bin/env python3
"""
Compare outputs: base DistilGPT2 vs fine-tuned model.
Saves results to output/inference_results.json for the GitHub Pages report.
"""

import json
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


PROMPTS = [
    "Hello",
    "Good morning",
    "Hi there",
    "Hey",
    "Hello world",
    "Greetings",
    "Good evening",
    "Howdy",
]


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 60) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
        )
    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    # Return only the newly generated text
    return generated[len(prompt):].strip()


def load_model(path: str):
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(path)
    model.eval()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetuned", default="output/model", help="Fine-tuned model path")
    parser.add_argument("--base", default="distilgpt2", help="Base model name/path")
    args = parser.parse_args()

    print(f"Loading base model: {args.base}")
    base_model, base_tokenizer = load_model(args.base)

    print(f"Loading fine-tuned model: {args.finetuned}")
    ft_model, ft_tokenizer = load_model(args.finetuned)

    results = []
    for prompt in PROMPTS:
        print(f"\nPrompt: '{prompt}'")
        base_out = generate(base_model, base_tokenizer, prompt)
        ft_out = generate(ft_model, ft_tokenizer, prompt)
        print(f"  Base:        {base_out!r}")
        print(f"  Fine-tuned:  {ft_out!r}")
        results.append({
            "prompt": prompt,
            "base_output": base_out,
            "finetuned_output": ft_out,
        })

    Path("output").mkdir(exist_ok=True)
    with open("output/inference_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to output/inference_results.json")


if __name__ == "__main__":
    main()
