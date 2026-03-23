# Hello World SLM Finetuning

A minimal end-to-end example of fine-tuning a Small Language Model (SLM) on a custom dataset, with automated training via GitHub Actions and results published to GitHub Pages.

## What It Does

Fine-tunes **DistilGPT2** (82M parameters) on 30 hand-crafted greeting examples, then compares the base model vs fine-tuned outputs.

| Prompt | Base Model | Fine-tuned |
|--------|-----------|------------|
| `Hello` | *(generic text)* | Friendly greeting response |
| `Good morning` | *(generic text)* | Warm morning reply |
| `Hello world!` | *(generic text)* | Enthusiastic hello world response |

## Quick Start

```bash
pip install -r requirements.txt

# Fine-tune
python src/train.py

# Compare outputs
python src/inference.py

# Build report
python src/generate_report.py && open docs/index.html
```

## Project Structure

```
hello-finetuning/
├── .github/workflows/
│   ├── finetune.yml        # Train + publish on push to main
│   └── pages-preview.yml   # Deploy placeholder page
├── data/
│   └── hello_world.jsonl   # 30 greeting training examples
├── docs/
│   └── index.html          # GitHub Pages report
├── src/
│   ├── train.py            # Fine-tunes DistilGPT2
│   ├── inference.py        # Base vs fine-tuned comparison
│   └── generate_report.py  # Generates docs/index.html
└── requirements.txt
```

## GitHub Actions

The `finetune.yml` workflow:
1. Triggers on push to `main` (when `data/` or `src/` changes)
2. Installs dependencies and fine-tunes DistilGPT2
3. Runs inference comparison
4. Generates the HTML report
5. Deploys to GitHub Pages
6. Uploads the trained model as a downloadable artifact (30-day retention)

You can also trigger it manually via **Actions → Fine-tune SLM → Run workflow** with custom `epochs` and `learning_rate` inputs.

## GitHub Pages Setup

Enable Pages in your repo settings:
- **Settings → Pages → Source**: `GitHub Actions`

The live report will be at `https://<user>.github.io/<repo>/`.

## Model Details

| | Value |
|---|---|
| Architecture | DistilGPT2 (distilled GPT-2) |
| Parameters | 82M |
| Training objective | Causal language modelling |
| Epochs | 5 |
| Learning rate | 5e-5 |
| Batch size | 4 |
| Hardware | CPU (GitHub Actions ubuntu-latest) |
