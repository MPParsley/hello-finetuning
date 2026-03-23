#!/usr/bin/env python3
"""
Generates a self-contained docs/index.html report from training and inference results.
Deployed to GitHub Pages.
"""

import json
import os
from pathlib import Path
from datetime import datetime


def load_json(path: str, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def format_number(n):
    if n is None:
        return "N/A"
    if isinstance(n, float):
        return f"{n:.4f}"
    return f"{n:,}"


def main():
    summary = load_json("output/training_summary.json", {})
    results = load_json("output/inference_results.json", [])
    run_date = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # Build inference rows
    inference_rows = ""
    for r in results:
        prompt = r.get("prompt", "")
        base = r.get("base_output", "")
        ft = r.get("finetuned_output", "")
        inference_rows += f"""
        <tr>
          <td class="prompt">{prompt}</td>
          <td class="base">{base}</td>
          <td class="finetuned">{ft}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Hello World SLM Finetuning</title>
  <style>
    :root {{
      --bg: #0d1117; --surface: #161b22; --border: #30363d;
      --accent: #58a6ff; --green: #3fb950; --orange: #d29922;
      --text: #c9d1d9; --muted: #8b949e;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; line-height: 1.6; }}
    header {{ background: var(--surface); border-bottom: 1px solid var(--border); padding: 2rem; text-align: center; }}
    header h1 {{ font-size: 2rem; color: var(--accent); }}
    header p {{ color: var(--muted); margin-top: 0.5rem; }}
    .badge {{ display: inline-block; background: var(--green); color: #000; font-size: 0.75rem; font-weight: 600; padding: 0.2rem 0.6rem; border-radius: 2rem; margin: 0.25rem; }}
    .badge.orange {{ background: var(--orange); }}
    main {{ max-width: 1100px; margin: 2rem auto; padding: 0 1rem; }}
    section {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1.5rem; margin-bottom: 2rem; }}
    h2 {{ color: var(--accent); font-size: 1.25rem; margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; }}
    .metric {{ background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 1rem; text-align: center; }}
    .metric .value {{ font-size: 1.6rem; font-weight: 700; color: var(--accent); }}
    .metric .label {{ font-size: 0.8rem; color: var(--muted); margin-top: 0.25rem; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
    th {{ background: var(--bg); color: var(--muted); text-align: left; padding: 0.75rem 1rem; border-bottom: 1px solid var(--border); font-weight: 600; }}
    td {{ padding: 0.75rem 1rem; border-bottom: 1px solid var(--border); vertical-align: top; }}
    tr:last-child td {{ border-bottom: none; }}
    td.prompt {{ color: var(--accent); font-weight: 600; white-space: nowrap; }}
    td.base {{ color: var(--muted); font-style: italic; }}
    td.finetuned {{ color: var(--green); }}
    .pipeline {{ display: flex; gap: 0; flex-wrap: wrap; }}
    .step {{ flex: 1; min-width: 160px; background: var(--bg); border: 1px solid var(--border); padding: 1rem; text-align: center; position: relative; }}
    .step:not(:last-child)::after {{ content: "→"; position: absolute; right: -1px; top: 50%; transform: translateY(-50%); color: var(--accent); font-size: 1.2rem; z-index: 1; }}
    .step .icon {{ font-size: 1.5rem; }}
    .step .title {{ font-weight: 600; color: var(--accent); font-size: 0.9rem; margin-top: 0.25rem; }}
    .step .desc {{ font-size: 0.75rem; color: var(--muted); margin-top: 0.25rem; }}
    footer {{ text-align: center; color: var(--muted); font-size: 0.8rem; padding: 2rem; }}
    code {{ background: var(--bg); border: 1px solid var(--border); padding: 0.1rem 0.4rem; border-radius: 4px; font-size: 0.85rem; }}
  </style>
</head>
<body>
  <header>
    <h1>🤖 Hello World SLM Finetuning</h1>
    <p>Fine-tuning <strong>DistilGPT2</strong> on greeting conversations</p>
    <div style="margin-top:1rem;">
      <span class="badge">DistilGPT2 · 82M params</span>
      <span class="badge">HuggingFace Transformers</span>
      <span class="badge orange">GitHub Actions CI</span>
      <span class="badge">GitHub Pages</span>
    </div>
    <p style="margin-top:0.75rem; color: var(--muted); font-size:0.85rem;">Generated: {run_date}</p>
  </header>

  <main>

    <section>
      <h2>📋 Pipeline Overview</h2>
      <div class="pipeline">
        <div class="step"><div class="icon">📄</div><div class="title">Training Data</div><div class="desc">30 greeting JSONL examples</div></div>
        <div class="step"><div class="icon">🏋️</div><div class="title">Fine-tune</div><div class="desc">DistilGPT2 with HuggingFace Trainer</div></div>
        <div class="step"><div class="icon">📊</div><div class="title">Evaluate</div><div class="desc">Compare base vs fine-tuned outputs</div></div>
        <div class="step"><div class="icon">🌐</div><div class="title">Publish</div><div class="desc">Results on GitHub Pages</div></div>
      </div>
    </section>

    <section>
      <h2>📈 Training Metrics</h2>
      <div class="metrics">
        <div class="metric"><div class="value">{summary.get('model_name', 'distilgpt2')}</div><div class="label">Model</div></div>
        <div class="metric"><div class="value">{format_number(summary.get('total_params'))}</div><div class="label">Parameters</div></div>
        <div class="metric"><div class="value">{summary.get('dataset_size', 'N/A')}</div><div class="label">Training Examples</div></div>
        <div class="metric"><div class="value">{summary.get('epochs', 'N/A')}</div><div class="label">Epochs</div></div>
        <div class="metric"><div class="value">{format_number(summary.get('final_loss'))}</div><div class="label">Final Loss</div></div>
        <div class="metric"><div class="value">{summary.get('training_time_seconds', 'N/A')}s</div><div class="label">Training Time</div></div>
        <div class="metric"><div class="value">{summary.get('learning_rate', 'N/A')}</div><div class="label">Learning Rate</div></div>
        <div class="metric"><div class="value">{summary.get('batch_size', 'N/A')}</div><div class="label">Batch Size</div></div>
      </div>
    </section>

    <section>
      <h2>🔬 Base vs Fine-tuned Outputs</h2>
      <p style="color: var(--muted); font-size: 0.85rem; margin-bottom: 1rem;">
        Comparing completions from the original <code>distilgpt2</code> checkpoint vs after fine-tuning on greeting data.
      </p>
      <table>
        <thead>
          <tr>
            <th>Prompt</th>
            <th>🤖 Base Model</th>
            <th>✨ Fine-tuned</th>
          </tr>
        </thead>
        <tbody>{inference_rows}
        </tbody>
      </table>
    </section>

    <section>
      <h2>⚙️ How It Works</h2>
      <table>
        <tr><th>Component</th><th>Details</th></tr>
        <tr><td>Model</td><td><code>distilgpt2</code> — 82M parameter causal language model (distilled from GPT-2)</td></tr>
        <tr><td>Dataset</td><td>30 hand-crafted greeting pairs in JSONL format (<code>data/hello_world.jsonl</code>)</td></tr>
        <tr><td>Training</td><td>HuggingFace <code>Trainer</code> with causal LM objective, {summary.get('epochs', 5)} epochs, lr={summary.get('learning_rate', '5e-5')}</td></tr>
        <tr><td>CI/CD</td><td>GitHub Actions runs training on every push to <code>main</code>, uploads model artifact</td></tr>
        <tr><td>Reporting</td><td>This page auto-generated by <code>src/generate_report.py</code> and deployed via GitHub Pages</td></tr>
      </table>
    </section>

  </main>

  <footer>
    <p>Built with ❤️ using HuggingFace Transformers · GitHub Actions · GitHub Pages</p>
    <p style="margin-top:0.5rem;"><a href="https://github.com" style="color: var(--accent);">View source on GitHub</a></p>
  </footer>
</body>
</html>
"""

    Path("docs").mkdir(exist_ok=True)
    with open("docs/index.html", "w") as f:
        f.write(html)
    print("Report written to docs/index.html")


if __name__ == "__main__":
    main()
