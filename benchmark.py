"""
Phase 3 — Inference benchmarking engine.
Measures: tokens/sec, time to first token, total latency, memory usage.
Runs all three models at temperatures 0.0 and 0.7.

Usage:
  python benchmark.py                      # full benchmark, all models
  python benchmark.py --model llama3.2:3b  # single model only
  python benchmark.py --quick              # 3 prompts instead of 10
"""
import time
import json
import csv
import typer
import psutil
import ollama
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from config import MODELS, TEMPERATURES, RESULTS_DIR

app = typer.Typer(add_completion=False)
console = Console()

BENCHMARK_PROMPTS = [
    "Explain what a transformer neural network is in two sentences.",
    "What is the difference between supervised and unsupervised learning?",
    "Write a Python function that reverses a string.",
    "What are three benefits of containerization with Docker?",
    "Explain gradient descent in simple terms.",
    "What is the capital of France and what is it known for?",
    "Write a haiku about artificial intelligence.",
    "What does REST stand for and what are its key principles?",
    "Explain the difference between RAM and disk storage.",
    "Summarize what the Turing test measures.",
]

QUICK_PROMPTS = BENCHMARK_PROMPTS[:3]


def measure_inference(model: str, prompt: str, temperature: float) -> dict:
    """Run one inference call and return detailed metrics."""
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1e6  # MB

    start = time.perf_counter()
    first_token_time = None
    full_response = ""
    chunk_count = 0

    try:
        for chunk in ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            options={"temperature": temperature, "seed": 42}
        ):
            token = chunk["message"]["content"]
            if first_token_time is None and token.strip():
                first_token_time = time.perf_counter() - start
            full_response += token
            chunk_count += 1

    except Exception as e:
        return {"error": str(e)}

    total_time = time.perf_counter() - start
    mem_after = process.memory_info().rss / 1e6
    word_count = len(full_response.split())

    return {
        "model": model,
        "temperature": temperature,
        "prompt_preview": prompt[:60] + "..." if len(prompt) > 60 else prompt,
        "response_preview": full_response[:80] + "..." if len(full_response) > 80 else full_response,
        "time_to_first_token_s": round(first_token_time or 0, 3),
        "total_latency_s": round(total_time, 3),
        "word_count": word_count,
        "approx_words_per_sec": round(word_count / total_time, 2) if total_time > 0 else 0,
        "memory_delta_mb": round(mem_after - mem_before, 1),
    }


@app.command()
def run(
    model: str = typer.Option(None, "--model", "-m", help="Run single model only"),
    quick: bool = typer.Option(False, "--quick", help="Use 3 prompts instead of 10"),
    temperatures: str = typer.Option("0.0,0.7", "--temps", help="Comma-separated temperatures"),
):
    """Benchmark all models and export results to CSV."""

    models_to_run = [model] if model else list(MODELS.keys())
    temps = [float(t) for t in temperatures.split(",")]
    prompts = QUICK_PROMPTS if quick else BENCHMARK_PROMPTS

    console.print(f"\n[bold]Benchmarking {len(models_to_run)} model(s) × "
                  f"{len(temps)} temperature(s) × {len(prompts)} prompts[/bold]")
    console.print(f"[dim]Total runs: {len(models_to_run) * len(temps) * len(prompts)}[/dim]\n")

    all_results = []

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        for mdl in models_to_run:
            for temp in temps:
                task = progress.add_task(
                    f"[cyan]{MODELS[mdl]['display_name']}[/cyan] @ temp={temp}",
                    total=len(prompts)
                )
                for prompt in prompts:
                    result = measure_inference(mdl, prompt, temp)
                    all_results.append(result)
                    progress.advance(task)

    # Save raw JSON
    Path(RESULTS_DIR).mkdir(exist_ok=True)
    ts = int(time.time())
    json_path = Path(RESULTS_DIR) / f"benchmark_{ts}.json"
    json_path.write_text(json.dumps(all_results, indent=2))

    # Save CSV
    csv_path = Path(RESULTS_DIR) / f"benchmark_{ts}.csv"
    if all_results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)

    # Print summary table
    console.print("\n[bold]Results Summary[/bold]\n")
    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("Model", style="cyan", min_width=16)
    table.add_column("Temp", justify="right", min_width=5)
    table.add_column("TTFT (s)", justify="right", min_width=9)
    table.add_column("Latency (s)", justify="right", min_width=11)
    table.add_column("Words/s", justify="right", min_width=8)
    table.add_column("Mem delta", justify="right", min_width=10)

    # Group and average by model+temp
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in all_results:
        if "error" not in r:
            grouped[(r["model"], r["temperature"])].append(r)

    for (mdl, temp), runs in sorted(grouped.items()):
        avg = lambda k: sum(r[k] for r in runs) / len(runs)
        table.add_row(
            MODELS[mdl]["display_name"],
            str(temp),
            f"{avg('time_to_first_token_s'):.2f}",
            f"{avg('total_latency_s'):.2f}",
            f"{avg('approx_words_per_sec'):.1f}",
            f"{avg('memory_delta_mb'):.0f} MB",
        )

    console.print(table)
    console.print(f"\n[dim]Raw results → {json_path}[/dim]")
    console.print(f"[dim]CSV export  → {csv_path}[/dim]")


if __name__ == "__main__":
    app()
