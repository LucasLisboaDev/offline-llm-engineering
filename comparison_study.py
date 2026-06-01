"""
Phase 5 — Model comparison study.
Runs 30 standardized prompts across all three models at temp 0 and 0.7.
Exports a full CSV + prints a summary report.

Usage:
  python comparison_study.py           # full study (takes 20-40 min)
  python comparison_study.py --quick   # 5 prompts, quick validation
"""
import time
import json
import csv
import typer
import ollama
from pathlib import Path
from collections import defaultdict
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from config import MODELS, TEMPERATURES, RESULTS_DIR

app = typer.Typer(add_completion=False)
console = Console()

# 30 standardized prompts across 6 categories (5 each)
COMPARISON_PROMPTS = [
    # Factual recall
    {"id": "F01", "category": "factual", "prompt": "What is the capital of Australia?"},
    {"id": "F02", "category": "factual", "prompt": "In what year did the Berlin Wall fall?"},
    {"id": "F03", "category": "factual", "prompt": "What does HTTP stand for?"},
    {"id": "F04", "category": "factual", "prompt": "What is the chemical symbol for gold?"},
    {"id": "F05", "category": "factual", "prompt": "How many planets are in our solar system?"},
    # Reasoning
    {"id": "R01", "category": "reasoning", "prompt": "If a train travels 60 mph for 2.5 hours, how far does it go? Show your work."},
    {"id": "R02", "category": "reasoning", "prompt": "A bat and ball cost $1.10. The bat costs $1 more than the ball. How much does the ball cost?"},
    {"id": "R03", "category": "reasoning", "prompt": "What comes next in this sequence: 2, 6, 18, 54, ___? Explain why."},
    {"id": "R04", "category": "reasoning", "prompt": "If all Bloops are Razzles and all Razzles are Lazzles, are all Bloops definitely Lazzles?"},
    {"id": "R05", "category": "reasoning", "prompt": "I have 3 apples. I give away half, then receive 4 more. How many do I have?"},
    # Coding
    {"id": "C01", "category": "coding", "prompt": "Write a Python function to check if a number is prime."},
    {"id": "C02", "category": "coding", "prompt": "Write a Python one-liner to flatten a list of lists."},
    {"id": "C03", "category": "coding", "prompt": "What is wrong with this code: for i in range(10): print(i) if i == 5 break"},
    {"id": "C04", "category": "coding", "prompt": "Write a SQL query to find duplicate email addresses in a users table."},
    {"id": "C05", "category": "coding", "prompt": "Explain what this does: lambda x: x if x <= 1 else x * f(x-1)"},
    # Summarization
    {"id": "S01", "category": "summarization", "prompt": "Summarize the concept of machine learning in exactly two sentences."},
    {"id": "S02", "category": "summarization", "prompt": "Explain blockchain to a 10-year-old in three sentences."},
    {"id": "S03", "category": "summarization", "prompt": "What is DevOps? Answer in under 50 words."},
    {"id": "S04", "category": "summarization", "prompt": "Describe the water cycle in two sentences."},
    {"id": "S05", "category": "summarization", "prompt": "What is Docker and why do developers use it? Keep it brief."},
    # Creative
    {"id": "CR01", "category": "creative", "prompt": "Write a two-sentence story about a robot discovering music."},
    {"id": "CR02", "category": "creative", "prompt": "Give me three creative names for a coffee shop that specializes in AI-themed drinks."},
    {"id": "CR03", "category": "creative", "prompt": "Write a haiku about debugging code at 2am."},
    {"id": "CR04", "category": "creative", "prompt": "Describe the color blue to someone who has never seen color."},
    {"id": "CR05", "category": "creative", "prompt": "Write a one-paragraph product description for an invisible umbrella."},
    # Instruction following
    {"id": "I01", "category": "instruction", "prompt": "List exactly 4 programming languages, one per line, nothing else."},
    {"id": "I02", "category": "instruction", "prompt": "Reply with only the number of words in this sentence: The quick brown fox jumps."},
    {"id": "I03", "category": "instruction", "prompt": "Translate 'Hello, how are you?' to Spanish. Reply with only the translation."},
    {"id": "I04", "category": "instruction", "prompt": "Write the alphabet backwards. Only the letters, no spaces or punctuation."},
    {"id": "I05", "category": "instruction", "prompt": "Give me a word that rhymes with 'orange'. One word only."},
]

QUICK_PROMPTS = COMPARISON_PROMPTS[:5]


def run_single(model: str, prompt_item: dict, temperature: float) -> dict:
    start = time.perf_counter()
    first_token_time = None
    response_text = ""

    try:
        for chunk in ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt_item["prompt"]}],
            stream=True,
            options={"temperature": temperature, "seed": 42,"num_predict": 300}
        ):
            token = chunk["message"]["content"]
            if first_token_time is None and token.strip():
                first_token_time = time.perf_counter() - start
            response_text += token

        total_time = time.perf_counter() - start
        word_count = len(response_text.split())

        return {
            "prompt_id": prompt_item["id"],
            "category": prompt_item["category"],
            "model": model,
            "model_display": MODELS[model]["display_name"],
            "temperature": temperature,
            "prompt": prompt_item["prompt"],
            "response": response_text.strip(),
            "time_to_first_token_s": round(first_token_time or 0, 3),
            "total_latency_s": round(total_time, 3),
            "word_count": word_count,
            "words_per_sec": round(word_count / total_time, 2) if total_time > 0 else 0,
            "error": None,
        }

    except Exception as e:
        return {
            "prompt_id": prompt_item["id"],
            "category": prompt_item["category"],
            "model": model,
            "model_display": MODELS[model]["display_name"],
            "temperature": temperature,
            "prompt": prompt_item["prompt"],
            "response": "",
            "time_to_first_token_s": 0,
            "total_latency_s": 0,
            "word_count": 0,
            "words_per_sec": 0,
            "error": str(e),
        }


@app.command()
def run(
    quick: bool = typer.Option(False, "--quick", help="Run 5 prompts only for testing"),
    models_filter: str = typer.Option(None, "--model", "-m", help="Run one model only"),
):
    """Run the full model comparison study and export results."""

    prompts = QUICK_PROMPTS if quick else COMPARISON_PROMPTS
    models = [models_filter] if models_filter else list(MODELS.keys())
    total = len(models) * len(TEMPERATURES) * len(prompts)

    console.print(f"\n[bold]Model Comparison Study[/bold]")
    console.print(f"[dim]{len(models)} models × {len(TEMPERATURES)} temps × {len(prompts)} prompts = {total} runs[/dim]")
    if not quick:
        console.print(f"[dim]Estimated time: {total * 15 // 60}–{total * 30 // 60} minutes on CPU[/dim]\n")

    all_results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console
    ) as progress:
        main_task = progress.add_task("Overall progress", total=total)

        for model in models:
            for temp in TEMPERATURES:
                for p in prompts:
                    desc = f"[cyan]{MODELS[model]['display_name']}[/cyan] T={temp} [{p['id']}]"
                    progress.update(main_task, description=desc)
                    result = run_single(model, p, temp)
                    all_results.append(result)
                    progress.advance(main_task)

    # Save outputs
    Path(RESULTS_DIR).mkdir(exist_ok=True)
    ts = int(time.time())

    json_path = Path(RESULTS_DIR) / f"comparison_{ts}.json"
    json_path.write_text(json.dumps(all_results, indent=2))

    csv_path = Path(RESULTS_DIR) / f"comparison_{ts}.csv"
    if all_results:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)

    # Summary table
    console.print("\n[bold]Performance Summary (averages across all prompts)[/bold]\n")
    table = Table(show_header=True, header_style="bold", box=None)
    table.add_column("Model", style="cyan")
    table.add_column("Temp", justify="right")
    table.add_column("Avg TTFT", justify="right")
    table.add_column("Avg latency", justify="right")
    table.add_column("Avg words/s", justify="right")
    table.add_column("Avg length", justify="right")

    grouped = defaultdict(list)
    for r in all_results:
        if not r["error"]:
            grouped[(r["model"], r["temperature"])].append(r)

    for (mdl, temp), runs in sorted(grouped.items()):
        avg = lambda k: sum(r[k] for r in runs) / len(runs)
        table.add_row(
            MODELS[mdl]["display_name"],
            str(temp),
            f"{avg('time_to_first_token_s'):.2f}s",
            f"{avg('total_latency_s'):.2f}s",
            f"{avg('words_per_sec'):.1f}",
            f"{avg('word_count'):.0f} words",
        )

    console.print(table)
    console.print(f"\n[dim]Full results → {json_path}[/dim]")
    console.print(f"[dim]CSV export  → {csv_path}[/dim]")
    console.print("\n[dim]Tip: open the CSV in Excel or run: python analyze_results.py[/dim]")


if __name__ == "__main__":
    app()
