"""
Phase 2 — Command-line interface for local AI assistant.
Usage:
  python cli.py "Your question here"
  python cli.py "Your question" --model mistral:7b
  python cli.py "Your question" --model phi4-mini --temp 0.0
  python cli.py "Your question" --no-stream
"""
import time
import json
import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from pathlib import Path
import ollama
from config import MODELS, RESULTS_DIR

app = typer.Typer(add_completion=False)
console = Console()

MODEL_NAMES = list(MODELS.keys())


@app.command()
def ask(
    prompt: str = typer.Argument(..., help="The question or prompt to send"),
    model: str = typer.Option("llama3.2:3b", "--model", "-m", help=f"Model: {MODEL_NAMES}"),
    temperature: float = typer.Option(0.7, "--temp", "-t", help="Temperature: 0.0 (deterministic) to 1.0 (creative)"),
    stream: bool = typer.Option(True, "--stream/--no-stream", help="Stream tokens as they generate"),
    save: bool = typer.Option(False, "--save", "-s", help="Save result to results/"),
):
    """Ask a local AI model a question. Runs 100% offline."""

    if model not in MODEL_NAMES:
        console.print(f"[red]Unknown model '{model}'. Choose from: {MODEL_NAMES}[/red]")
        raise typer.Exit(1)

    model_info = MODELS[model]
    console.print(Panel(
        f"[bold]{model_info['display_name']}[/bold]  "
        f"[dim]{model_info['params_billions']}B params · temp={temperature}[/dim]",
        title="Local AI Assistant",
        border_style="dim"
    ))
    console.print(f"[dim]Prompt:[/dim] {prompt}\n")

    start_time = time.perf_counter()
    first_token_time = None
    full_response = ""
    token_count = 0

    try:
        if stream:
            for chunk in ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                options={"temperature": temperature}
            ):
                token = chunk["message"]["content"]
                if first_token_time is None and token.strip():
                    first_token_time = time.perf_counter() - start_time
                full_response += token
                token_count += 1
                console.print(token, end="")
        else:
            console.print("[dim]Generating...[/dim]")
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                options={"temperature": temperature}
            )
            full_response = response["message"]["content"]
            first_token_time = time.perf_counter() - start_time
            token_count = len(full_response.split())
            console.print(full_response)

    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        console.print("[dim]Is 'ollama serve' running?[/dim]")
        raise typer.Exit(1)

    total_time = time.perf_counter() - start_time
    words = len(full_response.split())
    tps = words / total_time if total_time > 0 else 0

    console.print(f"\n\n[dim]─────────────────────────────────────[/dim]")
    console.print(
        f"[dim]First token: {first_token_time:.2f}s  |  "
        f"Total: {total_time:.2f}s  |  "
        f"~{tps:.1f} words/s[/dim]"
    )

    if save:
        Path(RESULTS_DIR).mkdir(exist_ok=True)
        record = {
            "model": model,
            "temperature": temperature,
            "prompt": prompt,
            "response": full_response,
            "metrics": {
                "time_to_first_token_s": round(first_token_time, 3),
                "total_latency_s": round(total_time, 3),
                "approx_words_per_sec": round(tps, 2),
            }
        }
        ts = int(time.time())
        out = Path(RESULTS_DIR) / f"cli_{model.replace(':', '_')}_{ts}.json"
        out.write_text(json.dumps(record, indent=2))
        console.print(f"[dim]Saved → {out}[/dim]")


if __name__ == "__main__":
    app()
