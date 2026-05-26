"""
Phase 5 helper — analyze comparison results and print a formatted report.
Usage:
  python analyze_results.py results/comparison_<timestamp>.csv
  python analyze_results.py  # auto-picks the most recent comparison CSV
"""
import sys
import csv
from pathlib import Path
from collections import defaultdict
from rich.console import Console
from rich.table import Table

console = Console()


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def find_latest_csv() -> Path | None:
    results = sorted(Path("results").glob("comparison_*.csv"), reverse=True)
    return results[0] if results else None


def main():
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        path = find_latest_csv()
        if not path:
            console.print("[red]No comparison CSV found in results/. Run comparison_study.py first.[/red]")
            return
        console.print(f"[dim]Auto-loading: {path}[/dim]\n")

    rows = load_csv(path)
    if not rows:
        console.print("[red]Empty CSV.[/red]")
        return

    console.print(f"[bold]Analysis Report — {path.name}[/bold]")
    console.print(f"[dim]{len(rows)} total runs[/dim]\n")

    # Group by model + temp
    grouped = defaultdict(list)
    for r in rows:
        if not r.get("error"):
            grouped[(r["model_display"], r["temperature"])].append(r)

    # Performance table
    console.print("[bold]Average performance by model + temperature[/bold]\n")
    table = Table(show_header=True, header_style="bold", box=None)
    table.add_column("Model", style="cyan", min_width=16)
    table.add_column("Temp", justify="right")
    table.add_column("TTFT avg", justify="right")
    table.add_column("Latency avg", justify="right")
    table.add_column("Words/s avg", justify="right")
    table.add_column("Resp length", justify="right")

    for (model, temp), runs in sorted(grouped.items()):
        def avg(k):
            vals = [float(r[k]) for r in runs if r.get(k)]
            return sum(vals) / len(vals) if vals else 0
        table.add_row(
            model, temp,
            f"{avg('time_to_first_token_s'):.2f}s",
            f"{avg('total_latency_s'):.2f}s",
            f"{avg('words_per_sec'):.1f}",
            f"{avg('word_count'):.0f} words",
        )
    console.print(table)

    # Category breakdown
    console.print("\n[bold]Avg latency by category (all models combined)[/bold]\n")
    cat_grouped = defaultdict(list)
    for r in rows:
        if not r.get("error"):
            cat_grouped[r["category"]].append(float(r["total_latency_s"]))

    cat_table = Table(show_header=True, header_style="bold", box=None)
    cat_table.add_column("Category", style="purple")
    cat_table.add_column("Avg latency", justify="right")
    cat_table.add_column("Prompts", justify="right")

    for cat, latencies in sorted(cat_grouped.items()):
        cat_table.add_row(cat, f"{sum(latencies)/len(latencies):.2f}s", str(len(latencies)))
    console.print(cat_table)

    console.print("\n[dim]Tip: open the CSV in Excel/Sheets for custom charts and deeper analysis.[/dim]")


if __name__ == "__main__":
    main()
