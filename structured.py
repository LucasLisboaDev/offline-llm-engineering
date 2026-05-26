"""
Phase 4 — Structured JSON outputs with Pydantic validation + retry logic.
Enforces a schema on model responses. Retries once on invalid JSON, then fails gracefully.

Usage:
  python structured.py "Analyze the sentiment of: I love this product!"
  python structured.py "Extract info from: John Smith, 34, engineer at Acme Corp"
  python structured.py --schema entity "Tell me about Paris, France"
"""
import json
import re
import time
import typer
import ollama
from enum import Enum
from pydantic import BaseModel, ValidationError, field_validator
from rich.console import Console
from rich.syntax import Syntax
from config import MODELS

app = typer.Typer(add_completion=False)
console = Console()


# ── Pydantic schemas ───────────────────────────────────────────────────────────

class SentimentResult(BaseModel):
    sentiment: str
    confidence: float
    reasoning: str
    keywords: list[str]

    @field_validator("sentiment")
    @classmethod
    def sentiment_must_be_valid(cls, v):
        allowed = {"positive", "negative", "neutral"}
        if v.lower() not in allowed:
            raise ValueError(f"sentiment must be one of {allowed}, got '{v}'")
        return v.lower()

    @field_validator("confidence")
    @classmethod
    def confidence_range(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"confidence must be between 0 and 1, got {v}")
        return v


class EntityResult(BaseModel):
    name: str
    type: str
    attributes: dict[str, str]
    summary: str


SCHEMAS = {
    "sentiment": {
        "model": SentimentResult,
        "system_prompt": """You are a sentiment analysis engine.
You MUST respond with ONLY valid JSON matching this exact schema:
{
  "sentiment": "positive" | "negative" | "neutral",
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<one sentence explanation>",
  "keywords": ["<word1>", "<word2>", ...]
}
Do not include any text outside the JSON object.""",
    },
    "entity": {
        "model": EntityResult,
        "system_prompt": """You are an entity extraction engine.
You MUST respond with ONLY valid JSON matching this exact schema:
{
  "name": "<primary entity name>",
  "type": "<person | place | organization | concept>",
  "attributes": {"<key>": "<value>", ...},
  "summary": "<one sentence description>"
}
Do not include any text outside the JSON object.""",
    },
}


# ── Core extraction logic ──────────────────────────────────────────────────────

def extract_json(text: str) -> dict:
    """Pull JSON from model output — handles markdown fences and stray text."""
    # Try direct parse first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Try extracting from ```json ... ``` fences
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding first { ... } block
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"No valid JSON found in response:\n{text[:200]}")


def run_structured(
    prompt: str,
    schema_name: str = "sentiment",
    model: str = "llama3.2:3b",
    temperature: float = 0.0,
) -> tuple[BaseModel | None, dict]:
    """
    Run structured inference with retry logic.
    Returns (parsed_result, metrics_dict).
    On permanent failure, returns (None, metrics_dict).
    """
    schema_config = SCHEMAS[schema_name]
    PydanticModel = schema_config["model"]
    system_prompt = schema_config["system_prompt"]

    metrics = {
        "model": model,
        "schema": schema_name,
        "attempts": 0,
        "success": False,
        "total_time_s": 0,
    }

    def attempt(messages: list, attempt_num: int):
        start = time.perf_counter()
        response = ollama.chat(
            model=model,
            messages=messages,
            stream=False,
            options={"temperature": temperature}
        )
        elapsed = time.perf_counter() - start
        content = response["message"]["content"]
        return content, elapsed

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    total_start = time.perf_counter()

    # Attempt 1
    metrics["attempts"] = 1
    try:
        content, t1 = attempt(messages, 1)
        raw_json = extract_json(content)
        result = PydanticModel(**raw_json)
        metrics["success"] = True
        metrics["total_time_s"] = round(time.perf_counter() - total_start, 3)
        return result, metrics

    except (ValueError, ValidationError, json.JSONDecodeError) as e:
        console.print(f"[yellow]Attempt 1 failed: {e}. Retrying...[/yellow]")

    # Attempt 2 — re-prompt with explicit correction request
    retry_messages = messages + [
        {"role": "assistant", "content": content},
        {"role": "user", "content": (
            "Your response was not valid JSON. "
            "Please respond with ONLY a JSON object matching the schema. "
            "No explanation, no markdown, just the raw JSON object."
        )},
    ]

    metrics["attempts"] = 2
    try:
        content2, t2 = attempt(retry_messages, 2)
        raw_json = extract_json(content2)
        result = PydanticModel(**raw_json)
        metrics["success"] = True
        metrics["total_time_s"] = round(time.perf_counter() - total_start, 3)
        console.print("[green]Retry succeeded.[/green]")
        return result, metrics

    except (ValueError, ValidationError, json.JSONDecodeError) as e:
        console.print(f"[red]Attempt 2 also failed: {e}. Failing gracefully.[/red]")
        metrics["success"] = False
        metrics["total_time_s"] = round(time.perf_counter() - total_start, 3)
        metrics["error"] = str(e)
        return None, metrics


# ── CLI command ────────────────────────────────────────────────────────────────

class SchemaChoice(str, Enum):
    sentiment = "sentiment"
    entity = "entity"


@app.command()
def extract(
    prompt: str = typer.Argument(..., help="Text to analyze"),
    schema: SchemaChoice = typer.Option(SchemaChoice.sentiment, "--schema", help="Output schema to enforce"),
    model: str = typer.Option("llama3.2:3b", "--model", "-m"),
    temperature: float = typer.Option(0.0, "--temp", "-t"),
):
    """Run structured inference and validate output with Pydantic."""

    console.print(f"\n[bold]Structured extraction[/bold]  "
                  f"[dim]schema={schema.value} · model={model} · temp={temperature}[/dim]\n")

    result, metrics = run_structured(prompt, schema.value, model, temperature)

    if result:
        console.print("[green]Validation passed[/green]\n")
        output = json.dumps(result.model_dump(), indent=2)
        console.print(Syntax(output, "json", theme="monokai", background_color="default"))
    else:
        console.print("[red]Extraction failed after 2 attempts.[/red]")

    console.print(f"\n[dim]Attempts: {metrics['attempts']} · "
                  f"Time: {metrics['total_time_s']}s · "
                  f"Success: {metrics['success']}[/dim]")


if __name__ == "__main__":
    app()
