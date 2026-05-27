# offline-llm-engineering

Offline LLM inference pipeline built with Ollama. Benchmarks three local small language models — Llama 3.2 3B, Phi-4 Mini, and Mistral 7B — running entirely on CPU with zero cloud dependency.

Built as Portfolio Project 2 for an AI Engineering portfolio. Covers local inference, CLI tooling, inference benchmarking, structured output validation, and model comparison.

---

## Why local inference matters

In production AI engineering, sending data to a cloud API is often not an option:

- **Privacy regulations** — HIPAA (healthcare), GDPR (EU), SOC 2, and financial regulations prohibit sending sensitive data to third-party APIs
- **Latency requirements** — cloud round-trips add 200–800ms before the first token; local inference eliminates network overhead entirely
- **Cost at scale** — per-token billing scales linearly; a local GPU server is a fixed capital cost with a typical 3–6 month breakeven at high volume
- **Edge deployment** — drones, medical devices, factory equipment, and offline environments have no internet access; the model must run where the data lives

---

## Models

| Model | Parameters | Creator | Disk size | Architecture focus |
|---|---|---|---|---|
| llama3.2:3b | 3B | Meta | 2.0 GB | General purpose, fast |
| phi4-mini | 3.8B | Microsoft | 2.5 GB | Reasoning efficiency |
| mistral:7b | 7B | Mistral AI | 4.4 GB | Quality baseline |

---

## Hardware

- **Machine**: Intel Mac (x86_64)
- **Inference**: CPU-only, no GPU acceleration
- **RAM**: System RAM (no dedicated VRAM)
- **Ollama version**: 0.24.0
- **Python**: 3.12

> All benchmark numbers in this repo reflect CPU-only inference on Intel hardware.
> Apple Silicon (M-series) or NVIDIA GPU results will be proportionally faster.

---

## Project structure

```
offline-llm-engineering/
├── cli.py                  # Phase 2: CLI interface — stream responses, measure latency
├── benchmark.py            # Phase 3: automated benchmarking across all models
├── structured.py           # Phase 4: JSON schema enforcement + Pydantic validation
├── comparison_study.py     # Phase 5: 30-prompt standardized model comparison
├── analyze_results.py      # Phase 5: read CSVs, print formatted analysis report
├── verify_setup.py         # Phase 1: confirms Ollama + all models are ready
├── config.py               # shared model definitions and constants
├── requirements.txt        # Python dependencies
├── prompts/
│   ├── benchmark_prompts.json     # 10 prompts for Phase 3
│   └── comparison_prompts.json    # 30 prompts across 6 categories for Phase 5
├── results/                # auto-generated CSVs and JSONs (git-ignored)
└── report/
    └── technical_report.md        # full findings with numbers and analysis
```

---

## Setup

### 1. Install Ollama

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Start Ollama as a background service

```bash
brew services start ollama   # macOS
# or: ollama serve           # manual foreground mode
```

### 3. Pull the three models (one-time download)

```bash
ollama pull llama3.2:3b    # 2.0 GB
ollama pull phi4-mini      # 2.5 GB
ollama pull mistral:7b     # 4.4 GB
```

### 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 5. Verify everything works

```bash
python3 verify_setup.py
```

Expected output:
```
[OK]  llama3.2:3b  (2.0 GB)
[OK]  phi4-mini    (2.5 GB)
[OK]  mistral:7b   (4.4 GB)
Phase 1 COMPLETE — ready to build Phase 2
```

---

## Usage

### Ask a question (Phase 2)

```bash
python3 cli.py "What is a transformer neural network?"
python3 cli.py "Explain Docker" --model mistral:7b
python3 cli.py "Write a haiku about programming" --temp 0.0
python3 cli.py "Your question" --model phi4-mini --temp 0.7 --save
```

Flags:
- `--model` — choose `llama3.2:3b`, `phi4-mini`, or `mistral:7b` (default: llama3.2:3b)
- `--temp` — temperature 0.0 (deterministic) to 1.0 (creative) (default: 0.7)
- `--save` — write result JSON to `results/`

### Run benchmarks (Phase 3)

```bash
python3 benchmark.py                      # all models, 10 prompts
python3 benchmark.py --quick              # 3 prompts, fast validation
python3 benchmark.py --model llama3.2:3b  # single model only
```

### Structured outputs with Pydantic (Phase 4)

```bash
python3 structured.py "I absolutely loved this product!"
python3 structured.py "Tell me about Paris, France" --schema entity
python3 structured.py "This was terrible" --model mistral:7b
```

### Full model comparison study (Phase 5)

```bash
python3 comparison_study.py --quick   # 5 prompts, verify it works
python3 comparison_study.py           # full 30-prompt study
python3 analyze_results.py            # analyze and print latest results
```

---

## Phase 2 — Manual baseline results

First manual benchmark run. Same prompt across all three models, temperature 0.7, Intel Mac CPU.

**Prompt:** *"What is a large language model?"*

| Model | TTFT | Total latency | Words/sec | Params |
|---|---|---|---|---|
| Llama 3.2 3B | 12.69s | 138.79s | 2.4 | 3B |
| Phi-4 Mini | 9.84s | 54.72s | 2.4 | 3.8B |
| Mistral 7B | 19.25s | 145.65s | 0.7 | 7B |

**Temperature comparison — Llama 3.2 3B, same prompt:**

| Temperature | Total latency | Behavior |
|---|---|---|
| 0.7 | 138.79s | Non-deterministic, varied output |
| 0.0 | 104.87s | Deterministic, 25% faster |

**Cold vs warm inference — Llama 3.2 3B:**

| State | TTFT | Total |
|---|---|---|
| Cold (first run) | 8.66s | 12.37s |
| Warm (model in RAM) | 0.41s | 4.08s |
| Speedup | **21× faster** | 3× faster |

**Key observations:**
- Phi-4 Mini finished 2.5× faster than Llama despite having more parameters — architecture efficiency matters more than raw parameter count
- Mistral 7B dropped to 0.7 words/sec on CPU, hitting memory bandwidth limits — same model on GPU would close this gap significantly
- Phi-4 hallucinated "GPT-3 developed by Microsoft" at both temp 0.7 and temp 0.0 — a training data error that temperature cannot fix
- Cold start cost is 21× higher than warm inference — production systems must pre-warm models before serving real traffic

Full automated benchmark results: see `report/technical_report.md`

---

## Concepts covered

| Concept | Where it appears |
|---|---|
| Local vs cloud inference | Architecture, setup |
| Model quantization (GGUF/4-bit) | Model files, why 7B fits in 4GB |
| Time to first token (TTFT) | CLI metrics, benchmark output |
| Cold start vs warm inference | Phase 2 manual testing |
| Temperature and determinism | CLI --temp flag, comparison runs |
| Tokens/sec and memory bandwidth | Phase 3 benchmarks |
| Structured output + Pydantic validation | Phase 4 |
| Model comparison methodology | Phase 5 |

---

## Technical report

See [`report/technical_report.md`](report/technical_report.md) for full benchmark tables, model quality analysis, and quality-vs-speed tradeoff conclusions.
