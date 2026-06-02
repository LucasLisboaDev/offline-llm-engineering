# offline-llm-engineering

Offline LLM inference pipeline. Benchmarks three small language models — Llama 3.2 3B, Phi-4 Mini, and Mistral 7B — running entirely on a consumer Intel Mac with zero cloud dependency.

**300+ inference calls. Real numbers. No cherry-picking.**

→ Full analysis: [`report/technical_report.md`](report/technical_report.md)

---

## What this is

A complete local AI engineering project covering five phases:

- **CLI tool** — stream responses from any of the three models, measure TTFT and latency in real time
- **Inference benchmarking** — automated 60-run benchmark across all models and temperatures, exported to CSV
- **Structured outputs** — JSON schema enforcement with Pydantic validation and retry logic
- **Model comparison study** — 180 runs across 30 standardized prompts in 6 categories, documented with quality analysis

Everything runs offline. No API keys. No data leaves the machine.

---

## Why local inference matters

Cloud LLM APIs are not always an option in production:

**Privacy** — HIPAA, GDPR, and SOC 2 prohibit sending sensitive data to third-party APIs. In healthcare, finance, and legal, the model must run where the data lives.

**Latency** — A round-trip to OpenAI adds 200–600ms before the first token. For real-time applications, that is a hard blocker. Local inference eliminates the network entirely.

**Cost at scale** — Per-token billing scales linearly with volume. A local GPU server is a fixed capital cost. Breakeven is typically 3–6 months at high request volume.

**Edge deployment** — Factory equipment, medical devices, and field systems have no reliable internet. The model must be self-contained on the device.

---

## Results

### Phase 5 — Full comparison study (180 runs)
30 prompts × 3 models × 2 temperatures · Intel MacBook Pro · 6-Core i7 · CPU-only

| Model | Temp | Avg TTFT | Avg latency | Words/sec | Verdict |
|---|---|---|---|---|---|
| Llama 3.2 3B | 0.0 | 11.76s | 38.23s | 1.3 | ✅ Best for CPU |
| Llama 3.2 3B | 0.7 | 12.45s | 38.42s | 1.1 | ✅ Consistent |
| Phi-4 Mini | 0.0 | 12.65s | 52.07s | 1.2 | ✅ Good at T=0 |
| Phi-4 Mini | 0.7 | 24.83s | 120.70s | 0.7 | ⚠️ Degrades at T=0.7 |
| Mistral 7B | 0.0 | 23.79s | 71.69s | 0.6 | ⚠️ Slow, needs GPU |
| Mistral 7B | 0.7 | 34.26s | 106.10s | 0.4 | ❌ Unusable on CPU |

### Phase 3 — Automated benchmark (60 runs)
10 prompts × 3 models × 2 temperatures

| Model | Temp | Avg TTFT | Avg latency | Words/sec |
|---|---|---|---|---|
| Llama 3.2 3B | 0.0 | 7.46s | 119.40s | 1.4 |
| Llama 3.2 3B | 0.7 | 5.49s | 113.07s | 1.5 |
| Phi-4 Mini | 0.0 | 6.30s | 101.84s | 1.3 |
| Phi-4 Mini | 0.7 | 4.73s | 120.14s | 1.5 |
| Mistral 7B | 0.0 | 12.18s | 206.29s | 0.7 |
| Mistral 7B | 0.7 | 15.81s | 357.87s | 0.6 |

### Phase 2 — Cold vs warm inference (Llama 3.2 3B)

| State | TTFT | Total latency |
|---|---|---|
| Cold (model loading from disk) | 8.66s | 12.37s |
| Warm (model resident in RAM) | 0.41s | 4.08s |
| **Speedup** | **21×** | **3×** |

---

## Key findings

**Llama 3.2 3B is the only model suitable for interactive CPU deployment.** 38s average latency at both temperatures, best instruction compliance, predictable behavior regardless of temperature setting.

**Phi-4 Mini has a temperature cliff.** At T=0.0 it is competitive. At T=0.7 latency triples and training data leakage appears — the model completes instruction-following templates from its training data instead of answering the question. One factual prompt triggered 1,100+ words of irrelevant content. Always use T=0.0 with explicit conciseness instructions.

**Mistral 7B needs a GPU.** 0.4 words/sec at T=0.7 on CPU. The 7B parameter count creates a memory bandwidth bottleneck that CPU RAM cannot sustain. On a GPU (10–20× higher bandwidth), this gap closes significantly.

**Cold start costs 21× more than warm inference.** Production systems must pre-warm models on startup. Never let the first real user request hit a cold model.

**Quality and speed are orthogonal.** The fastest model had the best instruction compliance. The most architecturally efficient model had a training data defect. The highest quality creative output came from the slowest model. Speed benchmarks alone cannot substitute for quality evaluation.

---

## Models

| Model | Parameters | Creator | Disk size | Notes |
|---|---|---|---|---|
| llama3.2:3b | 3B | Meta | 2.0 GB | Best overall for CPU |
| phi4-mini | 3.8B | Microsoft | 2.5 GB | Use T=0.0 only |
| mistral:7b | 7B | Mistral AI | 4.4 GB | Needs GPU for interactive use |

All models are 4-bit GGUF quantized. Without quantization Mistral 7B would require ~28GB RAM. Quantization makes local inference practical on consumer hardware.

---

## Hardware

- **Machine**: MacBook Pro · 6-Core Intel Core i7 · 2.6GHz
- **RAM**: 16GB · CPU inference only · no GPU acceleration
- **Ollama**: 0.24.0 · llama.cpp backend
- **Python**: 3.12

All numbers are CPU-only. Apple Silicon (M-series) or NVIDIA GPU hardware will be 3–10× faster on throughput.

---

## Setup

```bash
# 1. Install Ollama
brew install ollama          # macOS
curl -fsSL https://ollama.com/install.sh | sh   # Linux

# 2. Start as background service
brew services start ollama

# 3. Pull models (one-time, ~9GB total)
ollama pull llama3.2:3b
ollama pull phi4-mini
ollama pull mistral:7b

# 4. Install Python dependencies
pip install -r requirements.txt

# 5. Verify everything works
python3 verify_setup.py
```

---

## Usage

### CLI — ask a question (Phase 2)

```bash
python3 cli.py "What is a transformer neural network?"
python3 cli.py "Explain Docker" --model mistral:7b
python3 cli.py "Write a haiku" --temp 0.0
python3 cli.py "Your question" --model phi4-mini --save
```

### Benchmark (Phase 3)

```bash
python3 benchmark.py            # 60 runs, all models
python3 benchmark.py --quick    # 18 runs, fast test
```

### Structured outputs with Pydantic (Phase 4)

```bash
python3 structured.py "I loved this product"
python3 structured.py "Tell me about Paris France" --schema entity
python3 structured.py "This was terrible" --model mistral:7b
```

### Model comparison study (Phase 5)

```bash
python3 comparison_study.py --quick   # 30 runs, verify it works
python3 comparison_study.py           # 180 runs, full study
python3 analyze_results.py            # print analysis of latest results
```

---

## Project structure

```
offline-llm-engineering/
├── cli.py                   # Phase 2: CLI, streaming, latency measurement
├── benchmark.py             # Phase 3: automated benchmark, CSV export
├── structured.py            # Phase 4: Pydantic validation, retry logic
├── comparison_study.py      # Phase 5: 30-prompt comparison study
├── analyze_results.py       # Phase 5: results analysis and reporting
├── verify_setup.py          # Phase 1: setup verification
├── config.py                # shared model config
├── requirements.txt
├── prompts/
│   ├── benchmark_prompts.json
│   └── comparison_prompts.json
├── results/                 # auto-generated (git-ignored)
└── report/
    └── technical_report.md  # full findings, all phases
```

---

## Concepts demonstrated

| Concept | Where |
|---|---|
| Local vs cloud inference tradeoffs | Architecture, Phase 1 |
| GGUF quantization | Model setup, report section 2 |
| Time to first token (TTFT) | CLI output, Phase 2–3 |
| Cold start vs warm inference | Phase 2, 21× speedup measured |
| Temperature and determinism | Phase 2 comparison, Phase 5 |
| Memory bandwidth bottleneck | Phase 3 Mistral analysis |
| Pydantic schema validation | Phase 4 |
| Retry logic and graceful failure | Phase 4 |
| Training data leakage | Phase 5 Phi-4 finding |
| Confidence calibration | Phase 4 sentiment analysis |
| Prompt injection defense | comparison_study.py CONCISE_SUFFIX |