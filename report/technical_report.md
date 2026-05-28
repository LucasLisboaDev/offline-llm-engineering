# Technical Report — Offline LLM Inference Benchmarks

**Project:** offline-llm-engineering  
**Hardware:** Intel Mac, CPU-only, no GPU acceleration  
**Ollama version:** 0.24.0  
**Python:** 3.12  
**Date:** May 2026  

---

## 1. Objective

This report documents the inference performance, output quality, and quality-vs-speed tradeoffs of three small language models (SLMs) running entirely offline on consumer hardware. The goal is to produce empirical data that informs real deployment decisions in privacy-sensitive, latency-constrained, or cost-constrained environments.

---

## 2. Why local inference

Sending data to a cloud LLM API is not always possible. Three hard constraints appear repeatedly in production:

**Privacy regulations** — HIPAA prohibits sending patient data to third-party APIs. GDPR restricts cross-border data transfer. SOC 2 compliance often requires data residency. In all these cases, the model must run where the data lives.

**Latency budgets** — A cloud API call adds a minimum 200–400ms network round-trip before the first token arrives. Applications with real-time requirements — voice assistants, robotics, live document processing — cannot absorb this cost.

**Edge environments** — Factory floor equipment, medical devices, drones, and submarines have no reliable internet connection. The model must be self-contained on the device.

Local inference with quantized models on consumer hardware is now a viable solution for all three constraints.

---

## 3. Models evaluated

| Model | Parameters | Creator | Disk size | Quantization |
|---|---|---|---|---|
| llama3.2:3b | 3 billion | Meta | 2.0 GB | 4-bit GGUF |
| phi4-mini | 3.8 billion | Microsoft | 2.5 GB | 4-bit GGUF |
| mistral:7b | 7 billion | Mistral AI | 4.4 GB | 4-bit GGUF |

### Why these three

These models represent three distinct points on the quality-vs-speed curve. Llama 3.2 3B is Meta's general-purpose small model optimized for broad tasks. Phi-4 Mini is Microsoft's efficiency-focused model trained on curated high-quality data rather than raw scale — designed to punch above its parameter count. Mistral 7B is the quality baseline — the smallest model that consistently competes with much larger models on reasoning benchmarks.

### What GGUF quantization means

All three models are stored in GGUF format at 4-bit quantization. A model's weights are originally stored as 32-bit floating point numbers — 4 bytes per parameter. At 7 billion parameters, Mistral would require ~28 GB at full precision. 4-bit quantization converts each weight to a 4-bit integer, reducing memory by approximately 8× with minimal quality degradation on most tasks. This is what makes running 7B parameter models on a laptop feasible.

---

## 4. Methodology

### Infrastructure

- **Runtime:** Ollama 0.24.0 with llama.cpp backend
- **Interface:** Custom Python CLI using the official `ollama` Python client
- **Measurement:** `time.perf_counter()` for wall-clock timing, `psutil` for memory delta
- **Streaming:** All runs use streaming mode — tokens are measured as they arrive

### Metrics captured per run

| Metric | Definition | Why it matters |
|---|---|---|
| Time to First Token (TTFT) | Seconds from prompt send to first token received | User-perceived responsiveness |
| Total latency | Seconds from prompt send to last token | End-to-end response time |
| Words per second | Total words ÷ total latency | Throughput proxy (see note below) |
| Memory delta | RAM change during inference (psutil) | Deployment memory budgeting |

> **Note on words/sec vs tokens/sec:** The Ollama Python streaming API returns text chunks, not raw token counts. Words/sec is used as a consistent proxy. True tokens/sec would be approximately 1.3× higher (average English token is ~0.75 words). All comparisons in this report use words/sec for consistency.

### Temperature settings

Every prompt was run at two temperatures:

- **Temperature 0.0** — deterministic. The model always selects the highest-probability next token. Same prompt always produces identical output. Used for factual tasks, structured output, and reproducible benchmarks.
- **Temperature 0.7** — controlled randomness. The probability distribution is rescaled before sampling, allowing non-peak tokens to be selected. More creative and varied output, but non-deterministic.

---

## 5. Phase 2 results — manual baseline

**Prompt:** *"What is a large language model?"*  
**Condition:** Temperature 0.7, single run per model, Ollama warm after first load

### 5.1 Cross-model comparison

| Model | TTFT | Total latency | Words/sec | Params |
|---|---|---|---|---|
| Llama 3.2 3B | 12.69s | 138.79s | 2.4 | 3B |
| Phi-4 Mini | 9.84s | 54.72s | 2.4 | 3.8B |
| Mistral 7B | 19.25s | 145.65s | 0.7 | 7B |

**Finding 1 — Phi-4 efficiency anomaly:** Phi-4 Mini completed in 54.72s despite having 27% more parameters than Llama 3.2 3B (3.8B vs 3B). This is the clearest demonstration that parameter count does not determine inference speed. Microsoft trained Phi-4 on a curated dataset with an emphasis on reasoning efficiency — the result is a model whose computational graph is cheaper to evaluate per token than a larger raw dataset model. Architecture and training methodology matter more than scale.

**Finding 2 — Mistral 7B memory bandwidth ceiling:** Mistral 7B dropped to 0.7 words/sec — 3.4× slower than the other two models. On CPU-only inference, the bottleneck is not computation but memory bandwidth. The CPU must read model weights from RAM for every token generated. At 7B parameters, there are approximately 2.3× more weights to read per forward pass than at 3B parameters. Consumer CPU RAM bandwidth (~50 GB/s) cannot sustain the read throughput required for fast 7B inference. On a GPU with 600–900 GB/s memory bandwidth, this gap narrows dramatically.

### 5.2 Temperature comparison — Llama 3.2 3B

| Temperature | Total latency | Words/sec | Output character |
|---|---|---|---|
| 0.7 | 138.79s | 2.4 | Varied, slightly longer |
| 0.0 | 104.87s | 2.6 | Deterministic, 25% faster |

Temperature 0.0 was 25% faster on total latency. The mechanism: at temp 0.7, the model rescales its full vocabulary probability distribution before each token selection — a non-trivial floating point operation multiplied across every token in a ~280 word response. At temp 0.0, this step is skipped (argmax is used directly). Small per-token cost, significant aggregate savings on long outputs.

### 5.3 Cold start vs warm inference — Llama 3.2 3B

| State | TTFT | Total latency | Notes |
|---|---|---|---|
| Cold (first run) | 8.66s | 12.37s | Model loaded from SSD into RAM |
| Warm (subsequent run) | 0.41s | 4.08s | Model resident in RAM |
| Speedup | **21× faster TTFT** | 3× faster total | — |

The 8.66s cold TTFT is almost entirely disk I/O — reading the 2.0GB model file from SSD into RAM. Once resident, TTFT drops to 0.41s. This 21× difference has direct production implications: any deployment must pre-warm models on startup before serving real traffic, or the first user of each session absorbs a 8–20 second penalty.

---

## 6. Quality observations — Phase 2

Raw performance metrics do not capture output quality. Manual review of responses revealed:

**Phi-4 Mini hallucination (confirmed at both temp 0.0 and temp 0.7):**  
Phi-4 stated that GPT-3 was "developed by Microsoft." GPT-3 was developed by OpenAI. Critically, this error persisted at temperature 0.0 — meaning it is not random variation but a confident wrong belief embedded in the model's weights from training data. This is more dangerous than a hallucination that varies with temperature, because it will always be wrong on this fact regardless of how the prompt is phrased.

**Llama 3.2 3B minor inaccuracy (temp 0.7 only):**  
At temperature 0.7, Llama described training as "masked language modeling" — which is BERT's technique. GPT-style models like Llama use causal language modeling (predict the next token). This error did not appear at temperature 0.0, confirming it was temperature-induced sampling noise rather than a fixed weight error.

**Mistral 7B — most accurate, most concise:**  
Despite being slowest on this hardware, Mistral 7B produced the most factually accurate and concisely written response. This is consistent with its reputation as a quality-optimized model.

**Conclusion:** Speed metrics and quality metrics are orthogonal. The fastest model (Phi-4) had the worst factual accuracy on this prompt. The slowest model (Mistral 7B) had the best. Model selection cannot be made on performance benchmarks alone.

---

## 7. Phases 3–5 results

> **Status:** In progress. This section will be updated as benchmark runs complete.

### 7.1 Automated benchmark — Phase 3 results

10 prompts × 3 models × 2 temperatures = 60 total runs

| Model | Temp | Avg TTFT | Avg latency | Words/sec | Mem delta |
|---|---|---|---|---|---|
| Llama 3.2 3B | 0.0 | 7.46s | 119.40s | 1.4 | ~0 MB |
| Llama 3.2 3B | 0.7 | 5.49s | 113.07s | 1.5 | ~0 MB |
| Phi-4 Mini | 0.0 | 6.30s | 101.84s | 1.3 | ~0 MB |
| Phi-4 Mini | 0.7 | 4.73s | 120.14s | 1.5 | ~0 MB |
| Mistral 7B | 0.0 | 12.18s | 206.29s | 0.7 | ~0 MB |
| Mistral 7B | 0.7 | 15.81s | 357.87s | 0.6 | ~0 MB |

Key findings:
- Mistral 7B at T=0.7 averaged 357s — unsuitable for interactive CPU deployment
- Phi-4 Mini achieved best TTFT at 4.73s — most responsive for streaming use cases  
- Temperature effect on latency is model-specific — Llama gets faster at T=0.7, Mistral gets 73% slower
- Memory delta ≈ 0 across all runs — model resident in RAM, no per-inference allocation cost

### 7.2 Structured output validation — Phase 4
*To be completed: Pydantic schema enforcement, retry success rates per model*

### 7.3 Model comparison study — Phase 5
*To be completed: 30 prompts × 3 models × 2 temperatures = 180 runs*

---

## 8. Preliminary conclusions

Based on Phase 2 manual testing:

**Use Llama 3.2 3B when:** speed and low memory footprint are the priority. Best for conversational applications and simple Q&A on CPU hardware.

**Use Phi-4 Mini when:** you need faster total response time and the task involves reasoning. Verify factual accuracy on your specific domain before deploying — training data errors exist.

**Use Mistral 7B when:** output quality is the priority and you have GPU acceleration available. On CPU-only hardware, the memory bandwidth ceiling makes it impractical for interactive use cases.

**For all models:** always pre-warm before serving real traffic. Cold start penalty (8–20s) is unacceptable in any user-facing application.

---

## 9. Reproducing these results

```bash
# Clone and setup
git clone https://github.com/LucasLisboaDev/offline-llm-engineering
cd offline-llm-engineering
pip install -r requirements.txt

# Pull models (one-time)
ollama pull llama3.2:3b && ollama pull phi4-mini && ollama pull mistral:7b

# Verify
python3 verify_setup.py

# Run manual baseline
python3 cli.py "What is a large language model?" --model llama3.2:3b
python3 cli.py "What is a large language model?" --model phi4-mini
python3 cli.py "What is a large language model?" --model mistral:7b

# Run automated benchmark
python3 benchmark.py --quick
```

Results will vary based on hardware. Document your hardware specs in README.md before sharing results.
