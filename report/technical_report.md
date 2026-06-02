# Technical Report — Offline LLM Inference Benchmarks

**Author:** Lucas Lisboa  
**Hardware:** MacBook Pro · 6-Core Intel Core i7 2.6GHz · 16GB RAM · CPU-only inference  
**Stack:** Python 3.12 · Ollama 0.24.0 · llama.cpp backend  
**Repo:** https://github.com/LucasLisboaDev/offline-llm-engineering  

---

## TL;DR

Ran 300+ inference calls across three local SLMs on a consumer Intel Mac. Llama 3.2 3B is the only model suitable for interactive CPU deployment. Phi-4 Mini works at temperature 0 but collapses at 0.7 due to training data leakage. Mistral 7B needs a GPU — it hits a memory bandwidth ceiling on CPU that makes it unusable for anything interactive. All models run 100% offline with zero data leaving the machine.

---

## 1. Why this exists

Cloud LLM APIs are not always an option. Three hard constraints appear repeatedly in production engineering:

**Privacy regulations.** HIPAA prohibits sending patient data to third-party APIs. GDPR restricts cross-border data transfer. SOC 2 often requires data residency. In regulated industries, the model must run where the data lives — full stop.

**Latency budgets.** A round-trip to OpenAI or Anthropic adds 200–600ms before the first token. For real-time applications — voice interfaces, robotics, live document processing — that's a hard blocker. Local inference eliminates the network entirely.

**Edge environments.** Factory floor equipment, medical devices, drones, and offline field systems have no reliable internet. The model must be self-contained on the device.

This project measures what you actually get when you run SLMs locally on consumer hardware — real numbers, real tradeoffs, no cherry-picking.

---

## 2. Setup

### Models

| Model | Parameters | Creator | Disk size | Format |
|---|---|---|---|---|
| llama3.2:3b | 3B | Meta | 2.0 GB | 4-bit GGUF |
| phi4-mini | 3.8B | Microsoft | 2.5 GB | 4-bit GGUF |
| mistral:7b | 7B | Mistral AI | 4.4 GB | 4-bit GGUF |

All models run via Ollama's llama.cpp backend. GGUF 4-bit quantization compresses weights from 32-bit floats to 4-bit integers — roughly 8× size reduction with minimal quality loss on most tasks. Without quantization, Mistral 7B would require ~28GB RAM. With it: 4.4GB.

### Why these three

These models represent three distinct positions on the quality-vs-speed curve. Llama 3.2 3B is Meta's general-purpose small model. Phi-4 Mini is Microsoft's efficiency-focused model trained on curated high-quality data rather than raw scale. Mistral 7B is the quality baseline — the smallest model that consistently competes with much larger models on reasoning benchmarks. Together they let you see whether parameter count, architecture, or training methodology dominates performance on consumer hardware.

### Hardware context

Everything in this report runs on a 2019 MacBook Pro with a 6-Core Intel Core i7 at 2.6GHz and 16GB RAM. No GPU, no Apple Silicon, no NVIDIA CUDA. This is the worst-case scenario for local inference — and it still works. Readers on Apple M-series or NVIDIA hardware should expect 3–10× better throughput numbers.

---

## 3. Methodology

### Metrics

| Metric | Definition | Why it matters |
|---|---|---|
| Time to First Token (TTFT) | Seconds from prompt send to first token received | User-perceived responsiveness — what the user feels |
| Total latency | Wall-clock seconds from prompt to last token | End-to-end cost for batch pipelines |
| Words/sec | Total words ÷ total latency | Throughput proxy (true tok/s ≈ 1.3× this) |
| Memory delta | RSS change during inference via psutil | Per-call memory overhead for capacity planning |

Words/sec is used instead of tokens/sec because the Ollama Python streaming API returns text chunks rather than raw token counts. The conversion factor is approximately 1.3× for English text (average token is ~0.75 words).

### Temperature settings

Every prompt ran at two temperatures:

- **0.0** — deterministic. Argmax token selection. Same input always produces identical output. Used for factual tasks, structured data extraction, reproducible pipelines.
- **0.7** — controlled randomness. Probability distribution rescaled before sampling. More varied and sometimes more creative output, but non-deterministic.

Running both temperatures on every prompt reveals whether a model's behavior changes under randomness — which turns out to be the most important finding in this study.

### Prompt categories

Phase 5 used 30 standardized prompts across 6 categories (5 each): factual recall, reasoning, coding, summarization, creative, and instruction following. This distribution was chosen to expose different model failure modes — factual prompts test knowledge accuracy, instruction prompts test compliance, reasoning prompts test multi-step logic.

---

## 4. Phase 2 — Manual baseline

First manual benchmark. Single prompt across all three models, both temperatures, on a warm Ollama instance.

**Prompt:** *"What is a large language model?"*

### Cross-model comparison (temp=0.7)

| Model | TTFT | Total latency | Words/sec | Params |
|---|---|---|---|---|
| Llama 3.2 3B | 12.69s | 138.79s | 2.4 | 3B |
| Phi-4 Mini | 9.84s | 54.72s | 2.4 | 3.8B |
| Mistral 7B | 19.25s | 145.65s | 0.7 | 7B |

Phi-4 finished 2.5× faster than Llama despite having more parameters. This is the clearest early signal that architecture and training methodology matter more than raw parameter count on this hardware.

### Temperature comparison — Llama 3.2 3B

| Temp | Total latency | Notes |
|---|---|---|
| 0.7 | 138.79s | Non-deterministic |
| 0.0 | 104.87s | Deterministic, 25% faster |

Temperature 0.0 was 25% faster. The mechanism: at T=0.7, the model rescales its full vocabulary probability distribution before each token selection. At T=0.0, this step is skipped entirely (argmax). Small per-token savings compound across a 280-word response.

### Cold start vs warm inference — Llama 3.2 3B

| State | TTFT | Total | Notes |
|---|---|---|---|
| Cold | 8.66s | 12.37s | Model loading from SSD into RAM |
| Warm | 0.41s | 4.08s | Model resident in RAM |
| Speedup | **21×** | 3× | — |

The 8.66s cold TTFT is almost entirely disk I/O — reading the 2GB model file from SSD into RAM. Once resident, TTFT drops to 0.41s. Production implication: pre-warm models on server startup. Never let the first real user request hit a cold model.

### Quality observation

Phi-4 Mini stated that GPT-3 was "developed by Microsoft" at both temp 0.7 and temp 0.0. GPT-3 is OpenAI. This error persisting at temperature 0.0 means it is not random sampling noise — it is a confident wrong belief embedded in the model's weights from training data. A confident wrong answer is more dangerous than a random one because it will always be wrong on this fact regardless of how the prompt is phrased.

---

## 5. Phase 3 — Automated benchmark (60 runs)

10 prompts × 3 models × 2 temperatures = 60 inference calls. Averages are more reliable than single observations — output length variance is the biggest source of noise in single-run benchmarks.

| Model | Temp | Avg TTFT | Avg latency | Words/sec | Mem delta |
|---|---|---|---|---|---|
| Llama 3.2 3B | 0.0 | 7.46s | 119.40s | 1.4 | ~0 MB |
| Llama 3.2 3B | 0.7 | 5.49s | 113.07s | 1.5 | ~0 MB |
| Phi-4 Mini | 0.0 | 6.30s | 101.84s | 1.3 | ~0 MB |
| Phi-4 Mini | 0.7 | 4.73s | 120.14s | 1.5 | ~0 MB |
| Mistral 7B | 0.0 | 12.18s | 206.29s | 0.7 | ~0 MB |
| Mistral 7B | 0.7 | 15.81s | 357.87s | 0.6 | ~0 MB |

**Mistral 7B at T=0.7 averaged 357 seconds.** Nearly 6 minutes per response. At T=0.7 Mistral wrote longer responses — more tokens generated means more weight reads from RAM, and at 7B parameters on CPU that compounds badly. This is not a marginal slowdown — it is a deployment blocker. Mistral 7B at temperature 0.7 is unsuitable for any interactive application on CPU-only hardware.

**Memory delta ≈ 0 across all 60 runs.** Ollama keeps models resident in RAM between calls. No re-loading, no garbage collection overhead per inference. The memory cost is the initial model load — after that, individual inference calls have negligible memory impact.

**Temperature affects models differently.** Llama gets 6% faster at T=0.7 (wrote shorter responses). Phi-4 gets 18% slower at T=0.7 (wrote longer responses). Mistral gets 73% slower at T=0.7 (wrote dramatically longer responses). Temperature selection is model-specific — there is no universal rule.

---

## 6. Phase 4 — Structured outputs and Pydantic validation

Production AI pipelines cannot consume raw LLM text directly. This phase enforces JSON schemas on model outputs and validates them with Pydantic, with a retry-once-then-fail-gracefully mechanism.

### Results

| Input | Model | Schema | Attempts | Success | Confidence | Time |
|---|---|---|---|---|---|---|
| "I love this product" | Llama 3B | sentiment | 1 | ✓ | 0.9 | 45.8s |
| "Worst experience ever" | Llama 3B | sentiment | 1 | ✓ | 0.9 | 16.7s |
| "Okay, nothing special" | Llama 3B | sentiment | 1 | ✓ | 0.5 | 24.3s |
| "Tell me about Paris France" | Llama 3B | entity | 2 | ✗ | — | 102.0s |
| "I loved this" | Mistral 7B | sentiment | 1 | ✓ | 1.0 | 158.4s |
| "I loved this" | Phi-4 Mini | sentiment | 1 | ✓ | 0.9 | 59.9s |

**Confidence calibration works correctly.** Strong sentiment inputs scored 0.9. Ambiguous input ("okay, nothing special") scored 0.5. The model correctly expressed lower certainty on genuinely ambiguous text — this is called calibration and it is exactly the behavior you want in production. Use confidence scores as a routing signal: high confidence responses go straight through, low confidence ones get flagged for human review.

**Entity schema failed due to JSON truncation, not model non-compliance.** The model ran out of token budget mid-response and the JSON was never closed. Root cause: missing `num_predict` limit. Fix: added `num_predict=500` to inference options and tightened the prompt to constrain output length. Passed on attempt 1 after fix. Lesson: token budget management is a required part of production prompt engineering, not an afterthought.

**Mistral returned confidence=1.0 on subjective text.** No NLP model should be 100% certain on subjective language. This indicates Mistral is treating confidence as a label rather than a genuine probability estimate. In production, add a validator that flags confidence=1.0 for review.

**The retry system worked as designed.** Failed on attempt 1, re-prompted with explicit correction instruction, failed again, returned None with error metadata. The pipeline did not crash. Graceful failure in AI pipelines is not optional — a single bad model response must never take down an entire batch job.

---

## 7. Phase 5 — Model comparison study (180 runs)

30 prompts × 3 models × 2 temperatures = 180 inference calls. This is the most statistically reliable dataset in the project. Prompts span 6 categories: factual, reasoning, coding, summarization, creative, instruction following.

### Performance summary

| Model | Temp | Avg TTFT | Avg latency | Words/sec | Avg length |
|---|---|---|---|---|---|
| Llama 3.2 3B | 0.0 | 11.76s | 38.23s | 1.3 | 47 words |
| Llama 3.2 3B | 0.7 | 12.45s | 38.42s | 1.1 | 44 words |
| Phi-4 Mini | 0.0 | 12.65s | 52.07s | 1.2 | 60 words |
| Phi-4 Mini | 0.7 | 24.83s | 120.70s | 0.7 | 75 words |
| Mistral 7B | 0.0 | 23.79s | 71.69s | 0.6 | 41 words |
| Mistral 7B | 0.7 | 34.26s | 106.10s | 0.4 | 40 words |

### Quality findings

**Instruction following — Llama wins.** Prompt: "List exactly 4 programming languages, one per line, nothing else." Llama: exactly 4 items, nothing else. Phi-4: added explanatory text after the list. Mistral: added blank lines between items. For any task where format compliance matters, Llama 3B is the most reliable on this hardware.

**Reasoning — R02 (bat and ball) was correctly solved by all three.** The answer is $0.05 for the ball. All three models arrived at the right answer through algebraic reasoning. This is the classic cognitive bias question that trips humans — small models handle it correctly at T=0.0.

**Reasoning — R05 (apples) exposed fractional arithmetic weakness.** "I have 3 apples. I give away half, then receive 4 more." Correct answer: 5.5 (or 5 if you assume whole apples only). Llama returned 5.5 (technically correct). Phi-4 returned 6 (rounded up without saying so). Mistral returned 5 (rounded down without saying so). None gave a fully rigorous answer. This is a known weakness of sub-7B models on fractional word problems.

**Creative quality — Mistral's output is noticeably better.** On the haiku prompt at T=0.7, Mistral produced "Midnight's quiet hush / Tangled threads unravel slow / Coffee fuels progress." Llama produced a competent but generic haiku. When output quality matters more than speed and GPU resources are available, Mistral's training investment shows.

**Phi-4 training data leakage at T=0.7.** Multiple prompts triggered Phi-4 to generate instruction-following templates from its training data rather than answering the question. One factual prompt ("What is the capital of Australia?") produced 1,100+ words including content from an entirely different domain. This behavior appears more frequently at T=0.7 — added randomness allows the model to drift toward completing training templates. Mitigation: use T=0.0 for all Phi-4 deployments and add explicit conciseness instructions to every prompt.

---

## 8. Conclusions and deployment recommendations

### On CPU-only hardware (this study's conditions)

**Use Llama 3.2 3B** as your primary model. Fastest, most consistent, best instruction compliance. Temperature barely affects it (38.2s at T=0.0 vs 38.4s at T=0.7). Safe to use at either temperature setting.

**Use Phi-4 Mini at T=0.0 only** for tasks requiring reasoning depth. At T=0.0 it is competitive (52s avg) and produces high-quality structured responses. At T=0.7 it degrades significantly and is prone to training data leakage. Never deploy Phi-4 at T=0.7 on CPU without extensive output validation.

**Do not use Mistral 7B for interactive applications on CPU.** 0.4 words/sec at T=0.7 is below practical threshold for any user-facing system. Reserve Mistral for batch processing pipelines at T=0.0, or wait for GPU hardware.

### On GPU hardware (projected)

Apple M2 Pro or NVIDIA RTX 3080 would change the picture. GPU memory bandwidth is 10–20× higher than CPU RAM bandwidth — the bottleneck that cripples Mistral on CPU effectively disappears. On GPU hardware, the recommendation likely reverses: Mistral 7B becomes the default choice due to its quality advantage, with Llama 3B reserved for highest-throughput requirements.

### Universal recommendations regardless of hardware

**Always pre-warm models before serving real traffic.** Cold start penalty on this hardware: 8–20 seconds TTFT. Warm TTFT: 0.4–2 seconds. The first user request must never absorb the cold start cost.

**Set explicit token limits on every inference call.** Without `num_predict` limits, models can generate arbitrarily long responses. Phi-4's training data leakage can produce 1,000+ word responses to one-sentence factual questions. Token limits are not optional in production.

**Never trust raw model output without validation.** All three models failed instruction compliance on at least one prompt. Pydantic schema validation with retry logic is the minimum bar for any production pipeline consuming structured model output.

**Quality and speed are orthogonal.** The fastest model (Llama 3B) had the best instruction following. The most architecturally efficient model (Phi-4) had a hidden training data defect. The highest quality creative output came from the slowest model (Mistral). You cannot infer quality from performance benchmarks. Test both independently on your specific task distribution.

---

## 9. Reproducing these results

```bash
git clone https://github.com/LucasLisboaDev/offline-llm-engineering
cd offline-llm-engineering
pip install -r requirements.txt

# Pull models once
ollama pull llama3.2:3b && ollama pull phi4-mini && ollama pull mistral:7b

# Verify
python3 verify_setup.py

# Run benchmarks
python3 benchmark.py --quick          # Phase 3 quick test
python3 comparison_study.py --quick   # Phase 5 quick test
python3 comparison_study.py           # Full 180-run study (~3hrs on Intel CPU)
python3 analyze_results.py            # Print analysis
```

Results will differ based on hardware. Document your specs before sharing numbers. The relative rankings between models are likely to hold across hardware — the absolute latency values will not.

---

## 10. What I would do differently

**Use a proper tokenizer for token counting.** Words/sec is a reasonable proxy but true tokens/sec requires tokenizer-level counting. The `tiktoken` library or Ollama's eval_count field in non-streaming responses would give exact numbers.

**Run each prompt 3× and report mean ± std deviation.** Single-run latency has meaningful variance — thermal throttling, OS scheduling, and memory pressure all introduce noise. Three runs per prompt would produce more defensible numbers.

**Test on Apple Silicon for GPU comparison.** The most direct way to demonstrate the CPU vs GPU delta would be running the same benchmark suite on an M2 or M3 Mac. The bandwidth difference would explain the Mistral performance gap concretely rather than projecting it.

**Add a quality scoring rubric.** Manual response review catches issues that latency metrics miss (as Phi-4's hallucination showed), but it doesn't scale. A secondary LLM-as-judge evaluation using a larger model to score response quality would make the quality analysis reproducible and quantitative.