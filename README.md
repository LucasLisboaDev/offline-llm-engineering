# Local AI Assistant — Portfolio Project 2

Offline inference pipeline using Ollama. Benchmarks three local SLMs with zero cloud dependency.

## Hardware (fill this in)

- **CPU**: 
- **RAM**: 
- **GPU**: 
- **OS**: 
- **Ollama version**: run `ollama --version`
-Hardware: Intel Mac, CPU-only inference, no GPU acceleration 

## Models

| Model | Params | Creator | Size |
|-------|--------|---------|------|
| llama3.2:3b | 3B | Meta | ~2.0 GB |
| phi4-mini | 3.8B | Microsoft | ~2.5 GB |
| mistral:7b | 7B | Mistral AI | ~4.1 GB |

## Setup

```bash
# 1. Install Ollama
brew install ollama          # macOS
# or: curl -fsSL https://ollama.com/install.sh | sh   (Linux)

# 2. Start the server (keep this terminal open)
ollama serve

# 3. Pull models (one-time download, then fully offline)
ollama pull llama3.2:3b
ollama pull phi4-mini
ollama pull mistral:7b

# 4. Install Python dependencies
pip install -r requirements.txt

# 5. Verify everything works
python verify_setup.py
```

## Usage

### Phase 2 — CLI
```bash
python cli.py "What is a transformer neural network?"
python cli.py "Explain Docker" --model mistral:7b
python cli.py "Write a haiku" --model phi4-mini --temp 0.0
python cli.py "Your question" --save   # saves JSON to results/
```

### Phase 3 — Benchmarking
```bash
python benchmark.py                      # all models, 10 prompts each
python benchmark.py --quick              # 3 prompts, fast test
python benchmark.py --model llama3.2:3b  # single model
```

### Phase 4 — Structured outputs
```bash
python structured.py "I absolutely loved this restaurant!"
python structured.py "Tell me about Paris, France" --schema entity
python structured.py "This product is terrible" --model mistral:7b
```

### Phase 5 — Model comparison study
```bash
python comparison_study.py --quick      # 5 prompts, verify it works
python comparison_study.py              # full 30-prompt study
python analyze_results.py               # analyze latest results
```

## Project structure

```
local-ai-assistant/
├── config.py               # model definitions, shared config
├── cli.py                  # Phase 2: CLI interface
├── benchmark.py            # Phase 3: inference benchmarking
├── structured.py           # Phase 4: Pydantic + retry logic
├── comparison_study.py     # Phase 5: 30-prompt comparison
├── analyze_results.py      # Phase 5: results analysis
├── verify_setup.py         # Phase 1: setup verification
├── requirements.txt
├── prompts/                # prompt JSON files
├── results/                # auto-generated CSVs and JSONs
└── report/                 # technical report (write after runs)
```

## Why local inference matters

- **Privacy**: HIPAA, GDPR, SOC 2 — regulated industries can't send data to cloud APIs
- **Latency**: No network round-trip (200–800ms saved per request)
- **Cost**: Fixed hardware cost vs per-token billing at scale
- **Edge**: Works offline — drones, medical devices, factory floor
