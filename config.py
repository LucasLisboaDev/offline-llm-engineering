MODELS = {
    "llama3.2:3b": {
        "display_name": "Llama 3.2 3B",
        "creator": "Meta",
        "params_billions": 3,
        "expected_size_gb": 2.0,
    },
    "phi4-mini": {
        "display_name": "Phi-4 Mini",
        "creator": "Microsoft",
        "params_billions": 3.8,
        "expected_size_gb": 2.5,
    },
    "mistral:7b": {
        "display_name": "Mistral 7B",
        "creator": "Mistral AI",
        "params_billions": 7,
        "expected_size_gb": 4.1,
    },
}

OLLAMA_HOST = "http://localhost:11434"
TEMPERATURES = [0.0, 0.7]
RESULTS_DIR = "results"
