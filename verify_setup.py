"""
Phase 1 verification script.
Run this to confirm Ollama is running and all three models are available.
Usage: python verify_setup.py
"""
import sys
import time

def verify():
    print("=" * 50)
    print("  Local AI Assistant — Phase 1 Verification")
    print("=" * 50)

    # Check ollama is importable
    try:
        import ollama
    except ImportError:
        print("\nERROR: 'ollama' package not installed.")
        print("Fix: pip install -r requirements.txt")
        sys.exit(1)

    # Check server is running
    print("\n[1/3] Checking Ollama server connection...")
    try:
        models_response = ollama.list()
        available = [m["model"] for m in models_response["models"]]
        print(f"      Connected to Ollama. Models on disk: {len(available)}")
    except Exception as e:
        print(f"\nERROR: Cannot connect to Ollama server.")
        print(f"Fix:   Open a terminal and run:  ollama serve")
        print(f"Detail: {e}")
        sys.exit(1)

    # Check all three required models are present
    print("\n[2/3] Checking required models...")
    required = ["llama3.2:3b", "phi4-mini", "mistral:7b"]
    missing = []

    for model in required:
        found = any(model in m for m in available)
        status = "OK" if found else "MISSING"
        size_info = ""
        if found:
            for m in models_response["models"]:
                if model in m["model"]:
                    size_info = f"  ({m['size'] / 1e9:.1f} GB)"
        print(f"      {'[OK]' if found else '[!!]'}  {model}{size_info}")
        if not found:
            missing.append(model)

    if missing:
        print(f"\nERROR: {len(missing)} model(s) not found.")
        print("Fix:   Run these commands:")
        for m in missing:
            print(f"         ollama pull {m}")
        sys.exit(1)

    # Quick inference test
    print("\n[3/3] Running quick inference test (llama3.2:3b)...")
    try:
        start = time.perf_counter()
        response = ollama.chat(
            model="llama3.2:3b",
            messages=[{"role": "user", "content": "Reply with exactly three words: SETUP IS OK"}],
            options={"temperature": 0}
        )
        elapsed = time.perf_counter() - start
        content = response["message"]["content"].strip()
        print(f"      Response : {content}")
        print(f"      Latency  : {elapsed:.2f}s")
    except Exception as e:
        print(f"\nERROR: Inference test failed: {e}")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("  Phase 1 COMPLETE — ready to build Phase 2")
    print("=" * 50)
    print("\nNext step: open cli.py and run:")
    print("  python cli.py \"What is a large language model?\"")

if __name__ == "__main__":
    verify()
