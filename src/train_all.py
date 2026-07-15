"""Train every model in the repo with a single command.

Run:  python3 src/train_all.py                    # all models
      python3 src/train_all.py qwen3 gemma4       # only these two
      python3 src/train_all.py --list              # show available models

Each model's own train.py is run as a subprocess in its own directory, so
there are no import conflicts between the identically-named modules (every
folder has its own config.py, model.py, tokenizer.py, etc.).
"""

import subprocess
import sys
import time
import os

# ── Model registry ──────────────────────────────────────────────────────────
# Maps short name → (directory relative to project root, description).
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")

MODELS = {
    "qwen3":    ("qwen3",    "Qwen3 dense (the baseline)"),
    "gemma4":   ("gemma4",   "Gemma-style decoder"),
    "deepseek3":("deepseek3","DeepSeek-V3 sparse (MLA + MoE)"),
    "qwen3_5":  ("qwen3_5",  "Qwen3.5 hybrid (Gated DeltaNet)"),
    "acestep":  ("acestep",  "ACE-Step v1.5 music pipeline"),
}

# Default training order (acestep last — it's the slowest).
DEFAULT_ORDER = ["qwen3", "gemma4", "deepseek3", "qwen3_5", "acestep"]


def print_header(name, desc):
    """Print a visible separator before each model's training run."""
    width = 60
    print()
    print("=" * width)
    print(f"  {name}  —  {desc}")
    print("=" * width)


def train_model(name, desc):
    """Run a model's train.py in its own directory. Returns (name, seconds, ok)."""
    model_dir = os.path.join(PROJECT_ROOT, MODELS[name][0])
    train_script = os.path.join(model_dir, "train.py")

    if not os.path.isfile(train_script):
        print(f"  ⚠  {train_script} not found, skipping.")
        return (name, 0.0, False)

    print_header(name, desc)
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "train.py"],
        cwd=model_dir,
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n  ✗  {name} failed (exit code {result.returncode})")
        return (name, elapsed, False)

    print(f"\n  ✓  {name} done in {elapsed:.1f}s")
    return (name, elapsed, True)


def print_summary(results):
    """Print a final summary table of all training runs."""
    print()
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  {'Model':<12} {'Time':>8}   {'Status'}")
    print(f"  {'─'*12} {'─'*8}   {'─'*8}")
    total = 0.0
    for name, elapsed, ok in results:
        status = "✓ ok" if ok else "✗ fail"
        print(f"  {name:<12} {elapsed:7.1f}s   {status}")
        total += elapsed
    print(f"  {'─'*12} {'─'*8}")
    print(f"  {'total':<12} {total:7.1f}s")
    print()


def main():
    args = sys.argv[1:]

    # --list: show available models and exit.
    if "--list" in args:
        print("Available models:")
        for name in DEFAULT_ORDER:
            _, desc = MODELS[name]
            print(f"  {name:<12}  {desc}")
        return

    # --help
    if "--help" in args or "-h" in args:
        print(__doc__)
        print("Available models:")
        for name in DEFAULT_ORDER:
            _, desc = MODELS[name]
            print(f"  {name:<12}  {desc}")
        return

    # Pick which models to train.
    if args:
        chosen = []
        for arg in args:
            if arg in MODELS:
                chosen.append(arg)
            else:
                print(f"Unknown model: '{arg}'")
                print(f"Available: {', '.join(DEFAULT_ORDER)}")
                sys.exit(1)
    else:
        chosen = DEFAULT_ORDER

    print(f"Training {len(chosen)} model(s): {', '.join(chosen)}")

    # Train each model sequentially.
    results = []
    for name in chosen:
        _, desc = MODELS[name]
        results.append(train_model(name, desc))

    print_summary(results)

    # Exit with failure if any model failed.
    if not all(ok for _, _, ok in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
