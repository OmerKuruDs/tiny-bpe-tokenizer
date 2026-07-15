# 🔤 Tiny BPE Tokenizer — Modern LLM Architectures from Scratch

Five modern transformer architectures re-implemented from scratch in pure
PyTorch, each small enough to **train on a CPU in under a minute** and
**read end-to-end in an afternoon**. Every model learns the same task:
generating realistic Turkish names from a ~7 700-word corpus.

The goal is not performance — it is **understanding**. Each folder is a
self-contained, dependency-free implementation of a real production
architecture, shrunk to toy size so you can trace every tensor.

## Models at a glance

| Folder | Architecture | Params | Layers | Attention | MLP | What makes it different |
|--------|-------------|--------|--------|-----------|-----|------------------------|
| `qwen3/` | Qwen3 dense | 20 000 | 2 | GQA + QK-Norm + RoPE | SwiGLU | The clean baseline — every other model is a variation of this |
| `gemma4/` | Gemma 4 | 65 824 | 6 | Local/global sliding window | GeGLU | Sandwich norm, per-layer embeddings, 5:1 local-to-global ratio |
| `deepseek3/` | DeepSeek-V3 | ~50 000 | 3 | MLA (multi-head latent) | MoE (4 experts + 1 shared) | KV compressed into a tiny latent; aux-loss-free load balancing |
| `qwen3_5/` | Qwen3.5 hybrid | 42 408 | 4 | Gated DeltaNet + full attn | SwiGLU | 3 cheap linear-attention layers per 1 full-attention layer |
| `acestep/` | ACE-Step v1.5 | 570 084 | — | Self + cross attention (DiT) | — | 4-stage music pipeline: Planner → FSQ → DiT → VAE |

Plus a **`lora/`** folder that bolts adapters (LoRA, rsLoRA, DoRA, VeRA, PiSSA)
onto any of the four text models without touching their weights.

## Quick start

```bash
# 1. Clone and set up
git clone https://github.com/OmerKuruDs/tiny-bpe-tokenizer.git
cd tiny-bpe-tokenizer
uv venv && source .venv/bin/activate
uv pip install torch

# 2. Train a single model
cd qwen3
python3 train.py          # ~30s on CPU → tiny_qwen.pt
python3 generate.py 20    # generate 20 names

# 3. Or train ALL models at once
cd ..
python3 src/train_all.py  # trains qwen3, gemma4, deepseek3, qwen3_5, acestep
```

## Project structure

```
tiny-bpe-tokenizer/
├── config.ini              # shared config (data file path)
├── data/
│   ├── cleaned_words.txt   # ~7 700 Turkish words/names
│   ├── cleaning_the_words.py
│   └── words.csv           # raw source data
│
├── qwen3/                  # Qwen3 dense (the baseline)
│   ├── config.py           # ModelConfig dataclass
│   ├── rms_norm.py         # RMSNorm
│   ├── rotary.py           # RoPE (Rotary Position Embedding)
│   ├── attention.py        # GQA + QK-Norm
│   ├── mlp.py              # SwiGLU feed-forward
│   ├── block.py            # TransformerBlock (pre-norm)
│   ├── model.py            # TinyQwen (full model + generate)
│   ├── tokenizer.py        # BPETokenizer (Byte Pair Encoding)
│   ├── train.py            # training loop
│   └── generate.py         # inference script
│
├── gemma4/                 # Gemma-style decoder
│   └── ...                 # sandwich norm, sliding window, GeGLU, per-layer embeds
│
├── deepseek3/              # DeepSeek-V3 sparse
│   ├── mla.py              # Multi-head Latent Attention
│   ├── moe.py              # Mixture of Experts + aux-loss-free balancing
│   └── ...
│
├── qwen3_5/                # Qwen3.5 hybrid
│   ├── gated_deltanet.py   # Gated DeltaNet (linear attention)
│   └── ...
│
├── acestep/                # ACE-Step v1.5 music pipeline
│   ├── vae.py              # Oobleck VAE (waveform ↔ latent)
│   ├── fsq.py              # Finite Scalar Quantization bridge
│   ├── planner.py          # Autoregressive 5Hz code planner
│   ├── dit.py              # Diffusion Transformer (flow matching)
│   └── ...
│
├── lora/                   # LoRA and variants (architecture-agnostic)
│   ├── lora.py             # LoRA / rsLoRA config & layers
│   ├── dora.py             # DoRA (Weight-Decomposed LoRA)
│   ├── vera.py             # VeRA (Vector-based Random Matrix Adaptation)
│   ├── pissa.py            # PiSSA (Principal Singular values Adaptation)
│   ├── inject.py           # inject/merge/save adapter helpers
│   ├── pretrain.py         # pretrain a base model for adaptation
│   ├── train.py            # train an adapter on one letter
│   └── by_hand.py          # pencil-and-paper verification
│
├── src/
│   └── train_all.py        # train every model with one command
│
└── demo.ipynb              # interactive notebook demo
```

## Architecture highlights

### Qwen3 — the baseline

The clean modern decoder. Pre-norm blocks with RMSNorm, Grouped Query
Attention (GQA) with QK-Norm + RoPE, SwiGLU MLP, tied embeddings.

**Read order:** `config.py` → `rms_norm.py` → `rotary.py` → `attention.py` →
`mlp.py` → `block.py` → `model.py`

### Gemma 4 — local & global attention

Gemma's distinctive choices: **sandwich norm** (RMSNorm before and after each
sub-layer), **sliding-window** attention for local layers (5 local : 1 global),
**GeGLU** instead of SwiGLU, input embedding scaled by √hidden_size, and
**per-layer embeddings**.

### DeepSeek-V3 — MLA + MoE

Two signature pieces replace the Qwen3 baseline:

- **MLA (Multi-head Latent Attention):** K/V are compressed into a tiny
  KV latent and re-expanded — only the latent ever needs caching.
- **MoE (Mixture of Experts):** 4 small expert MLPs + 1 shared expert,
  top-2 routing with **aux-loss-free** load balancing via per-expert bias.

### Qwen3.5 — hybrid linear + full attention

Most layers use **Gated DeltaNet** (linear attention with the gated delta rule),
keeping a fixed-size memory matrix updated token by token. Every 4th layer is
full softmax attention for global context.

### ACE-Step v1.5 — the music pipeline

Four models in series: a **Planner** (autoregressive LM) writes 5 Hz codes, an
**FSQ bridge** lifts them to 25 Hz continuous latents, a **DiT** denoises noise
into the clean latent via flow matching, and a **VAE** decodes to audio.

### LoRA and friends

The `lora/` folder is architecture-agnostic — it imports any of the four text
models, freezes their weights, and trains small adapter layers. Five methods
are implemented: **LoRA**, **rsLoRA**, **DoRA**, **VeRA**, and **PiSSA**.

```bash
cd lora
python3 pretrain.py              # make base_qwen3.pt
python3 train.py                 # LoRA on qwen3, letter 'z'
python3 train.py gemma4 dora     # DoRA on gemma4
python3 train.py deepseek3 lora s  # LoRA on deepseek3, letter 's'
```

## Dataset

The training corpus is **~7 700 cleaned Turkish words** in
`data/cleaned_words.txt`, sourced from `words.csv` and cleaned with
`cleaning_the_words.py`. Each model uses a **Byte Pair Encoding (BPE)** tokenizer
trained from scratch with a vocabulary size of 128 tokens, learning frequent
character combinations (like "er", "in", "li") for more efficient sequences.

## Requirements

- Python 3.10+
- PyTorch 2.0+

```bash
uv venv && source .venv/bin/activate
uv pip install torch
```

No other dependencies are needed — everything is implemented from scratch.

## License

MIT
