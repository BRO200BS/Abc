# RWKV-8 "Heron" — v2, the 2026 community upgrade round

An from-scratch implementation of **RWKV-8** — the RWKV-7g time-mix, **ROSA**
(Rapid Online Suffix Automaton — retrieval beyond attention) and **DeepEmbed** —
written completely in Python (PyTorch + numba, zero custom CUDA/C++), tuned to
train and run on a **free Colab T4 GPU**.

**v2** is a full upgrade round over the original 2,100-line implementation:
every fix and feature below is validated by `test_rwkv8.py` (31 tests, CPU-only,
`python -m pytest test_rwkv8.py -v`).

---

## What changed in v2

### Correctness fixes (bugs found in the v1 code)
1. **`RosaLM` RNN-mode state bug** — `_forward_rnn` never wrote the token-shift
   states (`x_prev_a/b/c`) back into the state dict, so RNN-mode silently
   diverged from GPT-mode for the pure-ROSA LM. Fixed; both models now verify
   GPT == RNN to fp32 round-off.
2. **RNN-mode with batch > 1** — v1 shared the G per-group QKV automata across
   batch rows and stepped only batch element 0's symbols. v2 steps B*G live
   automata (`BatchedQKVSams`), validated step-for-step against per-object
   automata.
3. **Sampler** — v1 applied temperature *after* top-p/top-k (wrong
   distribution) and could raise `IndexError` from `searchsorted` when
   `top_p → 1`. v2 samples in log space (temperature first), clamps the
   cumulative index, and adds repetition/presence penalties and min-p.
4. **Resume-able checkpoints** — v1 saved only weights+config. v2 saves
   optimizer, scheduler, GradScaler, step, best-val; `resume=True` continues
   exactly where the run stopped.
5. **Robust eval** — `eval_bpc` now scores deterministic evenly-spaced windows
   (variance-free comparisons) and restores the async flags in a `finally`.
6. **`generate` is O(T)** — prefill once through the stateful path instead of
   recomputing the whole context per token (v1 was O(n²) with a silent 2048
   cap). Seeded, with penalties; GPT-mode kept as a validation reference.

### Performance
7. **Chunk-parallel WY/UT scan** (`rwkv7g_scan(..., mode="chunked"|"auto")`) —
   the per-timestep Python loop is replaced by an exact chunk-parallel form
   (cumulative per-channel decays, a strictly-upper UT system inverted by
   log-doubling, and 8 small matmuls per chunk). Validated exact vs the
   sequential loop in **fp64 forward AND backward** (atol 1e-9) and to fp32
   round-off (~1e-5) under the model's structural `ww ∈ (0.545, 1)` bound.
   ~T/chunk× fewer kernel launches; per-chunk gradient checkpointing kept.
8. **Exact GPU-side ROSA backend** (`rosa_backend="gpu"`) — a strided
   window-equality matcher (incremental `eq_L[t,s] = AND q[t-i]==k[s-i]`, no
   hashing, no collisions) that reproduces the SAM retrieval semantics
   (endpos-class property) for all match lengths ≤ `rosa_gpu_cap` (64 default,
   matching the numba engine's own LEN_CAP). Runs on any device; the CPU SAM
   engine remains the default and the fallback.
9. **numba multicore + batched live automata** — batch kernels were already
   `prange`-parallel; v2 adds the flat-array `_sam_qkv_step_rows_nb` kernel
   that advances N independent automata one step per call, replacing the
   Python `for g in range(G)` decode loops. Storage grows geometrically;
   `dump()/load()` serializes everything.
10. **`torch.compile`** — one config flag (`compile_model=True`) compiles the
    block stack.

### Modeling
11. **Soft-ROSA** (`rosa_grad="soft"`) — retrieval over the top-M suffix-link
    match candidates with a learned length prior (`len_scale`, `len_bias`):
    exact gradients into the weight read positions (p-weighted scatter) and
    the length prior; STE surrogate for routing bits; candidate 0 is the hard
    retrieval, so `soft(M=1)` is bit-identical to `ste` (tested).
12. **Tokenizer support** (`Tokenizer`) — `bytes` (with special tokens at
    ids ≥ 256), `tiktoken`, or any HuggingFace tokenizer. Everything
    downstream (ROSA automata, embeddings) is token-id-agnostic.
13. **Chat bookmark tokens** — `<sys>/<user>/<asst>/<C_begin>/<C_end>` wrap
    each conversational round so token-ROSA retrieves any round exactly
    ([1]'s RWKV-8 notes), wired through `encode_chat` and the chat CLI.
14. **Inner monologue** (experimental) — `inner_monologue=True` adds
    `Emb(ROSA(z))` of the model's own lagged argmax outputs to its input.
15. **Scale presets** — `RWKV8Config.preset("demo"|"t4max"|"golf16mb"|"test")`,
    per-layer `rosa_bits`, and dropout (`drop_p`).

### Training
16. **Schedules** — WSD (warmup-stable-decay) and cosine-with-restarts beside
    the default cosine; **weight-decay parameter groups** (norms, mixing
    scalars, gates, LoRA and embeddings decay-free); **gradient accumulation**.
17. **DDP** — `torchrun --nproc_per_node=N ...` + `ddp=True`; rank-0-only
    logging/checkpointing.
18. **Logging** — `RunLogger`: always CSV, optional TensorBoard / wandb;
    loss/bpc/lr/tok-s/grad-norm.
19. **EMA** — weight-averaged validation (identity-based, so the tied
    emb/head weight is handled correctly).
20. **Sequential data mode** — `SequentialData` walks the corpus like a
    compressor instead of random crops.

### Inference & serving
21. **Chat CLI** — streaming RNN-mode REPL with system prompt, bookmarks,
    penalties, and `save` to checkpoint the conversation state.
22. **RNN-state save/load** — RWKV states + token automata + every live QKV
    suffix automaton serialize to disk; restored states are bit-exact (tested).
23. **Quantization** — int8 dynamic + fp16 with a measured quality-delta
    report (`quant_quality_report`).
24. **Export** — safetensors (with graceful torch.save fallback) and ONNX:
    ROSA indices are exported as *inputs*, exporting the entire neural core
    cleanly (the suffix automata are integer algorithms that run on CPU — the
    same split the async training pipeline already uses).

### Evaluation
25. **Benchmark harness** — `eval_bpc_full` (whole-val sliding-window bpc),
    `per_position_losses` (state warm-up curve), `niah_sweep`
    (1k → 100k needle retrieval; all True by design).

## Files

| File | What it is |
|---|---|
| `rwkv8_rosa_colab.ipynb` | **Start here.** Colab notebook: upload → `Runtime → Run all` (T4). Original sections 1–10 + new v2 sections 10–13. |
| `rwkv8.py` | The complete v2 implementation (~3,900 lines, commented). |
| `test_rwkv8.py` | 31-test validation suite (CPU-only). |

## Quick start (local, no Colab)

```python
import rwkv8 as m

cfg  = m.RWKV8Config.preset("demo")               # or "t4max", "golf16mb"
model = m.RWKV8Heron(cfg)

idx  = torch.randint(0, 256, (4, 256))
logits = model(idx)                               # GPT-mode
state = model.init_state(1)                       # RNN-mode: O(1)/token
logits, state = model(idx[:, :1], state=state)

data = m.ByteData(m.download_enwik8("data"))
m.train_model(model, data, steps=3000, sched="wsd", ema_decay=0.999,
              device="cuda", logger=m.RunLogger(out_dir="runs/demo"))

toks = m.generate(model, list(b"First Citizen"), n_new=200, mode="rnn")
m.niah_sweep(model, filler_lens=[1_000, 100_000])
```

CLI: `python rwkv8.py train|chat|generate|benchmark|export ...`
(`python rwkv8.py --help`).

## Repository map (original contributions — credit these authors)

See the credits section at the end of the notebook; v2 adds nothing to the
architecture itself — it is engineering, correctness and tooling work on top
of [BlinkDL/RWKV-LM RWKV-v8] and the ROSA research community's code.

## Design notes & honest caveats (v2)

- The WY scan is exact for chunk ≤ 64 (the model's `ww ∈ (0.545, 1)` structural
  bound keeps e^l in fp32 range); the scan clamps larger chunks automatically.
- `rosa_backend="gpu"` is exact for match lengths ≤ `rosa_gpu_cap` (64); longer
  matches fall back to the best shorter one — the numba engine caps its
  recomputed match length at 64 as well. For strict unbounded exactness keep
  the CPU engine.
- Soft-ROSA is a GPT-mode training feature; RNN-mode inference always uses the
  exact hard single-match retrieval.
- enwik8 numbers remain raw-byte bpc. Not affiliated with the RWKV team.
  Research/education code, MIT — please credit the original authors.
