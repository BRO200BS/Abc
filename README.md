# RWKV-8 "Heron" — complete pure-Python implementation for Google Colab T4

An from-scratch, frontier-grade implementation of **RWKV-8** — the architecture released in
[BlinkDL/RWKV-LM `RWKV-v8`](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v8) — written
**completely in Python** (PyTorch + numba, zero custom CUDA/C++), tuned to train and run on a
**free Colab T4 GPU**.

```
RWKV-8 "Heron"  =  RWKV-7g time-mix        (recurrent generalized delta rule, exact kernel port)
                 + ROSA                     (Rapid Online Suffix Automaton — retrieval beyond attention)
                 + DeepEmbed                (per-token FFN modulation)
```

## Files

| File | What it is |
|---|---|
| `rwkv8_rosa_colab.ipynb` | **Start here.** Self-contained Colab notebook: upload → `Runtime → Run all` (T4). Full run ≈ 60–90 min, `FAST=True` ≈ 20–30 min. |
| `rwkv8.py` | The complete implementation (~2000 lines, commented) — also embedded in the notebook via `%%writefile`. |

## What's inside

- **Exact ROSA engine** — online suffix automata for plain token ROSA and QKV-ROSA
  (4-bit symbol groups), validated bit-for-bit against the reference's brute-force
  specification on hundreds of random streams. numba-JIT kernels (GIL-released) at
  ~25M symbol-steps/s with an exact pure-Python fallback; integer-only, FPGA-friendly
  flat transition tables; match truncation `K`; stateful automata for O(1)-per-token
  RNN-mode inference.
- **RWKV-7g time-mix** — a bit-exact pure-PyTorch port of `wkv7_cuda.cu`
  (`S ← S·diag(e^(−e^w)) − (S·kk)⊗(kk·a) + v⊗k; y = S·r`), with chunk-wise gradient
  checkpointing and fp32 state accumulation. Chunked == unchunked == manual loop,
  gradients exact to 0.0, state carry exact.
- **Hybrid ROSA backward pass** — exact gradients for payloads; **the value-read
  scatter gradient equals BlinkDL's flip-every-bit counterfactual VJP with cosine
  similarity 1.0000** (validated in the notebook); STE surrogate for routing bits
  (`grad_mode="ste"`, default), full exact counterfactual (`grad_mode="exact"`),
  and the reference's detached-branch behaviour (`grad_mode="none"`).
- **Two model families** — `RWKV8Heron` (full RWKV-8: time-mix ∥ ROSA + DeepEmbed FFN)
  and `RosaLM` (pure-ROSA LM, `rosa4bitLM`-style). Both with GPT-mode and stateful
  RNN-mode (verified equal to ~1e-6).
- **Async CPU–GPU pipeline** — token-ROSA indices prefetched on a background thread for
  the next batch; each block's QKV-ROSA retrieval runs on the CPU concurrently with the
  GPU time-mix (the ROSA-Tuning design).
- **Training + eval** — enwik8 (raw bytes, 90M/5M split), warmup-cosine AdamW, bits-per-char
  metrics, checkpoints, sampling; parameter-golf artifact report (the default 7.1M-param
  Heron fits the 16 MB fp16 budget).
- **Showcase demos** (all reproduce reference behaviours):
  - **Counting** — the faithful `251018` recipe; reaches **loss ≈ 0.245** (reference: ~0.25).
  - **MQAR** — parameter-free ROSA: **100%** at any context length; trained ablation:
    RWKV-7g alone 1.6% (chance) vs **RWKV-7g + ROSA 100%**.
  - **Reverse digits** — the `251105` recipe (L4-C32, 39.6K params); trains toward the
    reference's 99.6% with GPU-scale steps.
  - **Standalone ROSA compressor** — zero parameters, zero training: ~3.09 bpc on enwik8
    (beats a plain bigram model).
  - **Needle-in-a-haystack** — retrieved verbatim at any distance, by design.

## Quick start (local, no Colab)

```python
import rwkv8 as m

cfg  = m.RWKV8Config(vocab_size=256, n_layer=6, n_embd=256)   # 7.1M params
model = m.RWKV8Heron(cfg)

idx  = torch.randint(0, 256, (4, 256))
logits = model(idx)                       # GPT-mode
loss = F.cross_entropy(logits.reshape(-1, 256), idx[:, 1:].reshape(-1)); loss.backward()

state = model.init_state(1)               # RNN-mode: O(1)/token, unbounded context
logits, state = model(idx[:, :1], state=state)

data = m.ByteData(m.download_enwik8("data"))
m.train_model(model, data, steps=3000, device="cuda")         # trainer w/ async ROSA
toks = m.generate_rnn(model, list(b"First Citizen"), n_new=200)
```

## Repository-by-repository influence map

| Repository | What it contributed here |
|---|---|
| [BlinkDL/RWKV-LM · RWKV-v8](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v8) | Everything foundational: the RWKV-7g recurrence (`wkv7_cuda.cu`), ROSA (plain/QKV/4-bit, `samx_qkv_slow`, `rosa_slow_ref`), `Emb(ROSA(x))`, DeepEmbed, the demo tasks and their reference numbers, the counterfactual gradient |
| [xiaoiecc/qkv-rosa-fast-exact-backward](https://github.com/xiaoiecc/qkv-rosa-fast-exact-backward) | n-bit hard codes; the separate learned no-match payload `emb0`; exact counterfactual VJP as a first-class mode (`grad_mode="exact"`, numba-accelerated) |
| [wjie98/rosa_soft](https://github.com/wjie98/rosa_soft) | exact hard forward + surrogate backward; the value-read scatter gradient — which we show is *exactly* the counterfactual VJP (cosine 1.0000) |
| [johanwind/wind_rosa](https://github.com/johanwind/wind_rosa) | match truncation `K` in the engine; finite-difference gradient framing; O(1) lookups with linear state |
| [zyaaa-ux/ROSA-Tuning](https://github.com/zyaaa-ux/ROSA-Tuning) (arXiv 2602.02499) | binary discretization; counterfactual gradients; the **async CPU–GPU pipeline** (background-thread retrieval, batch prefetching) |
| [aabbdev/rosa](https://github.com/aabbdev/rosa) | learned read gate before retrieval enters the stream; match-length calibration (our `len_emb` buckets) |
| [bcml-ai/rosa-plus](https://github.com/bcml-ai/rosa-plus) | the statistical backoff idea — our standalone ROSA compressor's Witten–Bell-style length-weighted blending |
| [KakaruHayate/RWKV8-ROSA-FPGA](https://github.com/KakaruHayate/RWKV8-ROSA-FPGA) | the integer-only, CPU-side, no-float/no-hash design of the engine (flat int32 transition tables) |
| [Juste-Leo2/ROSA-GPU-RWKV8](https://github.com/Juste-Leo2/ROSA-GPU-RWKV8) | GPU-side ROSA as an alternative engine — documented in the notebook's extensions (on T4, the CPU engine + async pipeline already covers this model class) |
| [x-0D/RASP](https://github.com/x-0D/RASP) | the ROSA+ / CRF / syntactic-feature extension direction (notebook §10 extensions) |
| [openai/parameter-golf](https://github.com/openai/parameter-golf) | bits-per-byte evaluation, parameter budgeting, weight tying; the artifact report (default model: 14.2 MB fp16 < 16 MB budget) |

## Design notes & honest caveats

- **The QKV-ROSA semantics are exact** (same automaton as the reference; validated against
  brute force). The *retrieved match length* is recomputed exactly as the longest common
  suffix at the read position — the reference's carried length pointer can overstate, and
  the reference itself never consumes it.
- **`grad_mode="none"`** reproduces the reference LM behaviour exactly (the reference's
  RosaLM builds its branch output from detached tensors — no gradient into the code
  streams). `"ste"` (default) is our upgrade: exact value-read gradient + STE routing.
  `"exact"` is BlinkDL's full counterfactual, made practical with numba.
- **The reverse demo needs GPU-scale steps** to reach the reference's 99.6% (the
  reference trains far longer than any CPU budget; the notebook trains 4000 steps on T4
  and prints the trajectory).
- enwik8 numbers are **raw-byte bpc** (the honest byte-level metric; classic ~1.0 bpc
  figures use preprocessed text).
- Not affiliated with the RWKV team. Research/education code, MIT license — please credit
  the authors in the table above if you build on this.
