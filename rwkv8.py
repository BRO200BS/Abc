# ==========================================================================
# >>> section from p1_engine.py
# ==========================================================================

# ============================================================================
# rwkv8.py — RWKV-8 "Heron" LLM with ROSA, completely in Python (PyTorch)
#                                   v2 — the 2026 community upgrade round
#
# A from-scratch, GPU-ready (Google Colab T4) implementation of:
#   * RWKV-7g time-mix  (exact port of BlinkDL's cuda/wkv7_cuda.cu to PyTorch)
#   * ROSA              (Rapid Online Suffix Automaton — RWKV-8's neurosymbolic
#                        retrieval mechanism that goes "beyond attention")
#   * DeepEmbed         (RWKV-8's per-token FFN modulation)
#
# v2 additions (all validated by test_rwkv8.py):
#   * chunk-parallel RWKV-7g scan  (WY/UT form, ~10-30x fewer kernel launches;
#     exact to fp32 round-off vs the sequential loop — fwd AND bwd)
#   * exact GPU-side QKV/token ROSA backend (bit-packed window matching;
#     exact for match lengths <= rosa_gpu_cap)
#   * multi-candidate retrieval (suffix-link chain) powering grad_mode="soft"
#   * top-p/top-k/min-p/repetition-penalty sampler rewritten in log space
#     (fixes the top-p IndexError and the temperature-ordering bug)
#   * RosaLM RNN-mode state write-back fix (GPT == RNN now, really)
#   * resume-able checkpoints (optimizer/scheduler/scaler/step/rng), best-val
#   * fast O(T) RNN-mode generate with prefill; no silent context cap
#   * tokenizer support (bytes | tiktoken | HF) + chat bookmark tokens
#   * trainer: WSD / cosine-restart schedules, weight-decay param groups,
#     grad accumulation, DDP, csv/tensorboard/wandb logging, EMA, sequential
#     data mode
#   * chat CLI, RNN-state save/load, int8/fp16 quantization + quality report,
#     safetensors/ONNX export, benchmark harness (full-val bpc, per-position
#     loss curves, long-context NIAH sweep)
#   * config presets: demo / t4max / golf16mb / test
#
# References (all ideas here are credited to their authors):
#   [1] BlinkDL/RWKV-LM  RWKV-v8  (RWKV-8 "Heron", ROSA 1bit/4bit, rosaQKV,
#       rosa4bitLM, reverse demo, wkv7_cuda.cu) .............. MIT
#   [2] xiaoiecc/qkv-rosa-fast-exact-backward (n-bit codes, separate no-match
#       payload emb0, exact counterfactual VJP) .............. MIT
#   [3] wjie98/rosa_soft (exact hard forward; exact value-read gradient +
#       soft/STE surrogate for routing gradients) ............ MIT
#   [4] johanwind/wind_rosa (match truncation K, finite-difference gradients) MIT
#   [5] zyaaa-ux/ROSA-Tuning (arXiv 2602.02499: binary discretization,
#       counterfactual gradient, async CPU-GPU pipeline) ..... MIT
#   [6] bcml-ai/rosa-plus (statistical backoff predictor for no-match) .... GPL-3
#   [7] aabbdev/rosa (read gates, match-length calibration) .. MIT
#   [8] KakaruHayate/RWKV8-ROSA-FPGA (integer-only, CPU-side ROSA design) .
#   [9] Juste-Leo2/ROSA-GPU-RWKV8 (GPU-side ROSA explorations) .
#  [10] x-0D/RASP (ROSA+ with syntactic/CRF extensions — related work) .
#  [11] openai/parameter-golf (bits-per-byte eval, parameter budget,
#       weight tying) ........................................ MIT
#
# Design notes
#   * ROSA runs on the CPU (integer-only suffix automata — the reference
#     design: "no dot product, no softmax, no float, no KV cache"), the neural
#     network runs on the GPU.  numba JIT kernels are used when available,
#     with an exact pure-Python fallback so the code always runs.
#   * The QKV-ROSA retrieval is EXACT (identical semantics to samx_qkv_slow /
#     rosa_slow_ref in [1]).  Its backward pass is a hybrid:
#       - emb0/emb1/gate/len embeddings: exact gradients
#       - V (value) bits: EXACT gradient given the routing (scatter through the
#         retrieval read position, as in [3])
#       - Q/K (routing) bits: straight-through surrogate (default), sampled
#         counterfactual, or BlinkDL's exact flip-every-bit counterfactual
#         (grad_mode="exact", for validation on short sequences)
#   * The RWKV-7g scan is a faithful pure-PyTorch port of the CUDA kernel,
#     with optional gradient checkpointing per time-chunk to bound memory.
#   * Models support both GPT-mode (parallel over context) and stateful
#     RNN-mode (O(1) per token, carrying RWKV state + live suffix automata).
#
# License: MIT.  This is a research/education implementation, not affiliated
# with the RWKV team.
# ============================================================================

from __future__ import annotations

import copy
import math
import os
import random
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:  # numba is preinstalled on Google Colab; kernels release the GIL so ROSA
    import numba  # can run truly in parallel with GPU work (async pipeline [5])
    from numba import njit, prange
    NUMBA_OK = True
except Exception:  # pragma: no cover
    numba = None
    njit = prange = None
    NUMBA_OK = False

# ============================================================================
# Section 1 — ROSA: Rapid Online Suffix Automaton
# ============================================================================
# ROSA(x)[i] = the token that followed the PREVIOUS (most recent) occurrence of
# the LONGEST suffix of x[0..i] that has occurred before, or -1 if every suffix
# (down to length 1) is novel.  It is a parameter-free next-token predictor
# with perfect, lossless, unbounded-range recall — "beyond attention" [1].
#
# The QKV generalization [1] (samx): build the automaton over a KEY stream,
# stream-match a QUERY stream against it, and read the VALUE token that
# followed the matched key substring at its most recent occurrence:
#
#   for i, (q_i, k_i) in enumerate(zip(Q, K)):
#       (p, x) = extend carried match (w,h) by query symbol q_i
#       v = canonical SAM state for the length-x suffix
#       walk suffix links until a state with a previous occurrence (m>0, r>=0)
#       y[i] = V[r+1]            (r+1 <= i: reading V[i] is causal — V[i] is a
#                                 function of tokens <= i, used to predict i+1)
#       insert key symbol k_i into the automaton; update last-occurrence marks
# ============================================================================


# ----------------------------------------------------------------------------
# 1.1  Pure-Python reference implementations (ground truth, from [1])
# ----------------------------------------------------------------------------

def rosa_token_ref(x: List[int]) -> List[int]:
    """Plain ROSA over one token stream (verbatim port of rosa() in [1]).

    Returns y where y[i] = x[e+1] with e the end of the most recent previous
    occurrence of the longest repeated suffix of x[:i+1], or -1 if none.
    """
    n = len(x); y = [-1] * n; s = 2 * n + 1
    b: List[Optional[dict]] = [None] * s; c = [-1] * s; d = [0] * s; e = [-1] * s
    b[0] = {}; g = 0; z = 1
    for i, t in enumerate(x):
        r = z; z += 1; b[r] = {}; d[r] = d[g] + 1; p = g
        while p != -1 and t not in b[p]:
            b[p][t] = r; p = c[p]
        if p == -1:
            c[r] = 0
        else:
            q = b[p][t]
            if d[p] + 1 == d[q]:
                c[r] = q
            else:
                u = z; z += 1; b[u] = b[q].copy(); d[u] = d[p] + 1
                c[u] = c[q]; e[u] = e[q]
                while p != -1 and b[p][t] == q:
                    b[p][t] = u; p = c[p]
                c[q] = c[r] = u
        v = g = r; a = -1
        while v != -1:
            if d[v] > 0 and e[v] >= 0:
                a = x[e[v] + 1]; break
            v = c[v]
        y[i] = a; v = g
        while v != -1 and e[v] < i:
            e[v] = i; v = c[v]
    return y


def rosa_qkv_ref(qqq: List[int], kkk: List[int], vvv: List[int]) -> List[int]:
    """QKV-ROSA reference (verbatim port of samx_qkv_slow in [1]).

    SAM over the key stream; query stream matched incrementally; output the
    value symbol that followed the matched key substring's most recent
    occurrence.  -1 if no match at all (later mapped to a learned no-match
    payload, following [2]).
    """
    n = len(qqq); y = [-1] * n; s = 2 * n + 1
    t: List[Optional[dict]] = [None] * s; f = [-1] * s; m = [0] * s; r = [-1] * s
    t[0] = {}; g = 0; u = 1; w = h = 0
    assert n == len(kkk) == len(vvv)
    for i, (q, k) in enumerate(zip(qqq, kkk)):
        p, x = w, h
        while p != -1 and q not in t[p]:
            x = m[p] if x > m[p] else x; p = f[p]
        p, x = (t[p][q], x + 1) if p != -1 else (0, 0)
        v = p
        while f[v] != -1 and m[f[v]] >= x:
            v = f[v]
        while v != -1 and (m[v] <= 0 or r[v] < 0):
            v = f[v]
        y[i] = vvv[r[v] + 1] if v != -1 else -1
        w, h = p, x
        j = u; u += 1; t[j] = {}; m[j] = m[g] + 1; p = g
        while p != -1 and k not in t[p]:
            t[p][k] = j; p = f[p]
        if p == -1:
            f[j] = 0
        else:
            d = t[p][k]
            if m[p] + 1 == m[d]:
                f[j] = d
            else:
                b = u; u += 1; t[b] = t[d].copy(); m[b] = m[p] + 1
                f[b] = f[d]; r[b] = r[d]; f[d] = f[j] = b
                while p != -1 and t[p][k] == d:
                    t[p][k] = b; p = f[p]
        v = g = j
        while v != -1 and r[v] < i:
            r[v] = i; v = f[v]
    return y


def rosa_qkv_brute(q: List[int], k: List[int], v: List[int],
                   max_len: int = 0) -> Tuple[List[int], List[int]]:
    """Brute-force specification of QKV-ROSA (port of rosa_slow_ref in [1]).

    For each i: longest w (up to max_len) with the latest end j+w<=i such that
    k[j:j+w] == q[i+1-w:i+1]; then (idx[i], ln[i]) = (v[j+w], w).
    O(T^3) — used only to validate the SAM engines.
    """
    n = len(q)
    idx = [0] * n; ln = [0] * n
    for i in range(n):
        found = False
        for w in range(i + 1, 0, -1):
            if max_len and w > max_len:
                continue
            tsub = q[i + 1 - w: i + 1]
            for j in range(i - w, -1, -1):
                if k[j: j + w] == tsub:
                    idx[i] = v[j + w]; ln[i] = w; found = True; break
            if found:
                break
    return idx, ln


def _rosa_qkv_ref_full(qs: List[int], ks: List[int], vs: List[int]
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-pass pure-Python QKV SAM returning (sym, ln, pos) per position.

    sym = retrieved value symbol (-1 if no match), ln = match length,
    pos = read position in the value stream (-1 if no match).
    This is the exact algorithm of rosa_qkv_ref with instrumentation added.
    """
    n = len(qs)
    sym = np.full(n, -1, dtype=np.int32); ln = np.zeros(n, dtype=np.int32)
    pos = np.full(n, -1, dtype=np.int32)
    s = 2 * n + 1
    t: List[Optional[dict]] = [None] * s; f = [-1] * s; m = [0] * s
    r = [-1] * s; t[0] = {}; g = 0; u = 1; w = h = 0
    for i, (q, k) in enumerate(zip(qs, ks)):
        p, x = w, h
        while p != -1 and q not in t[p]:
            x = m[p] if x > m[p] else x; p = f[p]
        p, x = (t[p][q], x + 1) if p != -1 else (0, 0)
        v = p
        while f[v] != -1 and m[f[v]] >= x:
            v = f[v]
        while v != -1 and (m[v] <= 0 or r[v] < 0):
            v = f[v]
        if v != -1:
            sym[i] = vs[r[v] + 1]
            # true match length = longest common suffix of qs[:i+1], ks[:r+1]
            rr = r[v]
            lim = min(i, rr, 63) + 1
            L = 0
            while L < lim and qs[i - L] == ks[rr - L]:
                L += 1
            ln[i] = L
            pos[i] = rr + 1
        w, h = p, x
        j = u; u += 1; t[j] = {}; m[j] = m[g] + 1; p = g
        while p != -1 and k not in t[p]:
            t[p][k] = j; p = f[p]
        if p == -1:
            f[j] = 0
        else:
            d = t[p][k]
            if m[p] + 1 == m[d]:
                f[j] = d
            else:
                b = u; u += 1; t[b] = t[d].copy(); m[b] = m[p] + 1
                f[b] = f[d]; r[b] = r[d]; f[d] = f[j] = b
                while p != -1 and t[p][k] == d:
                    t[p][k] = b; p = f[p]
        v = g = j
        while v != -1 and r[v] < i:
            r[v] = i; v = f[v]
    return sym, ln, pos


def _rosa_qkv_ref_full_m(qs: List[int], ks: List[int], vs: List[int], M: int = 4
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                    np.ndarray, np.ndarray]:
    """Multi-candidate QKV-ROSA reference: at each position, walk the suffix-
    link chain and collect up to M distinct match candidates (longest first).

    Returns (sym[M,N], ln[M,N], pos[M,N], pad count) — candidates beyond the
    chain are padded with sym=-1, ln=0, pos=-1.  Candidate 0 is identical to
    the single-match engine.  Every candidate (ln_m, pos_m) satisfies:
    k[pos_m-ln_m : pos_m] == q[t-ln_m+1 : t+1] and pos_m <= t (causal),
    with strictly decreasing ln_m (the endpos-class property of SAMs)."""
    n = len(qs)
    sym = np.full((M, n), -1, dtype=np.int32)
    ln = np.zeros((M, n), dtype=np.int32)
    pos = np.full((M, n), -1, dtype=np.int32)
    s = 2 * n + 1
    t: List[Optional[dict]] = [None] * s; f = [-1] * s; m = [0] * s
    r = [-1] * s; t[0] = {}; g = 0; u = 1; w = h = 0
    for i, (q, k) in enumerate(zip(qs, ks)):
        p, x = w, h
        while p != -1 and q not in t[p]:
            x = m[p] if x > m[p] else x; p = f[p]
        p, x = (t[p][q], x + 1) if p != -1 else (0, 0)
        v = p
        while f[v] != -1 and m[f[v]] >= x:
            v = f[v]
        nc = 0
        vv = v
        while vv != -1 and nc < M:
            if m[vv] > 0 and r[vv] >= 0:
                rr = r[vv]
                lim = min(i, rr, 63) + 1
                L = 0
                while L < lim and qs[i - L] == ks[rr - L]:
                    L += 1
                if L > 0:
                    sym[nc, i] = vs[rr + 1]; ln[nc, i] = L; pos[nc, i] = rr + 1
                    nc += 1
            vv = f[vv]
        w, h = p, x
        j = u; u += 1; t[j] = {}; m[j] = m[g] + 1; p = g
        while p != -1 and k not in t[p]:
            t[p][k] = j; p = f[p]
        if p == -1:
            f[j] = 0
        else:
            d = t[p][k]
            if m[p] + 1 == m[d]:
                f[j] = d
            else:
                b = u; u += 1; t[b] = t[d].copy(); m[b] = m[p] + 1
                f[b] = f[d]; r[b] = r[d]; f[d] = f[j] = b
                while p != -1 and t[p][k] == d:
                    t[p][k] = b; p = f[p]
        v = g = j
        while v != -1 and r[v] < i:
            r[v] = i; v = f[v]
    return sym, ln, pos, np.full(n, nc, np.int32)


def rosa_qkv_ref_batch(qsym: np.ndarray, ksym: np.ndarray, vsym: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pure-Python fallback over rows of symbol streams [N,T]."""
    N, T = qsym.shape
    sym = np.full((N, T), -1, dtype=np.int32)
    ln = np.zeros((N, T), dtype=np.int32)
    pos = np.full((N, T), -1, dtype=np.int32)
    for n in range(N):
        s_, l_, p_ = _rosa_qkv_ref_full(
            [int(t) for t in qsym[n]], [int(t) for t in ksym[n]],
            [int(t) for t in vsym[n]])
        sym[n] = s_; ln[n] = l_; pos[n] = p_
    return sym, ln, pos


def rosa_token_ref_batch(ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Plain ROSA over each row of ids [N,T] (pure-Python fallback path).

    Uses the QKV SAM with q=k=v=row: mathematically identical to
    rosa_token_ref (the plain ROSA chain walk from the fresh state finds the
    same longest previously-occurring suffix), and gives match lengths too.
    """
    sym, ln, _pos = rosa_qkv_ref_batch(ids, ids, ids)
    return sym, ln

# ==========================================================================
# >>> section from p1b_numba.py
# ==========================================================================

# ----------------------------------------------------------------------------
# 1.2  numba-JIT suffix-automaton kernels (integer-only, GIL-released)
#      Flat int32 transition tables (alphabet x states) — no hashing, no
#      floats: the FPGA-friendly design point of [8], at CPU speed.
# ----------------------------------------------------------------------------

if NUMBA_OK:

    LEN_CAP = 64  # cap for exact match-length recomputation (see _sam_qkv_one)

    @njit(nogil=True, cache=True)
    def _sam_qkv_one(qs, ks, vs, A, K, out_sym, out_len, out_pos):
        """Run the QKV suffix automaton over one triple of symbol streams.

        qs/ks/vs: 1-D int32 arrays of symbols (0..A-1), length T.
        out_*:    preallocated int32[T] outputs.
        Alphabet A (16 for 4-bit groups, or the vocab for token ROSA).
        K > 0 caps the match length (wind_rosa truncation [4]); K == 0 means
        unbounded (exact reference semantics of [1]).
        """
        T = qs.shape[0]
        SMAX = 2 * T + 2
        trans = np.full((SMAX, A), -1, np.int32)
        link = np.full(SMAX, -1, np.int32)
        ln = np.zeros(SMAX, np.int32)
        last = np.full(SMAX, -1, np.int32)
        z = 1; g = 0; w = 0; h = 0
        for i in range(T):
            q = qs[i]; kk = ks[i]
            # --- query step: extend the carried match (w,h) with symbol q ---
            p = w; x = h
            while p != -1 and trans[p, q] == -1:
                if ln[p] < x:
                    x = ln[p]
                p = link[p]
            if p != -1:
                p = trans[p, q]; x = x + 1
            else:
                p = 0; x = 0
            if K > 0 and x > K:  # truncate the match to K (canonical retract)
                x = K
                vv = p
                while link[vv] != -1 and ln[link[vv]] >= x:
                    vv = link[vv]
                p = vv
            # --- retrieval: longest suffix of the match with a previous
            #     occurrence in the key stream (ties -> most recent) --------
            vv = p
            while link[vv] != -1 and ln[link[vv]] >= x:
                vv = link[vv]
            fs = -1; fl = 0; fp = -1
            while vv != -1:
                if ln[vv] > 0 and last[vv] >= 0:
                    fs = vs[last[vv] + 1]
                    # True match length: longest common suffix of qs[:i+1] and
                    # ks[:r+1] (r = end of the occurrence we read from).  The
                    # carried length x can overstate (the reference algorithm
                    # keeps a loose (state,len) pointer), so recompute exactly
                    # from the streams; capped for speed.
                    r = last[vv]
                    mlim = i if i < r else r
                    mlim = mlim + 1
                    if mlim > LEN_CAP:
                        mlim = LEN_CAP
                    if K > 0 and mlim > K:
                        mlim = K
                    L = 0
                    while L < mlim and qs[i - L] == ks[r - L]:
                        L += 1
                    fl = L
                    fp = last[vv] + 1
                    break
                vv = link[vv]
            out_sym[i] = fs; out_len[i] = fl; out_pos[i] = fp
            w = p; h = x
            # --- key step: SAM-extend with symbol kk ------------------------
            j = z; z += 1
            ln[j] = ln[g] + 1
            p = g
            while p != -1 and trans[p, kk] == -1:
                trans[p, kk] = j; p = link[p]
            if p == -1:
                link[j] = 0
            else:
                d = trans[p, kk]
                if ln[p] + 1 == ln[d]:
                    link[j] = d
                else:  # clone
                    b = z; z += 1
                    for a in range(A):
                        trans[b, a] = trans[d, a]
                    ln[b] = ln[p] + 1; link[b] = link[d]; last[b] = last[d]
                    while p != -1 and trans[p, kk] == d:
                        trans[p, kk] = b; p = link[p]
                    link[d] = b; link[j] = b
            g = j
            vv = g
            while vv != -1 and last[vv] < i:
                last[vv] = i; vv = link[vv]

    @njit(nogil=True, cache=True)
    def _phi_payload(sym, pos, gy_rows, e0, e1, nbits, T):
        """Downstream objective contributed by one group's payload pattern:
        sum_t sum_c gy[t,c] * payload[t,c],  payload = e0 (no match) or
        sign(bit of retrieved symbol) * e1 (match)."""
        s = 0.0
        for t in range(T):
            if pos[t] < 0:
                for c in range(nbits):
                    s += gy_rows[t * nbits + c] * e0[c]
            else:
                sy = sym[t]
                for c in range(nbits):
                    if (sy >> c) & 1:
                        sgn = 1.0
                    else:
                        sgn = -1.0
                    s += gy_rows[t * nbits + c] * sgn * e1[c]
        return s

    @njit(nogil=True, cache=True)
    def _cf_phi_one(qs, ks, vs, A, K, tau, q_row, k_row, v_row, gy_rows,
                    e0, e1, nbits, gq_row, gk_row, gv_row):
        """Exact single-symbol-flip counterfactual gradients for ONE group.

        qs/ks/vs: int32 [T] symbol streams of this group.
        q_row/k_row/v_row: float32 [T*nbits] continuous pre-binarization
            values (layout t*nbits + bit).
        gy_rows: float32 [T*nbits] downstream grads dL/d(payload).
        e0/e1: float32 [nbits] payloads of this group.
        BlinkDL's counterfactual rule [1]: g = (phi(bit=1) - phi(bit=0)) /
        (2 * max(|x|, tau)).
        """
        T = qs.shape[0]
        sym0 = np.empty(T, np.int32); ln0 = np.empty(T, np.int32)
        pos0 = np.empty(T, np.int32)
        _sam_qkv_one(qs, ks, vs, A, K, sym0, ln0, pos0)
        base = _phi_payload(sym0, pos0, gy_rows, e0, e1, nbits, T)
        sym1 = np.empty(T, np.int32); ln1 = np.empty(T, np.int32)
        pos1 = np.empty(T, np.int32)
        for which in range(3):  # 0=q, 1=k, 2=v
            for t in range(T):
                if which == 0:
                    cur = qs[t]
                elif which == 1:
                    cur = ks[t]
                else:
                    cur = vs[t]
                xrow = q_row if which == 0 else (k_row if which == 1 else v_row)
                xcont = xrow[t * nbits]
                mag = abs(xcont)
                if mag < tau:
                    mag = tau
                for c in range(nbits):
                    xcont = xrow[t * nbits + c]
                    mag = abs(xcont)
                    if mag < tau:
                        mag = tau
                    bit_on = (cur >> c) & 1
                    if bit_on:
                        p1 = base
                    else:
                        newv = cur | (1 << c)
                        if which == 0:
                            qs[t] = newv
                            _sam_qkv_one(qs, ks, vs, A, K, sym1, ln1, pos1)
                        elif which == 1:
                            ks[t] = newv
                            _sam_qkv_one(qs, ks, vs, A, K, sym1, ln1, pos1)
                        else:
                            vs[t] = newv
                            _sam_qkv_one(qs, ks, vs, A, K, sym1, ln1, pos1)
                        p1 = _phi_payload(sym1, pos1, gy_rows, e0, e1, nbits, T)
                    if not bit_on:
                        p0 = base
                    else:
                        newv = cur & (~(1 << c))
                        if which == 0:
                            qs[t] = newv
                            _sam_qkv_one(qs, ks, vs, A, K, sym1, ln1, pos1)
                        elif which == 1:
                            ks[t] = newv
                            _sam_qkv_one(qs, ks, vs, A, K, sym1, ln1, pos1)
                        else:
                            vs[t] = newv
                            _sam_qkv_one(qs, ks, vs, A, K, sym1, ln1, pos1)
                        p0 = _phi_payload(sym1, pos1, gy_rows, e0, e1, nbits, T)
                    # restore
                    if which == 0:
                        qs[t] = cur
                    elif which == 1:
                        ks[t] = cur
                    else:
                        vs[t] = cur
                    g = (p1 - p0) / (2.0 * mag)
                    if which == 0:
                        gq_row[t * nbits + c] = g
                    elif which == 1:
                        gk_row[t * nbits + c] = g
                    else:
                        gv_row[t * nbits + c] = g

    @njit(nogil=True, parallel=True, cache=True)
    def _sam_qkv_batch_nb(qsym, ksym, vsym, A, K, out_sym, out_len, out_pos):
        N = qsym.shape[0]
        for n in prange(N):
            _sam_qkv_one(qsym[n], ksym[n], vsym[n], A, K,
                         out_sym[n], out_len[n], out_pos[n])

    # ---- v2: multi-candidate retrieval (suffix-link chain, longest first) --
    @njit(nogil=True, cache=True)
    def _sam_qkv_one_m(qs, ks, vs, A, K, M, out_sym, out_len, out_pos):
        """_sam_qkv_one that returns up to M candidates per position.
        out_*: int32 [M, T]; candidates padded with (-1, 0, -1)."""
        T = qs.shape[0]
        SMAX = 2 * T + 2
        trans = np.full((SMAX, A), -1, np.int32)
        link = np.full(SMAX, -1, np.int32)
        ln = np.zeros(SMAX, np.int32)
        last = np.full(SMAX, -1, np.int32)
        z = 1; g = 0; w = 0; h = 0
        for i in range(T):
            q = qs[i]; kk = ks[i]
            p = w; x = h
            while p != -1 and trans[p, q] == -1:
                if ln[p] < x:
                    x = ln[p]
                p = link[p]
            if p != -1:
                p = trans[p, q]; x = x + 1
            else:
                p = 0; x = 0
            if K > 0 and x > K:
                x = K
                vv = p
                while link[vv] != -1 and ln[link[vv]] >= x:
                    vv = link[vv]
                p = vv
            vv = p
            while link[vv] != -1 and ln[link[vv]] >= x:
                vv = link[vv]
            for c in range(M):                      # reset candidate row
                out_sym[c, i] = -1; out_len[c, i] = 0; out_pos[c, i] = -1
            nc = 0
            while vv != -1 and nc < M:
                if ln[vv] > 0 and last[vv] >= 0:
                    r = last[vv]
                    mlim = (i if i < r else r) + 1
                    if mlim > LEN_CAP:
                        mlim = LEN_CAP
                    if K > 0 and mlim > K:
                        mlim = K
                    L = 0
                    while L < mlim and qs[i - L] == ks[r - L]:
                        L += 1
                    if L > 0:
                        out_sym[nc, i] = vs[r + 1]
                        out_len[nc, i] = L
                        out_pos[nc, i] = r + 1
                        nc += 1
                vv = link[vv]
            w = p; h = x
            j = z; z += 1
            ln[j] = ln[g] + 1
            p = g
            while p != -1 and trans[p, kk] == -1:
                trans[p, kk] = j; p = link[p]
            if p == -1:
                link[j] = 0
            else:
                d = trans[p, kk]
                if ln[p] + 1 == ln[d]:
                    link[j] = d
                else:
                    b = z; z += 1
                    for a in range(A):
                        trans[b, a] = trans[d, a]
                    ln[b] = ln[p] + 1; link[b] = link[d]; last[b] = last[d]
                    while p != -1 and trans[p, kk] == d:
                        trans[p, kk] = b; p = link[p]
                    link[d] = b; link[j] = b
            g = j
            vv = g
            while vv != -1 and last[vv] < i:
                last[vv] = i; vv = link[vv]

    @njit(nogil=True, parallel=True, cache=True)
    def _sam_qkv_batch_nb_m(qsym, ksym, vsym, A, K, M,
                            out_sym, out_len, out_pos):
        N = qsym.shape[0]
        for n in prange(N):
            _sam_qkv_one_m(qsym[n], ksym[n], vsym[n], A, K, M,
                           out_sym[n], out_len[n], out_pos[n])

    # ---- v2: batched RNN-mode stepping (one token for N live automata) -----
    @njit(nogil=True, parallel=True, cache=True)
    def _sam_qkv_step_rows_nb(trans, link, ln, last, meta, qh, kh, vh,
                              q, k, v, A, K, SMAX, TMAX,
                              out_sym, out_len, out_pos):
        """Advance N independent QKV automata by ONE symbol each (in place).

        trans [N,SMAX,A] int32; link/ln/last [N,SMAX] int32;
        meta [N,5] int64 rows: (g, w, h, n, z);
        qh/kh/vh [N,TMAX] int32 symbol histories (for exact LCS recompute);
        q/k/v [N] the symbols streamed at this step;
        out_* [N] retrieval results for this step."""
        N = q.shape[0]
        for n in prange(N):
            g = meta[n, 0]; w = meta[n, 1]; h = meta[n, 2]
            i = meta[n, 3]; z = meta[n, 4]
            qi = q[n]; ki = k[n]; vi = v[n]
            # ---- query step ----
            p = w; x = h
            while p != -1 and trans[n, p, qi] == -1:
                if ln[n, p] < x:
                    x = ln[n, p]
                p = link[n, p]
            if p != -1:
                p = trans[n, p, qi]; x = x + 1
            else:
                p = 0; x = 0
            if K > 0 and x > K:
                x = K
                vv = p
                while link[n, vv] != -1 and ln[n, link[n, vv]] >= x:
                    vv = link[n, vv]
                p = vv
            vv = p
            while link[n, vv] != -1 and ln[n, link[n, vv]] >= x:
                vv = link[n, vv]
            fs = -1; fl = 0; fp = -1
            while vv != -1:
                if ln[n, vv] > 0 and last[n, vv] >= 0:
                    r = last[n, vv]
                    rp = r + 1
                    fs = vi if rp == i else vh[n, rp]
                    mlim = (i if i < r else r) + 1
                    if mlim > LEN_CAP:
                        mlim = LEN_CAP
                    if K > 0 and mlim > K:
                        mlim = K
                    L = 0
                    while L < mlim:
                        qa = qi if (i - L) == i else qh[n, i - L]
                        if qa != kh[n, r - L]:
                            break
                        L += 1
                    fl = L; fp = rp
                    break
                vv = link[n, vv]
            out_sym[n] = fs; out_len[n] = fl; out_pos[n] = fp
            w = p; h = x
            # ---- key step: SAM-extend ----
            j = z; z += 1
            ln[n, j] = ln[n, g] + 1
            p = g
            while p != -1 and trans[n, p, ki] == -1:
                trans[n, p, ki] = j; p = link[n, p]
            if p == -1:
                link[n, j] = 0
            else:
                d = trans[n, p, ki]
                if ln[n, p] + 1 == ln[n, d]:
                    link[n, j] = d
                else:
                    b = z; z += 1
                    for a in range(A):
                        trans[n, b, a] = trans[n, d, a]
                    ln[n, b] = ln[n, p] + 1
                    link[n, b] = link[n, d]; last[n, b] = last[n, d]
                    while p != -1 and trans[n, p, ki] == d:
                        trans[n, p, ki] = b; p = link[n, p]
                    link[n, d] = b; link[n, j] = b
            g = j
            vv = g
            while vv != -1 and last[n, vv] < i:
                last[n, vv] = i; vv = link[n, vv]
            # ---- histories ----
            qh[n, i] = qi; kh[n, i] = ki; vh[n, i] = vi
            meta[n, 0] = g; meta[n, 1] = w; meta[n, 2] = h
            meta[n, 3] = i + 1; meta[n, 4] = z

    @njit(nogil=True, parallel=True, cache=True)
    def _sam_token_batch_nb(ids, A, out_sym, out_len, out_pos):
        N = ids.shape[0]
        for n in prange(N):
            _sam_qkv_one(ids[n], ids[n], ids[n], A, 0,
                         out_sym[n], out_len[n], out_pos[n])

    @njit(nogil=True, parallel=True, cache=True)
    def _cf_phi_batch_nb(qsym, ksym, vsym, A, K, tau, q_flat, k_flat, v_flat,
                         gy_flat, e0_all, e1_all, G, nbits,
                         gq_flat, gk_flat, gv_flat):
        """e0_all/e1_all: float32 [G, nbits] per-group payloads."""
        N = qsym.shape[0]
        for n in prange(N):
            g = n % G
            _cf_phi_one(qsym[n], ksym[n], vsym[n], A, K, tau,
                        q_flat[n], k_flat[n], v_flat[n], gy_flat[n],
                        e0_all[g], e1_all[g], nbits,
                        gq_flat[n], gk_flat[n], gv_flat[n])

    _NUMBA_KERNELS = True
else:  # pragma: no cover
    _sam_qkv_one = _sam_qkv_batch_nb = _sam_token_batch_nb = None
    _cf_phi_one = _cf_phi_batch_nb = _phi_payload = None
    _sam_qkv_one_m = _sam_qkv_batch_nb_m = None
    _sam_qkv_step_rows_nb = None
    _NUMBA_KERNELS = False


def _kernels_ok() -> bool:
    return _NUMBA_KERNELS


# ----------------------------------------------------------------------------
# 1.3  Engine dispatch: torch/numpy in -> retrieval indices out
# ----------------------------------------------------------------------------

def rosa_token_batch_np(ids: np.ndarray, vocab: int):
    """Plain ROSA over ids [N,T] (int).  Returns int32 (sym, ln) [N,T]."""
    ids = np.ascontiguousarray(ids.astype(np.int32))
    N, T = ids.shape
    if _NUMBA_KERNELS and vocab <= 4096:
        sym = np.empty((N, T), np.int32); ln = np.empty((N, T), np.int32)
        pos = np.empty((N, T), np.int32)
        _sam_token_batch_nb(ids, vocab, sym, ln, pos)
        return sym, ln
    return rosa_token_ref_batch(ids)


def rosa_qkv_batch_np(qsym: np.ndarray, ksym: np.ndarray, vsym: np.ndarray,
                      alphabet: int = 16, K: int = 0):
    """QKV-ROSA over symbol streams [N,T].  -> (sym, ln, pos) int32 [N,T]."""
    qsym = np.ascontiguousarray(qsym.astype(np.int32))
    ksym = np.ascontiguousarray(ksym.astype(np.int32))
    vsym = np.ascontiguousarray(vsym.astype(np.int32))
    N, T = qsym.shape
    if _NUMBA_KERNELS and alphabet <= 4096:
        sym = np.empty((N, T), np.int32); ln = np.empty((N, T), np.int32)
        pos = np.empty((N, T), np.int32)
        _sam_qkv_batch_nb(qsym, ksym, vsym, alphabet, K, sym, ln, pos)
        return sym, ln, pos
    return rosa_qkv_ref_batch(qsym, ksym, vsym)


def rosa_qkv_batch_np_m(qsym, ksym, vsym, alphabet: int = 16, K: int = 0,
                        M: int = 4):
    """Multi-candidate QKV-ROSA over [N,T] symbol streams.
    -> (sym, ln, pos) int32 [N,M,T], longest match first."""
    qsym = np.ascontiguousarray(qsym.astype(np.int32))
    ksym = np.ascontiguousarray(ksym.astype(np.int32))
    vsym = np.ascontiguousarray(vsym.astype(np.int32))
    N, T = qsym.shape
    sym = np.full((N, M, T), -1, np.int32)
    ln = np.zeros((N, M, T), np.int32)
    pos = np.full((N, M, T), -1, np.int32)
    if _NUMBA_KERNELS and alphabet <= 4096:
        _sam_qkv_batch_nb_m(qsym, ksym, vsym, alphabet, K, M, sym, ln, pos)
        return sym, ln, pos
    for n in range(N):
        s_, l_, p_, _nc = _rosa_qkv_ref_full_m(
            [int(t) for t in qsym[n]], [int(t) for t in ksym[n]],
            [int(t) for t in vsym[n]], M)
        sym[n] = s_; ln[n] = l_; pos[n] = p_
    return sym, ln, pos


def rosa_qkv_batch_torch_m(qsym: Tensor, ksym: Tensor, vsym: Tensor,
                           alphabet: int = 16, K: int = 0, M: int = 4):
    """Torch wrapper of rosa_qkv_batch_np_m: [B,G,T] -> [B,G,M,T]."""
    dev = qsym.device
    B, G, T = qsym.shape
    q = qsym.detach().to("cpu").numpy().reshape(B * G, T)
    kk = ksym.detach().to("cpu").numpy().reshape(B * G, T)
    v = vsym.detach().to("cpu").numpy().reshape(B * G, T)
    sym, ln, pos = rosa_qkv_batch_np_m(q, kk, v, alphabet, K, M)
    shp = (B, G, M, T)
    return (torch.from_numpy(sym).view(shp).to(dev),
            torch.from_numpy(ln).view(shp).to(dev),
            torch.from_numpy(pos).view(shp).to(dev))


def rosa_qkv_batch_gpu(qsym: Tensor, ksym: Tensor, vsym: Tensor,
                       alphabet: int = 16, K: int = 0, Lcap: int = 64,
                       row_chunk: int = 4):
    """v2: EXACT-window QKV-ROSA on ANY device (GPU-friendly, no automaton).

    Strided window equality: maintains eq_L[t, s] = AND_{i<L} q[t-i]==k[s-i]
    incrementally (eq_L = eq_1 & shift(eq_{L-1})), so matches are EXACT for
    every length up to Lcap — no hashing, no collisions.  For each t the
    longest L with any earlier key-window match wins; within that L the most
    recent end s wins.  This reproduces the SAM retrieval semantics (endpos
    classes share occurrence ends) for all matches of length <= Lcap; longer
    matches fall back to the best shorter one (the numba engine caps its
    recomputed length at LEN_CAP=64 as well).

    qsym/ksym/vsym: [R,T] int tensors (already B*G-flattened), same device.
    Returns (sym, ln, pos) int32 [R,T] on the input device.
    Memory: row_chunk*T*T bool — lower row_chunk if you OOM.
    """
    dev = qsym.device
    R, T = qsym.shape
    q = qsym.long(); k = ksym.long(); v = vsym.long()
    Lmax = min(Lcap, (K if K > 0 else Lcap), T)
    best_ln = torch.zeros(R, T, dtype=torch.int32, device=dev)
    best_pos = torch.full((R, T), -1, dtype=torch.int32, device=dev)
    if Lmax <= 0:
        return best_ln.new_full((R, T), -1), best_ln, best_pos
    s_idx = torch.arange(T, device=dev)
    causal = (s_idx[None, :] < s_idx[:, None])          # [T,T]  s < t
    for r0 in range(0, R, row_chunk):
        r1 = min(r0 + row_chunk, R)
        eq1 = (q[r0:r1, :, None] == k[r0:r1, None, :])  # [r,T,T] q[t]==k[s]
        eqL = eq1.clone()
        for L in range(1, Lmax + 1):
            ok_pos = (s_idx >= L - 1)                   # window fully in range
            mask = causal[None] & ok_pos[None, :, None] & ok_pos[None, None, :]
            eqm = eqL & mask
            has = eqm.any(-1)                           # [r,T]
            ridx = torch.where(eqm, s_idx[None, None, :],
                               torch.tensor(-1, device=dev)).amax(-1)
            upd = has & (L > best_ln[r0:r1])
            best_ln[r0:r1] = torch.where(
                upd, torch.full_like(best_ln[r0:r1], L), best_ln[r0:r1])
            best_pos[r0:r1] = torch.where(
                upd, (ridx + 1).to(torch.int32), best_pos[r0:r1])
            if L < Lmax:                                # grow the window by 1
                nxt = torch.zeros_like(eqL)
                nxt[:, 1:, 1:] = eqL[:, :-1, :-1] & eq1[:, 1:, 1:]
                eqL = nxt
    valid = best_ln > 0
    vp = v.gather(1, best_pos.long().clamp_min(0))
    sym = torch.where(valid, vp, torch.tensor(-1, device=dev))
    return sym.to(torch.int32), best_ln, best_pos


def rosa_token_batch_gpu(idx: Tensor, Lcap: int = 16, row_chunk: int = 4):
    """v2: plain token-ROSA via the GPU window matcher.  idx [B,T] ->
    (sym, ln) [B,T] — same contract as rosa_token_batch_torch."""
    sym, ln, _pos = rosa_qkv_batch_gpu(idx, idx, idx,
                                       alphabet=int(idx.max().item()) + 1
                                       if idx.numel() else 2,
                                       K=0, Lcap=Lcap, row_chunk=row_chunk)
    return sym, ln


def rosa_token_batch_torch(idx: Tensor):
    """Plain ROSA over a torch LongTensor [B,T] (any device).  Returns
    (sym, ln) int32 tensors on idx.device.  sym == -1 means "novel suffix"."""
    vocab = int(idx.max().item()) + 1 if idx.numel() else 1
    dev = idx.device
    ids = idx.detach().to("cpu", torch.int32).numpy()
    sym, ln = rosa_token_batch_np(ids, vocab)
    return (torch.from_numpy(sym).to(dev), torch.from_numpy(ln).to(dev))


def rosa_qkv_batch_torch(qsym: Tensor, ksym: Tensor, vsym: Tensor,
                         alphabet: int = 16, K: int = 0):
    """QKV-ROSA over torch symbol tensors [B,G,T] (uint8/int).  -> (sym, ln,
    pos) int32 tensors on the input device, same [B,G,T] layout."""
    dev = qsym.device
    B, G, T = qsym.shape
    q = qsym.detach().to("cpu").numpy().reshape(B * G, T)
    kk = ksym.detach().to("cpu").numpy().reshape(B * G, T)
    v = vsym.detach().to("cpu").numpy().reshape(B * G, T)
    sym, ln, pos = rosa_qkv_batch_np(q, kk, v, alphabet, K)
    return (torch.from_numpy(sym).view(B, G, T).to(dev),
            torch.from_numpy(ln).view(B, G, T).to(dev),
            torch.from_numpy(pos).view(B, G, T).to(dev))


def pack_symbols(x: Tensor, nbits: int) -> Tensor:
    """(x > 0) per channel, packed into nbits-wide symbols.  [B,T,C] -> [B,T,G]
    little-endian within each group (channel g*nbits + b -> symbol bit b)."""
    B, T, C = x.shape
    assert C % nbits == 0, f"C={C} not divisible by nbits={nbits}"
    bits = (x > 0).to(torch.uint8).view(B, T, C // nbits, nbits)
    shift = torch.arange(nbits, device=x.device, dtype=torch.uint8)
    sym = (bits << shift).sum(-1)  # [B, T, G] uint8
    return sym


# ----------------------------------------------------------------------------
# 1.35  Batched live automata — v2.  Steps N independent QKV suffix automata
#       (or token automata, via q==k==v) in ONE numba kernel call, replacing
#       the Python `for g in range(G)` / `for b in range(B)` loops of RNN
#       mode.  Falls back to per-object StatefulQKVSam without numba.
# ----------------------------------------------------------------------------

class BatchedQKVSams:
    """N independent live QKV suffix automata, flat-array backed.

    step(q_syms, k_syms, v_syms) advances every automaton by one symbol and
    returns (sym, ln, pos) int32 [N] — identical per-row results to
    StatefulQKVSam.extend (validated in test_rwkv8.py).  Storage grows
    geometrically; smax bounds states (SAM needs <= 2T+2 for a T-step stream).
    """

    def __init__(self, N: int, alphabet: int = 16, K: int = 0,
                 smax: int = 2048, tmax: int = 4096, seed_from=None):
        self.N = N; self.A = int(alphabet); self.K = int(K)
        self.smax = int(smax); self.tmax = int(tmax)
        self.trans = np.full((N, smax, self.A), -1, np.int32)
        self.link = np.full((N, smax), -1, np.int32)
        self.ln = np.zeros((N, smax), np.int32)
        self.last = np.full((N, smax), -1, np.int32)
        self.meta = np.zeros((N, 5), np.int64)      # g, w, h, n, z
        self.meta[:, 4] = 1                          # z: root allocated
        self.qh = np.zeros((N, tmax), np.int32)
        self.kh = np.zeros((N, tmax), np.int32)
        self.vh = np.zeros((N, tmax), np.int32)
        self._fallback = ([StatefulQKVSam(self.A) for _ in range(N)]
                          if not _NUMBA_KERNELS else None)

    # ---- growth ----
    def _grow_states(self):
        add = self.smax
        self.trans = np.concatenate(
            [self.trans, np.full((self.N, add, self.A), -1, np.int32)], axis=1)
        self.link = np.concatenate(
            [self.link, np.full((self.N, add), -1, np.int32)], axis=1)
        self.ln = np.concatenate(
            [self.ln, np.zeros((self.N, add), np.int32)], axis=1)
        self.last = np.concatenate(
            [self.last, np.full((self.N, add), -1, np.int32)], axis=1)
        self.smax += add

    def _grow_hist(self):
        add = self.tmax
        self.qh = np.concatenate([self.qh, np.zeros((self.N, add), np.int32)],
                                 axis=1)
        self.kh = np.concatenate([self.kh, np.zeros((self.N, add), np.int32)],
                                 axis=1)
        self.vh = np.concatenate([self.vh, np.zeros((self.N, add), np.int32)],
                                 axis=1)
        self.tmax += add

    # ---- stepping ----
    def step(self, q, k, v):
        """q/k/v: list/array/ints of length N.  -> (sym, ln, pos) [N]."""
        q = np.ascontiguousarray(np.asarray(q, np.int32))
        k = np.ascontiguousarray(np.asarray(k, np.int32))
        v = np.ascontiguousarray(np.asarray(v, np.int32))
        if self._fallback is not None:
            out = [sam.extend(int(q[n]), int(k[n]), int(v[n]))
                   for n, sam in enumerate(self._fallback)]
            a, b_, c = (np.array(x, np.int32) for x in zip(*out))
            return a, b_, c
        if self.meta[:, 4].max() + 8 > self.smax:
            self._grow_states()
        if self.meta[:, 3].max() + 2 > self.tmax:
            self._grow_hist()
        os_ = np.empty(self.N, np.int32); ol = np.empty(self.N, np.int32)
        op_ = np.empty(self.N, np.int32)
        _sam_qkv_step_rows_nb(self.trans, self.link, self.ln, self.last,
                              self.meta, self.qh, self.kh, self.vh,
                              q, k, v, self.A, self.K, self.smax, self.tmax,
                              os_, ol, op_)
        return os_, ol, op_

    # ---- conversion to/from per-object automata (state IO) ----
    def to_list(self) -> List[StatefulQKVSam]:
        sams = []
        for n in range(self.N):
            z = int(self.meta[n, 4]); i = int(self.meta[n, 3])
            s = StatefulQKVSam(self.A)
            s.trans = [{int(a): int(self.trans[n, st, a])
                        for a in range(self.A)
                        if self.trans[n, st, a] >= 0}
                       for st in range(z)]
            s.link = [int(self.link[n, st]) for st in range(z)]
            s.ln = [int(self.ln[n, st]) for st in range(z)]
            s.last = [int(self.last[n, st]) for st in range(z)]
            s.g = int(self.meta[n, 0]); s.w = int(self.meta[n, 1])
            s.h = int(self.meta[n, 2]); s.n = i
            s._vhist = [int(x) for x in self.vh[n, :i]]
            s._qhist = [int(x) for x in self.qh[n, :i]]
            s._khist = [int(x) for x in self.kh[n, :i]]
            sams.append(s)
        return sams

    @classmethod
    def from_list(cls, sams: List[StatefulQKVSam], K: int = 0,
                  smax: int = 2048, tmax: int = 4096) -> "BatchedQKVSams":
        N = len(sams)
        A = sams[0].A
        need_s = max(max(len(s.trans) for s in sams), 2)
        need_t = max(max(s.n for s in sams), 1)
        obj = cls(N, A, K, smax=max(smax, 2 * need_s), tmax=max(tmax, need_t + 2))
        for n, s in enumerate(sams):
            z = len(s.trans)
            for st, d in enumerate(s.trans):
                for a, tgt in d.items():
                    obj.trans[n, st, a] = tgt
            obj.link[n, :z] = s.link
            obj.ln[n, :z] = s.ln
            obj.last[n, :z] = s.last
            obj.meta[n] = (s.g, s.w, s.h, s.n, z)
            obj.qh[n, :s.n] = s._qhist
            obj.kh[n, :s.n] = s._khist
            obj.vh[n, :s.n] = s._vhist
        return obj

    # ---- serialization ----
    def dump(self) -> Dict:
        z = int(self.meta[:, 4].max()); i = int(self.meta[:, 3].max())
        return {"A": self.A, "K": self.K, "N": self.N,
                "trans": self.trans[:, :z].copy(), "link": self.link[:, :z].copy(),
                "ln": self.ln[:, :z].copy(), "last": self.last[:, :z].copy(),
                "meta": self.meta.copy(), "qh": self.qh[:, :i].copy(),
                "kh": self.kh[:, :i].copy(), "vh": self.vh[:, :i].copy()}

    @classmethod
    def load(cls, d: Dict) -> "BatchedQKVSams":
        z = d["trans"].shape[1]; i = d["qh"].shape[1]
        obj = cls(d["N"], d["A"], d["K"], smax=max(2048, 2 * z),
                  tmax=max(4096, i + 2))
        obj.trans[:, :z] = d["trans"]; obj.link[:, :z] = d["link"]
        obj.ln[:, :z] = d["ln"]; obj.last[:, :z] = d["last"]
        obj.meta[:] = d["meta"]
        obj.qh[:, :i] = d["qh"]; obj.kh[:, :i] = d["kh"]
        obj.vh[:, :i] = d["vh"]
        return obj


# ----------------------------------------------------------------------------
# 1.4  Stateful suffix automata — live RNN-mode inference (O(1) per token)
# ----------------------------------------------------------------------------

class StatefulTokenSam:
    """Incremental plain-ROSA automaton for one stream (RNN-mode inference).

    extend(tok) appends tok and returns (prediction, match_len) for the
    position just appended: prediction = the token that followed the longest
    repeated suffix (-1 if none), match_len = its length (0 if none).
    """

    def __init__(self, vocab: int):
        self.vocab = vocab
        self.trans: List[Dict[int, int]] = [{}]
        self.link = [-1]
        self.ln = [0]
        self.last = [-1]
        self.g = 0
        self.n = 0
        self._hist: List[int] = []

    def extend(self, tok: int) -> int:
        i = self.n
        r = len(self.trans)
        self.trans.append({}); self.link.append(-1)
        self.ln.append(self.ln[self.g] + 1); self.last.append(-1)
        p = self.g
        while p != -1 and tok not in self.trans[p]:
            self.trans[p][tok] = r; p = self.link[p]
        if p == -1:
            self.link[r] = 0
        else:
            q = self.trans[p][tok]
            if self.ln[p] + 1 == self.ln[q]:
                self.link[r] = q
            else:
                u = len(self.trans)
                self.trans.append(dict(self.trans[q]))
                self.link.append(self.link[q]); self.ln.append(self.ln[p] + 1)
                self.last.append(self.last[q])
                while p != -1 and self.trans[p].get(tok) == q:
                    self.trans[p][tok] = u; p = self.link[p]
                self.link[q] = u; self.link[r] = u
        v = self.g = r; a = -1; mlen = 0
        while v != -1:
            if self.ln[v] > 0 and self.last[v] >= 0:
                # last[v]+1 can be i (current position): reading the token
                # that follows at the current position means "the suffix also
                # occurred ending at i-1" — use the streamed-in token.
                rp = self.last[v] + 1
                a = tok if rp == i else self._hist[rp]
                rr = rp - 1
                lim = min(i, rr, 63) + 1
                L = 0
                while L < lim:
                    aa = i - L
                    ta = tok if aa == i else self._hist[aa]
                    if ta != self._hist[rr - L]:
                        break
                    L += 1
                mlen = L
                break
            v = self.link[v]
        y = a; v = self.g
        while v != -1 and self.last[v] < i:
            self.last[v] = i; v = self.link[v]
        self._hist.append(tok)
        self.n += 1
        return y, mlen


class StatefulQKVSam:
    """Incremental QKV-ROSA automaton for one channel-group (RNN-mode).

    extend(q_sym, k_sym, v_sym) -> (retrieved_sym, match_len, read_pos);
    read_pos == -1 means "no match".  Values are kept internally for later
    reads, so simply stream (q,k,v) symbols in order.
    """

    def __init__(self, alphabet: int = 16):
        self.A = alphabet
        self.trans: List[Dict[int, int]] = [{}]
        self.link = [-1]
        self.ln = [0]
        self.last = [-1]
        self.g = 0
        self.w = 0   # carried query-match state
        self.h = 0   # carried query-match length
        self.n = 0
        self._vhist: List[int] = []
        self._qhist: List[int] = []
        self._khist: List[int] = []

    def extend(self, q: int, k: int, v: int):
        i = self.n
        p, x = self.w, self.h
        while p != -1 and q not in self.trans[p]:
            x = min(x, self.ln[p]); p = self.link[p]
        if p != -1:
            p = self.trans[p][q]; x = x + 1
        else:
            p, x = 0, 0
        vv = p
        while self.link[vv] != -1 and self.ln[self.link[vv]] >= x:
            vv = self.link[vv]
        ret, mlen, rpos = -1, 0, -1
        while vv != -1:
            if self.ln[vv] > 0 and self.last[vv] >= 0:
                rpos = self.last[vv] + 1
                # rpos can equal i (the current position): reading the value
                # of the current token is causal (it is a function of tokens
                # <= i, used to predict i+1) — use the streamed-in value.
                ret = v if rpos == i else self._vhist[rpos]
                rr = rpos - 1
                lim = min(i, rr, 63) + 1
                L = 0
                while L < lim:
                    a = i - L
                    qa = q if a == i else self._qhist[a]  # pos i not yet stored
                    if qa != self._khist[rr - L]:
                        break
                    L += 1
                mlen = L
                break
            vv = self.link[vv]
        self.w, self.h = p, x
        # insert key symbol
        j = len(self.trans)
        self.trans.append({}); self.ln.append(self.ln[self.g] + 1)
        self.link.append(-1); self.last.append(-1)
        p = self.g
        while p != -1 and k not in self.trans[p]:
            self.trans[p][k] = j; p = self.link[p]
        if p == -1:
            self.link[j] = 0
        else:
            d = self.trans[p][k]
            if self.ln[p] + 1 == self.ln[d]:
                self.link[j] = d
            else:
                u = len(self.trans)
                self.trans.append(dict(self.trans[d]))
                self.ln.append(self.ln[p] + 1)
                self.link.append(self.link[d]); self.last.append(self.last[d])
                while p != -1 and self.trans[p].get(k) == d:
                    self.trans[p][k] = u; p = self.link[p]
                self.link[d] = u; self.link[j] = u
        self.g = j
        vv = self.g
        while vv != -1 and self.last[vv] < i:
            self.last[vv] = i; vv = self.link[vv]
        self._qhist.append(q)
        self._khist.append(k)
        self._vhist.append(v)
        self.n += 1
        return ret, mlen, rpos

# ==========================================================================
# >>> section from p2_layers.py
# ==========================================================================

# ============================================================================
# Section 2 — Neural layers
#   2.1  RosaTokenEmb   : Emb(ROSA(idx)) with NULL payload + length buckets
#   2.2  RosaQKV        : QKV-ROSA branch (exact forward, hybrid backward)
#   2.3  RWKV-7g scan   : exact pure-PyTorch port of wkv7_cuda.cu
#   2.4  RWKV_Tmix_x070 : the full RWKV-7 time-mix block (from [1])
#   2.5  FFN + DeepEmbed
# ============================================================================


# ----------------------------------------------------------------------------
# 2.1  Token-level ROSA embedding  —  "Add Emb(ROSA(x)) to suitable tensors"
#      (the kNN+LLM / RAG trick of [1]); NULL payload from [2]; length
#      calibration from [7].  Routing depends only on token ids, so this is
#      exactly differentiable w.r.t. all learned tables, and its indices can
#      be prefetched on a CPU thread while the GPU trains the previous batch
#      (the async pipeline of [5]).
# ----------------------------------------------------------------------------

class RosaTokenEmb(nn.Module):
    """out[t] = emb(ROSA(x)[t]) when the longest repeated suffix has a known
    continuation, else a learned NULL embedding; plus a match-length bucket
    embedding (how confident the retrieval is).
    v2: backend="gpu" computes the retrieval on-device via the exact-window
    matcher (no CPU round-trip); backend="cpu" keeps the SAM engine and the
    async prefetch pipeline."""

    def __init__(self, vocab: int, dim: int, len_buckets: int = 32,
                 backend: str = "cpu", gpu_cap: int = 16):
        super().__init__()
        self.vocab = vocab
        self.len_buckets = len_buckets
        self.backend = backend
        self.gpu_cap = gpu_cap
        self.emb = nn.Embedding(vocab, dim)
        self.null_emb = nn.Parameter(torch.zeros(dim))
        self.len_emb = nn.Embedding(len_buckets, dim)
        nn.init.zeros_(self.emb.weight)          # start silent; learn to use
        nn.init.zeros_(self.len_emb.weight)

    def forward(self, idx: Tensor, rosa_sym: Optional[Tensor] = None,
                rosa_ln: Optional[Tensor] = None) -> Tensor:
        if rosa_sym is None or rosa_ln is None:
            if self.backend == "gpu":
                rosa_sym, rosa_ln = rosa_token_batch_gpu(
                    idx, Lcap=self.gpu_cap)
            else:
                rosa_sym, rosa_ln = rosa_token_batch_torch(idx)
        nomatch = rosa_sym < 0                                   # [B,T]
        lookup = rosa_sym.clamp_min(0)
        out = self.emb(lookup) * (~nomatch).unsqueeze(-1)
        out = out + nomatch.unsqueeze(-1) * self.null_emb
        bucket = rosa_ln.clamp(0, self.len_buckets - 1)
        out = out + self.len_emb(bucket)
        return out


# ----------------------------------------------------------------------------
# 2.2  QKV-ROSA branch
#      Forward: EXACT suffix-automaton retrieval over nbits-symbol groups
#      (semantics of samx_qkv_slow / rosa_slow_ref in [1]; n-bit codes and
#      separate no-match payload emb0 from [2]).
#      Backward (hybrid, [3]-inspired):
#        * emb0 / emb1 ............ exact
#        * V (value) bits ......... exact given the routing: gradient is
#                                   scattered through the retrieval read
#                                   position (dense surrogate of [3])
#        * Q/K (routing) bits ..... straight-through surrogate by default;
#                                   grad_mode="exact" runs BlinkDL's
#                                   flip-every-bit counterfactual VJP [1,2]
# ----------------------------------------------------------------------------

_ROSA_POOL: Optional[ThreadPoolExecutor] = None


def _rosa_pool() -> ThreadPoolExecutor:
    """Single background worker for ROSA retrieval (kernels release the GIL,
    so this truly overlaps GPU compute — the async pipeline of [5])."""
    global _ROSA_POOL
    if _ROSA_POOL is None:
        _ROSA_POOL = ThreadPoolExecutor(max_workers=1)
    return _ROSA_POOL


def _ste_cf(x: Tensor, tau: float) -> Tensor:
    """Counterfactual step-width estimator (BlinkDL [1]): the finite
    difference (phi(bit=1) - phi(bit=0)) / (2 * max(|x|, tau)) corresponds
    to treating the bit's flip distance as 2*max(|x|, tau), i.e.
    d 1[x>0] / dx ~= 1 / (2 * max(|x|, tau)).  Using this estimator makes
    the analytic value-read gradient EXACTLY equal to the counterfactual
    VJP given the routing (validated in the test suite / notebook)."""
    return 1.0 / (2.0 * x.abs().clamp_min(tau))


class RosaQKVFunction(torch.autograd.Function):
    """payload[b,t,c] = emb0[c]                     (no match in this group)
                       sign(bit_c of retrieved symbol) * emb1[c]  (match)
    """

    @staticmethod
    def forward(ctx, q, k, v, sym, ln, pos, emb0, emb1, nbits, grad_mode,
                tau):
        B, T, C = q.shape
        G = C // nbits
        dev = q.device
        symL = sym.long()                                       # [B,G,T]
        match = (symL >= 0)                                     # [B,G,T]
        shifts = torch.arange(nbits, device=dev).view(1, 1, 1, nbits)
        bits = ((symL.clamp_min(0).unsqueeze(-1) >> shifts) & 1).to(q.dtype)
        sgn = 2.0 * bits - 1.0                                  # [B,G,T,nb]
        e1 = emb1.view(1, G, 1, nbits)
        e0 = emb0.view(1, G, 1, nbits)
        payload = torch.where(match.unsqueeze(-1), sgn * e1,
                              e0.expand(B, G, T, nbits))
        out = payload.permute(0, 2, 1, 3).reshape(B, T, C)
        ctx.save_for_backward(q, k, v, sym, pos, emb0, emb1)
        ctx.nbits = nbits
        ctx.grad_mode = grad_mode
        ctx.tau = tau
        return out

    @staticmethod
    def backward(ctx, gy):
        q, k, v, sym, pos, emb0, emb1 = ctx.saved_tensors
        nbits, tau = ctx.nbits, ctx.tau
        B, T, C = q.shape
        G = C // nbits
        dev = q.device

        symL = sym.long()
        bits = ((symL.clamp_min(0).unsqueeze(-1) >>
                 torch.arange(nbits, device=dev).view(1, 1, 1, nbits)) & 1)\
            .to(gy.dtype)
        sgn_g = 2.0 * bits - 1.0                                # [B,G,T,nb]
        match_g = (symL >= 0).to(gy.dtype)                      # [B,G,T]
        nomatch_g = 1.0 - match_g
        gy_g = gy.view(B, T, G, nbits).permute(0, 2, 1, 3)      # [B,G,T,nb]

        # ---- exact gradients for the payload embeddings -------------------
        g_emb0 = (gy_g * nomatch_g.unsqueeze(-1)).permute(0, 2, 1, 3)\
            .reshape(B, T, C).sum(dim=(0, 1))
        g_emb1 = (gy_g * sgn_g * match_g.unsqueeze(-1)).permute(0, 2, 1, 3)\
            .reshape(B, T, C).sum(dim=(0, 1))

        # ---- V gradient: exact given the routing (scatter through reads) --
        # payload[t,c] = (2*1[v[pos,c]>0] - 1) * emb1[c]
        # d payload[t,c] / d v[pos[t],c] = 2 * emb1[c] * ste_cf(v[pos,c])
        # (no payload-sign factor: the derivative of (2H-1) is +2H' always)
        posc = pos.clamp_min(0).long()                          # [B,G,T]
        v_g = v.view(B, T, G, nbits).permute(0, 2, 1, 3)        # [B,G,T,nb]
        v_read = v_g.gather(
            2, posc.unsqueeze(-1).expand(B, G, T, nbits))       # value read
        e1_g = emb1.view(1, G, 1, nbits)
        contrib = gy_g * 2.0 * e1_g * _ste_cf(v_read, tau) \
            * match_g.unsqueeze(-1)
        gv_g = torch.zeros_like(v_g)
        gv_g.scatter_add_(
            2, posc.unsqueeze(-1).expand(B, G, T, nbits), contrib)
        gv = gv_g.permute(0, 2, 1, 3).reshape(B, T, C)

        # ---- Q/K gradients -------------------------------------------------
        if ctx.grad_mode == "none":
            # reference-LM behaviour: no gradient into the code streams
            gq = gk = None
        elif ctx.grad_mode == "exact":
            if _cf_phi_batch_nb is None:
                raise RuntimeError(
                    "grad_mode='exact' requires numba (counterfactual kernel)")
            gq, gk = _rosa_qkv_exact_qk_grad(
                q, k, v, sym, pos, gy, emb0, emb1, nbits, tau)
        else:  # "ste": local surrogate (each position as if reading itself)
            e1_c = emb1.view(1, 1, C)
            gq = gy * 2.0 * e1_c * _ste_cf(q, tau)
            gk = gy * 2.0 * e1_c * _ste_cf(k, tau)
        gv = None if ctx.grad_mode == "none" else gv
        return (gq, gk, gv, None, None, None, g_emb0, g_emb1,
                None, None, None)


class RosaSoftQKVFunction(torch.autograd.Function):
    """v2: soft routing over the TOP-M suffix-link match candidates.

    payload[t] = sum_m p_m[t] * sgn(bits(sym_m[t])) * emb1   (match cand.)
                 (or emb0 when the position has NO candidate at all)
    p = softmax_m(len_scale * ln_m + len_bias) over valid candidates.

    Gradients: emb0/emb1 exact; V-bits scattered through every candidate's
    read position, weighted by p_m (exact given routing+weights); routing
    bits (q/k) get the same STE surrogate as the hard mode; len_scale /
    len_bias receive the exact softmax backward through the scores.
    Candidate 0 is the hard-mode retrieval, so this strictly generalizes it.
    """

    @staticmethod
    def forward(ctx, q, k, v, sym, ln, pos, emb0, emb1, nbits, tau,
                len_scale, len_bias):
        B, T, C = q.shape
        G = C // nbits
        M = sym.shape[2]
        dev = q.device
        symL = sym.long()                                       # [B,G,M,T]
        matchM = (symL >= 0)                                    # [B,G,M,T]
        shifts = torch.arange(nbits, device=dev).view(1, 1, 1, 1, nbits)
        bitsM = ((symL.clamp_min(0).unsqueeze(-1) >> shifts) & 1).to(q.dtype)
        sgnM = 2.0 * bitsM - 1.0                                # [B,G,M,T,nb]
        e1 = emb1.view(1, G, 1, 1, nbits)
        payloadM = (sgnM * e1) * matchM.unsqueeze(-1).to(q.dtype)
        # ---- softmax weights over candidates (masked to valid) ----------
        score = ln.float() * len_scale + len_bias               # [B,G,M,T]
        neg_inf = torch.finfo(q.dtype).min
        score = torch.where(matchM, score, torch.full_like(score, neg_inf))
        p = torch.softmax(score, dim=2)                         # over M
        any_match = matchM.any(dim=2, keepdim=True)             # [B,G,1,T]
        p = p * any_match.to(q.dtype)
        payload = (payloadM * p.unsqueeze(-1)).sum(dim=2)       # [B,G,T,nb]
        nomatch = (~any_match.squeeze(2))                       # [B,G,T]
        e0 = emb0.view(1, G, 1, nbits)
        payload = torch.where(nomatch.unsqueeze(-1),
                              e0.expand_as(payload), payload)
        out = payload.permute(0, 2, 1, 3).reshape(B, T, C)
        ctx.save_for_backward(q, k, v, sym, ln, pos, emb0, emb1, len_scale,
                              len_bias)
        ctx.nbits = nbits
        ctx.tau = tau
        return out

    @staticmethod
    def backward(ctx, gy):
        q, k, v, sym, ln, pos, emb0, emb1, len_scale, len_bias = \
            ctx.saved_tensors
        nbits, tau = ctx.nbits, ctx.tau
        B, T, C = q.shape
        G = C // nbits
        M = sym.shape[2]
        dev = q.device
        symL = sym.long()
        matchM = (symL >= 0)
        shifts = torch.arange(nbits, device=dev).view(1, 1, 1, 1, nbits)
        bitsM = ((symL.clamp_min(0).unsqueeze(-1) >> shifts) & 1).to(gy.dtype)
        sgnM = 2.0 * bitsM - 1.0
        e1 = emb1.view(1, G, 1, 1, nbits).to(gy.dtype)
        gy_g = gy.view(B, T, G, nbits).permute(0, 2, 1, 3)      # [B,G,T,nb]
        payloadM = (sgnM * e1) * matchM.unsqueeze(-1).to(gy.dtype)
        score = ln.float().to(gy.dtype) * len_scale.to(gy.dtype) \
            + len_bias.to(gy.dtype)
        neg_inf = torch.finfo(gy.dtype).min
        score = torch.where(matchM, score, torch.full_like(score, neg_inf))
        p = torch.softmax(score, dim=2)
        any_match = matchM.any(dim=2, keepdim=True)
        p = p * any_match.to(gy.dtype)
        nomatch = (~any_match.squeeze(2))

        # ---- emb0 / emb1 ---------------------------------------------------
        g_emb0 = (gy_g * nomatch.unsqueeze(-1).to(gy.dtype))\
            .permute(0, 2, 1, 3).reshape(B, T, C).sum(dim=(0, 1))
        # d payload/d emb1 = sum_m p_m * sgn_m  (per channel c)
        g_emb1 = (gy_g.unsqueeze(2)
                  * (sgnM * matchM.unsqueeze(-1).to(gy.dtype)
                     * p.unsqueeze(-1))).sum(dim=2)          # [B,G,T,nb]
        g_emb1 = g_emb1.permute(0, 2, 1, 3).reshape(B, T, C).sum(dim=(0, 1))

        # ---- V: scatter through every candidate's read position ------------
        posM = pos.long().clamp_min(0)                          # [B,G,M,T]
        v_g = v.view(B, T, G, nbits).permute(0, 2, 1, 3)        # [B,G,T,nb]
        gv = torch.zeros_like(v_g)
        e1_4 = emb1.view(1, G, 1, nbits).to(gy.dtype)           # [1,G,1,nb]
        contrib_base = gy_g * 2.0 * e1_4 * _ste_cf(v_g, tau)    # [B,G,T,nb]
        for m in range(M):
            pm = p[:, :, m].unsqueeze(-1)                       # [B,G,T,1]
            posm = posM[:, :, m]                                # [B,G,T]
            valid = matchM[:, :, m].unsqueeze(-1).to(gy.dtype)
            contrib = contrib_base * pm * valid
            gv.scatter_add_(2, posm.unsqueeze(-1)
                            .expand(B, G, T, nbits), contrib)
        gv = gv.permute(0, 2, 1, 3).reshape(B, T, C)

        # ---- len_scale / len_bias: exact softmax backward ------------------
        # A_m[t] = <gy, payload_m> (per candidate "logit-gradient")
        A = (gy_g.unsqueeze(2) * payloadM).sum(-1)              # [B,G,M,T]
        Abar = (p * A).sum(dim=2, keepdim=True)
        d_score = p * (A - Abar)                                # [B,G,M,T]
        mask = matchM.to(gy.dtype)
        d_len_scale = (d_score * ln.to(gy.dtype) * mask).sum()
        d_len_bias = (d_score * mask).sum()

        # ---- q/k: STE surrogate as in the hard mode ------------------------
        e1_c = emb1.view(1, 1, C).to(gy.dtype)
        gq = gy * 2.0 * e1_c * _ste_cf(q, tau)
        gk = gy * 2.0 * e1_c * _ste_cf(k, tau)
        return (gq, gk, gv, None, None, None, g_emb0, g_emb1, None, None,
                d_len_scale, d_len_bias)


def _rosa_qkv_exact_qk_grad(q, k, v, sym, pos, gy, emb0, emb1, nbits, tau):
    """Exact counterfactual VJP for the Q/K routing bits (BlinkDL [1],
    practicalized as in [2]): flip every symbol bit of q/k, rerun that
    group's automaton, measure the downstream change.  O(T^2) per group —
    validation / short sequences only."""
    B, T, C = q.shape
    G = C // nbits
    dev = q.device
    to_np = lambda t: t.detach().to("cpu").numpy()
    qs = to_np(pack_symbols(q, nbits).permute(0, 2, 1).reshape(B * G, T))\
        .astype(np.int32)
    ks_ = to_np(pack_symbols(k, nbits).permute(0, 2, 1).reshape(B * G, T))\
        .astype(np.int32)
    vs_ = to_np(pack_symbols(v, nbits).permute(0, 2, 1).reshape(B * G, T))\
        .astype(np.int32)
    flat = lambda t: to_np(
        t.view(B, T, G, nbits).permute(0, 2, 1, 3).reshape(B * G, T * nbits))\
        .astype(np.float32)
    qf, kf, vf, gyf = flat(q), flat(k), flat(v), flat(gy)
    e0a = to_np(emb0.detach()).reshape(G, nbits).astype(np.float32)
    e1a = to_np(emb1.detach()).reshape(G, nbits).astype(np.float32)
    gqf = np.zeros_like(qf); gkf = np.zeros_like(kf)
    gvf = np.zeros_like(vf)
    _cf_phi_batch_nb(qs, ks_, vs_, 1 << nbits, 0, float(tau),
                     qf, kf, vf, gyf, e0a, e1a, G, nbits, gqf, gkf, gvf)
    to_t = lambda a: torch.from_numpy(a.copy()).view(B, G, T, nbits)\
        .permute(0, 2, 1, 3).reshape(B, T, C).to(dev)
    return to_t(gqf), to_t(gkf)


class RosaQKVBranch(nn.Module):
    """Parallel symbolic-retrieval branch (the RWKV-8 "Heron" ROSA block).

    precompute(xn)  -> projects q/k/v, binarizes to nbits symbols, and (by
                       default) launches the CPU suffix-automaton retrieval on
                       a background thread (async pipeline [5]).
    finalize(h)     -> waits for the retrieval, applies the differentiable
                       payload assembly (RosaQKVFunction), a learned read
                       gate [7], match-length buckets, and the output proj.

    The reference [1] block is x = x + timemix(x) + rosaQKV(x): run
    precompute BEFORE the time-mix so the CPU retrieval overlaps the GPU
    time-mix/FFN work.

    grad_mode:
      "ste"    default; hybrid exact/STE backward (see RosaQKVFunction)
      "soft"   v2: route over the top-M suffix-link candidates with a learned
               length prior (RosaSoftQKVFunction); GPT-mode only (RNN-mode
               falls back to the hard single-match path)
      "exact"  BlinkDL counterfactual VJP for q/k (validation, slow)
      "none"  reference-LM behaviour: NO gradient into the q/k/v code
               streams (payload treated as a constant, exactly like
               260222_rosa4bitLM_L12.py where the layer output is built from
               detached tensors); only o / gate / emb0 / emb1 / len_emb learn.
    """

    def __init__(self, dim: int, nbits: int = 4, K: int = 0,
                 grad_mode: str = "ste", tau: float = 1e-3,
                 use_async: bool = True, use_len_emb: bool = True,
                 len_buckets: int = 32, o_init_zero: bool = True,
                 gate_bias: float = 2.0, backend: str = "cpu",
                 gpu_cap: int = 64, soft_M: int = 4):
        super().__init__()
        assert dim % nbits == 0
        self.dim = dim
        self.nbits = nbits
        self.K = K
        self.grad_mode = grad_mode
        self.tau = tau
        self.backend = backend
        self.gpu_cap = gpu_cap
        self.soft_M = soft_M
        self.use_async = bool(use_async and NUMBA_OK and backend == "cpu")
        if grad_mode == "soft" and backend != "cpu":
            self.backend = "cpu"     # soft mode needs the SAM candidate chain
        self.use_len_emb = use_len_emb
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.x_q = nn.Parameter(torch.zeros(1, 1, dim))
        self.x_k = nn.Parameter(torch.zeros(1, 1, dim))
        self.x_v = nn.Parameter(torch.zeros(1, 1, dim))
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)
        if o_init_zero:
            nn.init.zeros_(self.o.weight)     # residual stream stays clean
        self.emb0 = nn.Parameter(torch.zeros(dim))    # no-match payload [2]
        self.emb1 = nn.Parameter(torch.ones(dim))     # unit match payload
        if grad_mode == "soft":
            self.len_scale = nn.Parameter(torch.ones(()))
            self.len_bias = nn.Parameter(torch.zeros(()))
        else:
            self.len_scale = None
            self.len_bias = None
        D_GATE_LORA = 16                              # read gate [7]
        self.g1 = nn.Parameter(torch.zeros(dim, D_GATE_LORA))
        self.g2 = nn.Parameter(torch.zeros(D_GATE_LORA, dim))
        self.gb = nn.Parameter(torch.tensor(float(gate_bias)))  # start open
        if use_len_emb:
            self.len_emb = nn.Embedding(len_buckets, nbits)
            nn.init.zeros_(self.len_emb.weight)

    def _projections(self, x: Tensor, x_prev: Optional[Tensor] = None):
        xx = self.time_shift(x) - x
        if x_prev is not None:
            xx = torch.cat([x_prev, xx[:, 1:]], dim=1)
        q = self.q(x + xx * self.x_q)
        k = self.k(x + xx * self.x_k)
        v = self.v(x + xx * self.x_v)
        return q, k, v

    def precompute(self, xn: Tensor):
        B, T, C = xn.shape
        G = C // self.nbits
        q, k, v = self._projections(xn)
        qsym = pack_symbols(q, self.nbits).permute(0, 2, 1).contiguous()
        ksym = pack_symbols(k, self.nbits).permute(0, 2, 1).contiguous()
        vsym = pack_symbols(v, self.nbits).permute(0, 2, 1).contiguous()
        gate = torch.sigmoid(self.gb + (xn @ self.g1) @ self.g2)  # [B,T,C]
        fut = None
        if self.use_async:
            A, K = (1 << self.nbits), self.K
            BG = B * G
            qn = qsym.detach().to("cpu").numpy().astype(np.int32).reshape(BG, T)
            kn = ksym.detach().to("cpu").numpy().astype(np.int32).reshape(BG, T)
            vn = vsym.detach().to("cpu").numpy().astype(np.int32).reshape(BG, T)
            fut = _rosa_pool().submit(rosa_qkv_batch_np, qn, kn, vn, A, K)
        return dict(xn=xn, q=q, k=k, v=v, gate=gate, fut=fut,
                    qsym=qsym, ksym=ksym, vsym=vsym)

    def _assemble(self, h: dict, sym: Tensor, ln: Tensor, pos: Tensor
                  ) -> Tensor:
        if self.grad_mode == "none":
            # reference-LM behaviour: the payload is a constant; gradients
            # flow only into o / gate / emb0 / emb1 / len_emb via the ops
            # below (RosaQKVFunction with grad_mode "none" zeroes q/k/v).
            payload = RosaQKVFunction.apply(
                h["q"], h["k"], h["v"], sym, ln, pos, self.emb0, self.emb1,
                self.nbits, "none", self.tau)
        elif self.grad_mode == "soft":
            payload = RosaSoftQKVFunction.apply(
                h["q"], h["k"], h["v"], h["symM"], h["lnM"], h["posM"],
                self.emb0, self.emb1, self.nbits, self.tau,
                self.len_scale, self.len_bias)
            ln = h["lnM"][:, :, 0]      # length buckets use the best match
        else:
            payload = RosaQKVFunction.apply(
                h["q"], h["k"], h["v"], sym, ln, pos, self.emb0, self.emb1,
                self.nbits, self.grad_mode, self.tau)
        out = payload * h["gate"]
        if self.use_len_emb:
            B, T, C = payload.shape
            bucket = ln.permute(0, 2, 1).clamp(
                0, self.len_emb.num_embeddings - 1)              # [B,T,G]
            out = out + self.len_emb(bucket).reshape(B, T, C)
        return self.o(out)

    def _retrieve(self, h: dict):
        """Dispatch retrieval: async CPU SAM | sync CPU SAM | gpu windows."""
        B, G, T = h["qsym"].shape
        dev = h["q"].device
        if self.backend == "gpu":
            qf = h["qsym"].detach().reshape(B * G, T).to(dev)
            kf = h["ksym"].detach().reshape(B * G, T).to(dev)
            vf = h["vsym"].detach().reshape(B * G, T).to(dev)
            sym, ln, pos = rosa_qkv_batch_gpu(
                qf, kf, vf, alphabet=1 << self.nbits, K=self.K,
                Lcap=self.gpu_cap)
            return (sym.view(B, G, T), ln.view(B, G, T), pos.view(B, G, T))
        if h.get("fut") is not None:
            sym_np, ln_np, pos_np = h["fut"].result()
            return (torch.from_numpy(sym_np).to(dev).view(B, G, T),
                    torch.from_numpy(ln_np).to(dev).view(B, G, T),
                    torch.from_numpy(pos_np).to(dev).view(B, G, T))
        return rosa_qkv_batch_torch(h["qsym"], h["ksym"], h["vsym"],
                                    1 << self.nbits, self.K)

    def finalize(self, h: dict) -> Tensor:
        sym, ln, pos = self._retrieve(h)
        if self.grad_mode == "soft":
            symM, lnM, posM = rosa_qkv_batch_torch_m(
                h["qsym"], h["ksym"], h["vsym"], 1 << self.nbits, self.K,
                M=self.soft_M)
            h["symM"], h["lnM"], h["posM"] = symM, lnM, posM
        return self._assemble(h, sym, ln, pos)

    def forward(self, xn: Tensor) -> Tensor:
        return self.finalize(self.precompute(xn))

    # ---- RNN-mode (single token, live automata) ---------------------------
    def init_sams_batched(self, batch: int) -> "BatchedQKVSams":
        """v2: B*G live automata in one flat-array batch (fixes the v1 bug
        where RNN mode with batch > 1 silently shared G automata across the
        batch and dropped all but batch element 0)."""
        return BatchedQKVSams(batch * (self.dim // self.nbits),
                              alphabet=1 << self.nbits, K=self.K)

    def forward_step(self, xn1: Tensor, q: Tensor, k: Tensor, v: Tensor,
                     sams) -> Tuple[Tensor, List[StatefulQKVSam]]:
        """xn1: [B,1,C] normalized input (for the read gate); q,k,v: [B,1,C]
        already-projected streams; sams: a BatchedQKVSams of B*G automata
        (v2) or a list of G StatefulQKVSam (v1, batch=1 only) — extended in
        place.  Returns (branch output [B,1,C], sams)."""
        B, T, C = q.shape
        G = C // self.nbits
        assert T == 1
        qsym = pack_symbols(q, self.nbits)[:, 0]                # [B,G]
        ksym = pack_symbols(k, self.nbits)[:, 0]
        vsym = pack_symbols(v, self.nbits)[:, 0]
        if isinstance(sams, BatchedQKVSams):
            assert sams.N == B * G, f"sams.N={sams.N} != B*G={B * G}"
            syms_np, lns_np, poss_np = sams.step(
                qsym.detach().reshape(-1).cpu().numpy(),
                ksym.detach().reshape(-1).cpu().numpy(),
                vsym.detach().reshape(-1).cpu().numpy())
            dev = q.device
            sym = torch.from_numpy(syms_np).view(B, G, 1).to(dev)
            ln = torch.from_numpy(lns_np).view(B, G, 1).to(dev)
            pos = torch.from_numpy(poss_np).view(B, G, 1).to(dev)
        else:                                                   # v1 path, B=1
            assert B == 1, "list-of-SAMs RNN path supports batch=1 only"
            syms = torch.full((G, 1), -1, dtype=torch.long, device=q.device)
            lns = torch.zeros(G, 1, dtype=torch.long, device=q.device)
            poss = torch.full((G, 1), -1, dtype=torch.long, device=q.device)
            for g in range(G):
                r_, ml_, rp_ = sams[g].extend(int(qsym[0, g]),
                                              int(ksym[0, g]),
                                              int(vsym[0, g]))
                syms[g, 0] = r_; lns[g, 0] = ml_; poss[g, 0] = rp_
            sym = syms.unsqueeze(0)                             # [1,G,1]
            ln = lns.unsqueeze(0)
            pos = poss.unsqueeze(0)
        gate = torch.sigmoid(self.gb + (xn1 @ self.g1) @ self.g2)
        h = dict(xn=xn1, q=q, k=k, v=v, gate=gate, fut=None)
        return self._assemble(h, sym, ln, pos), sams

    def init_sams(self) -> List[StatefulQKVSam]:
        return [StatefulQKVSam(1 << self.nbits)
                for _ in range(self.dim // self.nbits)]


# ----------------------------------------------------------------------------
# 2.3  RWKV-7g recurrence — exact port of cuda/wkv7_cuda.cu (forward path)
#      per head (state S in R^{N x N}, channel vectors r,w,k,v,kk,agate):
#          ww  = exp(-exp(w))                  (per-channel decay, column-wise)
#          sa  = -(S @ kk)
#          S   = S * diag(ww) + sa ⊗ (kk*agate) + v ⊗ k
#          y   = S @ r
#      With ww = 1 and agate = 1 this is the delta rule S(I - kk kk^T) + v k^T
#      with kk L2-normalized — the "generalized delta rule" of RWKV-7.
# ----------------------------------------------------------------------------

def _unitri_inv(A: Tensor) -> Tensor:
    """(I + A)^{-1} for STRICTLY UPPER triangular A ([..., C, C]).

    Neumann series via log-doubling: S = sum_{k>=0} (-A)^k is exact (A
    nilpotent, A^C = 0).  S_{2m} = S_m + Z^m S_m with Z = -A.  ~log2(C)
    batched matmuls, no Python loop over time."""
    C = A.shape[-1]
    Z = -A
    S = torch.eye(C, dtype=A.dtype, device=A.device).expand_as(A) + 0.0
    Zm = Z
    it = max(1, (C - 1).bit_length())
    for _ in range(it):
        S = S + Zm @ S          # S_{2m} = S_m + Z^m S_m
        Zm = Zm @ Zm
    return S


def _rwkv7_chunk_wy(S0: Tensor, r: Tensor, w: Tensor, k: Tensor, v: Tensor,
                    kk: Tensor, agate: Tensor) -> Tuple[Tensor, Tensor]:
    """EXACT chunk-parallel evaluation of the RWKV-7g recurrence (v2).

    Per step:  S_t = S_{t-1} diag(ww_t) - (S_{t-1} kk_t) b_t^T + v_t k_t^T,
    y_t = S_t r_t,  with ww_t = exp(-exp(w_t)), b_t = kk_t * agate_t.

    WY/UT transform, per chunk (C steps, all ops batched over B,H):
      l_t = sum_{s<=t} log ww_s                  (per-channel, cumsum)
      c_i   = kk_i * e^{l_{i-1}}                 (state-side UT vectors)
      g^(t) = (I+A)^{-1} (Bhat Rplus^T)[:,t],    A = triu(Bhat Cdec^T, 1)
      m^(t)_i, h_i as derived; y and S_out assembled from 8 small matmuls.
    Verified against the sequential loop to fp32 round-off (fwd & bwd) in
    test_rwkv8.py.  Numerically safe: ww in (0.545, 1) structurally, so
    e^{l} in [0.545^C, 1] — no under/overflow for C <= 64 in fp32.

    S0: [B,H,N,N]; r/w/k/v/kk/agate: [B,H,C,N].  -> y [B,H,C,N], S_out.
    """
    B, H, C, N = r.shape
    b = kk * agate
    logw = -torch.exp(w)                                   # log(ww), <=0
    l = torch.cumsum(logw, dim=-2)                         # e^{l_t}
    el = torch.exp(l)
    el_prev = torch.cat([torch.ones_like(el[..., :1, :]),
                         el[..., :-1, :]], dim=-2)         # e^{l_{t-1}}
    Rplus = r * el
    Khat = k * torch.exp(-l)
    Bhat = b * torch.exp(-l)
    Cdec = kk * el_prev
    # ---- UT machinery -----------------------------------------------------
    A_mat = torch.triu(Bhat @ Cdec.transpose(-1, -2), diagonal=1)
    Minv = _unitri_inv(A_mat)                              # (I+A)^{-1}
    G0 = Bhat @ Rplus.transpose(-1, -2)                    # [B,H,C,C]
    # g^(t) solves the LEADING t x t subsystem only; matrices here are
    # contributor-rows x output-cols, so "i <= t" masks are UPPER triangular.
    Gmat = torch.triu(Minv @ torch.triu(G0, diagonal=0), diagonal=0)
    EffR = Rplus - Gmat.transpose(-1, -2) @ Cdec           # [B,H,C,N]
    yA = EffR @ S0.transpose(-1, -2)                       # S0 @ EffR[t]
    A1 = Khat @ Rplus.transpose(-1, -2)
    term1 = torch.triu(A1, diagonal=0).transpose(-1, -2) @ v
    A2 = Khat @ Cdec.transpose(-1, -2)
    h = torch.triu(A2, diagonal=1).transpose(-1, -2) @ v   # h_i = sum_{s<i}
    term2 = torch.triu(Gmat, diagonal=0).transpose(-1, -2) @ h
    y = yA + term1 - term2
    # ---- state carry -------------------------------------------------------
    elC = el[..., -1:, :]                                  # [B,H,1,N]
    BdC = Bhat * elC
    MvC = Minv @ BdC                                       # m^{(C)}_i vectors
    S_out = S0 * elC                                       # S0 @ diag(e^{lC})
    S_out = S_out - S0 @ (Cdec.transpose(-1, -2) @ MvC)
    S_out = S_out + (v.transpose(-1, -2) @ Khat) * elC
    S_out = S_out - h.transpose(-1, -2) @ MvC
    return y, S_out


def rwkv7g_scan(r: Tensor, w: Tensor, k: Tensor, v: Tensor, kk: Tensor,
                agate: Tensor, state: Optional[Tensor] = None,
                chunk: int = 32, mode: str = "auto"
                ) -> Tuple[Tensor, Tensor]:
    """r,w,k,v,kk,agate: [B,T,H,N]; state: [B,H,N,N].  Returns (y [B,T,H,N],
    state_out).  Runs in float32 for accumulation stability (state kept in
    fp32 like the CUDA kernel).

    mode: "sequential" — the per-timestep loop (reference, always exact);
          "chunked"     — the WY/UT chunk-parallel form (v2);
          "auto" (default) — chunked whenever T >= 2*chunk, else sequential.
    Optional per-chunk gradient checkpointing bounds activation memory for
    long contexts (applies to both paths)."""
    B, T, H, N = r.shape
    orig_dtype = r.dtype
    with torch.autocast(device_type=r.device.type, enabled=False):
        # fp32 for stability under fp16/bf16 autocast; fp64 allowed (tests)
        cd = r.dtype if r.dtype in (torch.float32, torch.float64) \
            else torch.float32
        r = r.to(cd); w = w.to(cd); k = k.to(cd); v = v.to(cd)
        kk = kk.to(cd); agate = agate.to(cd)
        if state is not None:
            state = state.to(cd)
        if state is None:
            state = r.new_zeros(B, H, N, N)

        def run_seq(S_in, r_c, w_c, k_c, v_c, kk_c, ag_c):
            ww_c = torch.exp(-torch.exp(w_c))
            bb_c = kk_c * ag_c
            outs = []
            S = S_in
            Tc = r_c.shape[1]
            for t in range(Tc):
                sa = -(S @ kk_c[:, t].unsqueeze(-1))           # [B,H,N,1]
                S = (S * ww_c[:, t].unsqueeze(-2)
                     + sa * bb_c[:, t].unsqueeze(-2)
                     + v_c[:, t].unsqueeze(-1) * k_c[:, t].unsqueeze(-2))
                outs.append((S @ r_c[:, t].unsqueeze(-1)).squeeze(-1))
            return torch.stack(outs, dim=1), S

        def run_wy(S_in, r_c, w_c, k_c, v_c, kk_c, ag_c):
            # scan convention is [B,C,H,N]; _rwkv7_chunk_wy wants [B,H,C,N]
            rb = r_c.transpose(1, 2)
            wb = w_c.transpose(1, 2)
            kb = k_c.transpose(1, 2)
            vb = v_c.transpose(1, 2)
            kkb = kk_c.transpose(1, 2)
            agb = ag_c.transpose(1, 2)
            y, S = _rwkv7_chunk_wy(S_in, rb, wb, kb, vb, kkb, agb)
            return y.transpose(1, 2), S

        use_chunked = (mode == "chunked" or
                       (mode == "auto" and T >= 2 * (chunk or 1)))
        if chunk is None or chunk <= 0 or T <= chunk:
            one_shot = True
        else:
            one_shot = False
        if use_chunked:
            run = run_wy
            # WY path is fp32-safe for chunk <= 64 (ww >= 0.545 structurally,
            # so e^{l} in [0.545^64, 1] — far from fp32 limits).  Clamp:
            if not one_shot and chunk > 64:
                chunk = 64
        else:
            run = run_seq
        grad_ckpt = (chunk is not None and chunk > 0 and not one_shot
                     and T > chunk and torch.is_grad_enabled())
        if one_shot:
            y, state = run(state, r, w, k, v, kk, agate)
        elif grad_ckpt:
            ys = []
            for s0 in range(0, T, chunk):
                s1 = min(s0 + chunk, T)
                y_c, state = torch.utils.checkpoint.checkpoint(
                    run, state, r[:, s0:s1], w[:, s0:s1], k[:, s0:s1],
                    v[:, s0:s1], kk[:, s0:s1], agate[:, s0:s1],
                    use_reentrant=False)
                ys.append(y_c)
            y = torch.cat(ys, dim=1)
        else:
            ys = []
            for s0 in range(0, T, chunk):
                s1 = min(s0 + chunk, T)
                y_c, state = run(state, r[:, s0:s1], w[:, s0:s1],
                                 k[:, s0:s1], v[:, s0:s1], kk[:, s0:s1],
                                 agate[:, s0:s1])
                ys.append(y_c)
            y = torch.cat(ys, dim=1)
    return y.to(orig_dtype), state


# ----------------------------------------------------------------------------
# 2.4  RWKV-7 time-mix block ("x070") — full port of RWKV_Tmix_x070 in [1],
#      with the CUDA op replaced by rwkv7g_scan.
# ----------------------------------------------------------------------------

def _ortho_init(x: Tensor, scale: float) -> Tensor:
    with torch.no_grad():
        shape = x.shape
        if len(shape) == 2:
            gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
            nn.init.orthogonal_(x, gain=gain * scale)
        elif len(shape) == 3:
            gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
            for i in range(shape[0]):
                nn.init.orthogonal_(x[i], gain=gain * scale)
        else:
            raise ValueError(shape)
        return x


class RWKV_Tmix_x070(nn.Module):
    """RWKV-7g time-mix.  forward(x, v_first) mirrors [1] and additionally
    accepts/returns the recurrent state for RNN-mode."""

    def __init__(self, n_embd: int, n_layer: int, layer_id: int,
                 head_size: int = 16, scan_chunk: int = 32,
                 scan_mode: str = "auto"):
        super().__init__()
        self.layer_id = layer_id
        self.head_size = head_size
        C = n_embd
        self.n_head = C // head_size
        assert C % self.n_head == 0
        H, N = self.n_head, head_size
        with torch.no_grad():
            ratio_0_to_1 = layer_id / max(1, n_layer - 1)
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C
            self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_v = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            www = torch.zeros(C); zigzag = torch.zeros(C); linear = torch.zeros(C)
            for n in range(C):
                linear[n] = n / (C - 1) - 0.5
                zigzag[n] = ((n % N) - ((N - 1) / 2)) / ((N - 1) / 2)
                zigzag[n] = zigzag[n] * abs(zigzag[n])
                www[n] = -6 + 6 * (n / (C - 1)) ** (1 + 1 * ratio_0_to_1 ** 0.3)
            D_DECAY_LORA = 16
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(_ortho_init(
                torch.zeros(D_DECAY_LORA, C), 0.1))
            self.w0 = nn.Parameter(www.reshape(1, 1, C) + 0.5 + zigzag * 2.5)
            D_AAA_LORA = 16
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(_ortho_init(
                torch.zeros(D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(
                torch.zeros(1, 1, C) - 0.19 + zigzag * 0.3 + linear * 0.4)
            D_MV_LORA = 16
            self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
            self.v2 = nn.Parameter(_ortho_init(
                torch.zeros(D_MV_LORA, C), 0.1))
            self.v0 = nn.Parameter(torch.zeros(1, 1, C) + 0.73 - linear * 0.4)
            D_GATE_LORA = 16
            self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
            self.g2 = nn.Parameter(_ortho_init(
                torch.zeros(D_GATE_LORA, C), 0.1))
            self.k_k = nn.Parameter(torch.zeros(1, 1, C) + 0.71 - linear * 0.1)
            self.k_a = nn.Parameter(torch.zeros(1, 1, C) + 1.02)
            self.r_k = nn.Parameter(torch.zeros(H, N) - 0.04)
            self.receptance = nn.Linear(C, C, bias=False)
            self.key = nn.Linear(C, C, bias=False)
            self.value = nn.Linear(C, C, bias=False)
            self.output = nn.Linear(C, C, bias=False)
            self.ln_x = nn.GroupNorm(H, C, eps=64e-5)
            self.receptance.weight.data.uniform_(-0.5 / (C ** 0.5),
                                                  0.5 / (C ** 0.5))
            self.key.weight.data.uniform_(-0.05 / (C ** 0.5),
                                          0.05 / (C ** 0.5))
            self.value.weight.data.uniform_(-0.5 / (C ** 0.5),
                                            0.5 / (C ** 0.5))
            self.output.weight.data.zero_()
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.scan_chunk = scan_chunk
        self.scan_mode = scan_mode

    def forward(self, x: Tensor, v_first: Tensor,
                state: Optional[Tensor] = None,
                x_prev: Optional[Tensor] = None,
                ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        B, T, C = x.size()
        H = self.n_head
        xx = self.time_shift(x) - x
        if x_prev is not None:              # RNN step: use previous x
            xx = torch.cat([x_prev - x[:, :1], xx[:, 1:]], dim=1)
        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g
        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(
                self.v0 + (xv @ self.v1) @ self.v2)   # value residual
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)  # icl rate
        g = torch.sigmoid(xg @ self.g1) @ self.g2
        kk = k * self.k_k
        kk = F.normalize(kk.view(B, T, H, -1), dim=-1, p=2.0).view(B, T, C)
        k = k * (1 + (a - 1) * self.k_a)
        # RWKV-7g recurrence (kernel call: q_=r, a_=-kk, b_=kk*a)
        y, state_out = rwkv7g_scan(
            r.view(B, T, H, -1), w.view(B, T, H, -1), k.view(B, T, H, -1),
            v.view(B, T, H, -1), kk.view(B, T, H, -1), a.view(B, T, H, -1),
            state, self.scan_chunk, mode=self.scan_mode)
        x = self.ln_x(y.reshape(B * T, C)).view(B, T, C)
        x = x + ((r.view(B, T, H, -1) * k.view(B, T, H, -1) * self.r_k)
                 .sum(dim=-1, keepdim=True) * v.view(B, T, H, -1)).view(B, T, C)
        x = self.output(x * g)
        return x, v_first, state_out


# ----------------------------------------------------------------------------
# 2.5  FFN with DeepEmbed (RWKV-8's first feature, May 2025)
# ----------------------------------------------------------------------------

class FFNDeepEmbed(nn.Module):
    """ReLu^2 FFN with optional DeepEmbed token modulation:
       1x:  value(relu(key(x))^2) * deepemb(idx)
       4x:  value(relu(key(x))^2 * deepemb(idx))     (better, more params)
    v2: optional dropout on the FFN output."""

    def __init__(self, dim: int, dim_ffn: int, vocab: int,
                 deepembed: str = "1x", drop_p: float = 0.0):
        super().__init__()
        self.deep_mode = deepembed            # "none" | "1x" | "4x"
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.x_k = nn.Parameter(torch.zeros(1, 1, dim))
        self.key = nn.Linear(dim, dim_ffn, bias=False)
        self.value = nn.Linear(dim_ffn, dim, bias=False)
        self.drop = (nn.Dropout(drop_p) if drop_p > 0 else nn.Identity())
        with torch.no_grad():
            self.value.weight.data.zero_()
            nn.init.orthogonal_(self.key.weight.data, gain=(4 ** 0.5) / 2)
        if deepembed == "1x":
            self.deepemb = nn.Embedding(vocab, dim)
            nn.init.ones_(self.deepemb.weight)
        elif deepembed == "4x":
            self.deepemb = nn.Embedding(vocab, dim_ffn)
            nn.init.ones_(self.deepemb.weight)

    def forward(self, x: Tensor, idx: Tensor,
                x_prev: Optional[Tensor] = None) -> Tensor:
        xx = self.time_shift(x) - x
        if x_prev is not None:
            xx = torch.cat([x_prev - x[:, :1], xx[:, 1:]], dim=1)
        xk = x + xx * self.x_k
        h = torch.relu(self.key(xk)) ** 2
        if self.deep_mode == "4x":
            return self.drop(self.value(h * self.deepemb(idx)))
        out = self.value(h)
        if self.deep_mode == "1x":
            out = out * self.deepemb(idx)
        return self.drop(out)

# ==========================================================================
# >>> section from p3_models.py
# ==========================================================================

# ============================================================================
# Section 3 — Models
#   3.1  RWKV8Config
#   3.2  RWKV8Block  —  x + timemix(x) + rosaQKV(x), then FFN(+DeepEmbed)
#   3.3  RWKV8Heron  —  the full RWKV-8 model (GPT-mode + stateful RNN-mode)
#   3.4  RosaLM      —  the pure-ROSA LM (no time-mix; like rosa4bitLM [1])
# ============================================================================


@dataclass
class RWKV8Config:
    vocab_size: int = 256          # byte-level by default (enwik8)
    n_layer: int = 6
    n_embd: int = 256              # ~7.3M params with everything enabled
    head_size: int = 16
    dim_ffn: Optional[int] = None  # default: 4 * n_embd
    # --- ROSA QKV branch ---
    use_qkv_rosa: bool = True
    rosa_bits: int = 4             # 4-bit symbol groups (16-symbol alphabet)
    rosa_K: int = 0                # match truncation (0 = unbounded, [4])
    rosa_grad: str = "ste"         # "ste" | "soft" (v2) | "exact" | "none"
    rosa_tau: float = 1e-3
    rosa_o_zero: bool = True       # zero-init the branch output projection
    rosa_gate_bias: float = 2.0    # read gate starts open (~0.88)
    use_len_emb: bool = True
    len_buckets: int = 32
    # --- v2 additions ---
    rosa_backend: str = "cpu"      # "cpu" (exact SAM) | "gpu" (exact-window,
                                   #   matches <= rosa_gpu_cap; any device)
    rosa_gpu_cap: int = 64         # exactness cap of the gpu backend
    rosa_soft_M: int = 4           # candidates used by grad_mode="soft"
    layer_rosa_bits: Optional[List[int]] = None   # per-layer override
    drop_p: float = 0.0            # residual/FFN dropout (small-data use)
    inner_monologue: bool = False  # experimental: self-routed Emb(ROSA(z))
    scan_mode: str = "auto"        # "auto" | "chunked" | "sequential" (v2)
    compile_model: bool = False    # torch.compile the blocks (v2)
    ctx_len: int = 256             # advisory default context for train_model
    # --- other RWKV-8 features ---
    use_token_rosa: bool = True    # Emb(ROSA(idx)) at the input ([1])
    deepembed: str = "1x"          # "none" | "1x" | "4x"  ([1] wiki)
    # --- plumbing ---
    async_rosa: bool = True        # CPU retrieval overlaps GPU compute ([5])
    scan_chunk: int = 32           # RWKV-7g scan checkpointing chunk
    tie_head: bool = True          # parameter golf ([11])
    ln0: bool = True               # LayerNorm right after embeddings
    emb_std: float = 0.02          # embedding init std
    block_ln: bool = True          # LayerNorms inside blocks (251018-style
                                   # demo models set this False)
    ffn_mode: str = "ffn"          # "ffn" | "linear" (251018 style) | "none"

    def __post_init__(self):
        if self.dim_ffn is None:
            self.dim_ffn = 4 * self.n_embd
        assert self.n_embd % self.rosa_bits == 0
        assert self.deepembed in ("none", "1x", "4x")
        assert self.rosa_grad in ("ste", "soft", "exact", "none")
        assert self.ffn_mode in ("ffn", "linear", "none")
        assert self.rosa_backend in ("cpu", "gpu")
        assert self.scan_mode in ("auto", "chunked", "sequential")
        if self.ffn_mode != "ffn":
            self.deepembed = "none"   # DeepEmbed lives inside the FFN
        if self.layer_rosa_bits is not None:
            assert len(self.layer_rosa_bits) == self.n_layer
            for nb in self.layer_rosa_bits:
                assert self.n_embd % nb == 0

    # ---- v2 presets --------------------------------------------------------
    @classmethod
    def preset(cls, name: str, **over) -> "RWKV8Config":
        """Ready-made configs: 'demo' (default), 't4max' (largest that
        trains comfortably on a free Colab T4), 'golf16mb' (fits the
        parameter-golf 16 MB fp16 budget with head-room), 'test' (tiny, for
        CPU tests)."""
        presets = {
            "demo": dict(vocab_size=256, n_layer=6, n_embd=256),
            "t4max": dict(vocab_size=256, n_layer=12, n_embd=512,
                          ctx_len=512, scan_mode="chunked"),
            "golf16mb": dict(vocab_size=256, n_layer=8, n_embd=384,
                             deepembed="1x"),
            "test": dict(vocab_size=64, n_layer=2, n_embd=64,
                         dim_ffn=128, len_buckets=8, async_rosa=False),
        }
        if name not in presets:
            raise KeyError(f"unknown preset {name!r}; "
                           f"one of {sorted(presets)}")
        cfg = cls(**{**presets[name], **over})
        if name == "t4max":
            cfg.ctx_len = 512      # advisory: used by train_model default
        return cfg

    def summary(self) -> str:
        return (f"RWKV8Config(L={self.n_layer}, D={self.n_embd}, "
                f"V={self.vocab_size}, ffn={self.dim_ffn}, "
                f"qkv_rosa={self.use_qkv_rosa}/{self.rosa_bits}bit/"
                f"{self.rosa_grad}/{self.rosa_backend}, "
                f"token_rosa={self.use_token_rosa}, "
                f"deepembed={self.deepembed}, tie={self.tie_head}, "
                f"scan={self.scan_mode})")


class RWKV8Block(nn.Module):
    """Reference block layout of 251024_rosaQKV_run.py [1]:
        xr  = rosaQKV(ln_c(x))          (parallel branch)
        xx  = RWKV7g(ln_a(x))           (time-mix)
        x   = x + xx + xr
        x   = x + FFN(ln_b(x))          (+ DeepEmbed [1] wiki)
    precompute/finalize split lets the CPU suffix-automaton retrieval run
    concurrently with the GPU time-mix ([5] async pipeline).

    RNN-mode (states is not None): every sub-layer receives its own previous
    normalized input for the token shift, exactly reproducing GPT-mode
    behaviour one position at a time."""

    _PREV_KEYS = ("x_prev_a", "x_prev_b", "x_prev_c")

    def __init__(self, cfg: RWKV8Config, layer_id: int, use_timemix: bool):
        super().__init__()
        C = cfg.n_embd
        self.layer_id = layer_id
        self.use_timemix = use_timemix
        self.use_rosa = cfg.use_qkv_rosa
        self.ffn_mode = cfg.ffn_mode
        self.drop = (nn.Dropout(cfg.drop_p) if cfg.drop_p > 0
                     else nn.Identity())
        ln = nn.LayerNorm if cfg.block_ln else nn.Identity
        self.ln_a = ln(C)
        self.ln_b = ln(C)
        self.ln_c = ln(C)
        if use_timemix:
            self.timemix = RWKV_Tmix_x070(
                C, cfg.n_layer, layer_id, cfg.head_size, cfg.scan_chunk,
                scan_mode=cfg.scan_mode)
        if self.use_rosa:
            nbits = (cfg.layer_rosa_bits[layer_id]
                     if cfg.layer_rosa_bits else cfg.rosa_bits)
            self.rosa = RosaQKVBranch(
                C, nbits=nbits, K=cfg.rosa_K,
                grad_mode=cfg.rosa_grad, tau=cfg.rosa_tau,
                use_async=cfg.async_rosa, use_len_emb=cfg.use_len_emb,
                len_buckets=cfg.len_buckets,
                o_init_zero=cfg.rosa_o_zero, gate_bias=cfg.rosa_gate_bias,
                backend=cfg.rosa_backend, gpu_cap=cfg.rosa_gpu_cap,
                soft_M=cfg.rosa_soft_M)
        if self.ffn_mode == "ffn":
            self.ffn = FFNDeepEmbed(C, cfg.dim_ffn, cfg.vocab_size,
                                    cfg.deepembed, drop_p=cfg.drop_p)
        elif self.ffn_mode == "linear":
            self.ffn = nn.Linear(C, C)   # the 251018 demo style [1]

    def forward(self, x: Tensor, idx: Tensor, v_first: Optional[Tensor],
                states: Optional[Dict] = None):
        if states is None:                                   # ---- GPT mode
            h = self.rosa.precompute(self.ln_c(x)) if self.use_rosa else None
            if self.use_timemix:
                xx, v_first, _ = self.timemix(self.ln_a(x), v_first)
            else:
                xx = x.new_zeros(x.shape)
            xr = self.rosa.finalize(h) if self.use_rosa else 0.0
            x = x + self.drop(xx + xr) if self.use_rosa else x + self.drop(xx)
            if self.ffn_mode == "ffn":
                x = x + self.drop(self.ffn(self.ln_b(x), idx))
            elif self.ffn_mode == "linear":
                x = x + self.drop(self.ffn(self.ln_b(x)))
            return x, v_first
        # ---- RNN mode (single position) ----
        xa = self.ln_a(x); xc = self.ln_c(x); xb = self.ln_b(x)
        if self.use_timemix:
            xx, v_first, S = self.timemix(
                xa, v_first, states.get("S"), states.get("x_prev_a"))
            states["S"] = S
        else:
            xx = x.new_zeros(x.shape)
        if self.use_rosa:
            q, k, v = self.rosa._projections(xc, states.get("x_prev_c"))
            xr, _ = self.rosa.forward_step(xc, q, k, v, states["sams"])
        else:
            xr = 0.0
        x = x + xx + xr if self.use_rosa else x + xx
        if self.ffn_mode == "ffn":
            x = x + self.ffn(xb, idx, states.get("x_prev_b"))
        elif self.ffn_mode == "linear":
            x = x + self.ffn(xb)
        states["x_prev_a"] = xa
        states["x_prev_b"] = xb
        states["x_prev_c"] = xc
        return x, v_first


class RWKV8Heron(nn.Module):
    """RWKV-8 "Heron": RWKV-7g backbone + ROSA branches + DeepEmbed.

    forward(idx)              -> logits [B,T,V]           (GPT-mode)
    forward(idx, state=s)     -> (logits [B,1,V], state)  (RNN-mode, O(1)/tok)
    """

    def __init__(self, cfg: RWKV8Config):
        super().__init__()
        self.cfg = cfg
        C, V = cfg.n_embd, cfg.vocab_size
        self.emb = nn.Embedding(V, C)
        nn.init.normal_(self.emb.weight, std=cfg.emb_std)
        self.rosa_tok = (RosaTokenEmb(V, C, cfg.len_buckets,
                                      backend=cfg.rosa_backend,
                                      gpu_cap=min(cfg.rosa_gpu_cap, 16))
                         if cfg.use_token_rosa else None)
        self.ln0 = nn.LayerNorm(C) if cfg.ln0 else nn.Identity()
        self.blocks = nn.ModuleList(
            [RWKV8Block(cfg, i, use_timemix=True) for i in range(cfg.n_layer)])
        self.ln_out = nn.LayerNorm(C)
        self.head = nn.Linear(C, V, bias=False)
        if cfg.tie_head:
            self.head.weight = self.emb.weight
        else:
            nn.init.zeros_(self.head.weight)
        self.inner_monologue = bool(cfg.inner_monologue and cfg.use_token_rosa)
        self._im_ids: Optional[Tensor] = None     # 1-step-lagged self tokens
        if cfg.compile_model:
            try:
                self.blocks = torch.compile(self.blocks)
            except Exception as e:                    # pragma: no cover
                print(f"[rwkv8 v2] torch.compile unavailable: {e}")

    def forward(self, idx: Tensor,
                rosa_tok: Optional[Tuple[Tensor, Tensor]] = None,
                state: Optional[Dict] = None):
        if state is not None:
            return self._forward_rnn(idx, state)
        x = self.emb(idx)
        if self.rosa_tok is not None:
            rt = rosa_tok if rosa_tok is not None else (None, None)
            x = x + self.rosa_tok(idx, rt[0], rt[1])
        if self.inner_monologue:
            # v2 experimental "inner monologue": the input also gets
            # Emb(ROSA(z)) where z is the model's OWN previous output
            # (argmax, detached, 1-step lagged).  At step 0 the buffer is
            # empty and this is a no-op.
            if self._im_ids is not None and self._im_ids.shape == idx.shape:
                x = x + self.rosa_tok(self._im_ids.to(idx.device))
        x = self.ln0(x)
        v_first = None
        for blk in self.blocks:
            x, v_first = blk(x, idx, v_first)
        logits = self.head(self.ln_out(x))
        if self.inner_monologue:
            self._im_ids = logits.argmax(-1).detach()
        return logits

    # ---------------- RNN-mode ----------------
    def init_state(self, batch: int = 1, device=None) -> Dict:
        cfg = self.cfg
        dev = device or next(self.parameters()).device
        H = cfg.n_embd // cfg.head_size
        state: Dict = {"pos": 0, "v_first": None}
        state["S"] = [torch.zeros(batch, H, cfg.head_size, cfg.head_size,
                                  device=dev) for _ in range(cfg.n_layer)]
        state["sams"] = ([blk.rosa.init_sams_batched(batch)
                          for blk in self.blocks]
                         if cfg.use_qkv_rosa else [None] * cfg.n_layer)
        for key in RWKV8Block._PREV_KEYS:
            state[key] = [None] * cfg.n_layer
        state["tok_sam"] = ([StatefulTokenSam(cfg.vocab_size)
                             for _ in range(batch)]
                            if self.rosa_tok is not None else None)
        return state

    def _forward_rnn(self, idx: Tensor, state: Dict):
        assert idx.shape[1] == 1, "RNN-mode: one position per call"
        idx = idx.long()
        B = idx.shape[0]
        x = self.emb(idx)
        if self.rosa_tok is not None:
            syms = torch.full((B, 1), -1, dtype=torch.long, device=idx.device)
            lns = torch.zeros(B, 1, dtype=torch.long, device=idx.device)
            for b in range(B):
                y, ml = state["tok_sam"][b].extend(int(idx[b, 0]))
                syms[b, 0] = y; lns[b, 0] = ml
            x = x + self.rosa_tok(idx, syms, lns)
        x = self.ln0(x)
        v_first = state["v_first"]
        for li, blk in enumerate(self.blocks):
            bstates = {k: state[k][li] for k in RWKV8Block._PREV_KEYS}
            bstates["S"] = state["S"][li]
            bstates["sams"] = state["sams"][li]
            x, v_first = blk(x, idx, v_first, bstates)
            for k in RWKV8Block._PREV_KEYS:
                state[k][li] = bstates[k]
            state["S"][li] = bstates["S"]
        state["v_first"] = v_first
        state["pos"] += 1
        return self.head(self.ln_out(x)), state


class RosaLM(nn.Module):
    """Pure-ROSA LM (no time-mix), following 260222_rosa4bitLM_L12.py [1]:
        x = emb(idx); blocks of [x + rosa(ln3(x)); x + ffn(ln2(x))]; head.
    Uses the upgraded RosaQKVBranch (emb0 no-match payload [2], read gate
    [7], match-length buckets [4][7]).  GPT-mode + stateful RNN-mode."""

    def __init__(self, cfg: RWKV8Config):
        super().__init__()
        cfg = copy.deepcopy(cfg)
        cfg.use_qkv_rosa = True
        self.cfg = cfg
        C, V = cfg.n_embd, cfg.vocab_size
        self.emb = nn.Embedding(V, C)
        nn.init.normal_(self.emb.weight, std=cfg.emb_std)
        self.rosa_tok = (RosaTokenEmb(V, C, cfg.len_buckets)
                         if cfg.use_token_rosa else None)
        self.ln0 = nn.LayerNorm(C) if cfg.ln0 else nn.Identity()
        self.blocks = nn.ModuleList(
            [RWKV8Block(cfg, i, use_timemix=False) for i in range(cfg.n_layer)])
        self.ln_out = nn.LayerNorm(C)
        self.head = nn.Linear(C, V, bias=False)
        if cfg.tie_head:
            self.head.weight = self.emb.weight
        else:
            nn.init.zeros_(self.head.weight)

    def forward(self, idx: Tensor, state: Optional[Dict] = None,
                rosa_tok: Optional[Tuple[Tensor, Tensor]] = None):
        if state is not None:
            return self._forward_rnn(idx, state)
        x = self.emb(idx)
        if self.rosa_tok is not None:
            rt = rosa_tok if rosa_tok is not None else (None, None)
            x = x + self.rosa_tok(idx, rt[0], rt[1])
        x = self.ln0(x)
        v_first = None
        for blk in self.blocks:
            x, v_first = blk(x, idx, v_first)
        return self.head(self.ln_out(x))

    def init_state(self, batch: int = 1, device=None) -> Dict:
        cfg = self.cfg
        return {
            "pos": 0, "v_first": None, "S": None,
            "tok_sam": ([StatefulTokenSam(cfg.vocab_size)
                         for _ in range(batch)]
                        if self.rosa_tok is not None else None),
            "sams": [blk.rosa.init_sams_batched(batch) for blk in self.blocks],
            "x_prev_a": [None] * cfg.n_layer,
            "x_prev_b": [None] * cfg.n_layer,
            "x_prev_c": [None] * cfg.n_layer,
        }

    def _forward_rnn(self, idx: Tensor, state: Dict):
        assert idx.shape[1] == 1
        idx = idx.long()
        B = idx.shape[0]
        x = self.emb(idx)
        if self.rosa_tok is not None:
            syms = torch.full((B, 1), -1, dtype=torch.long, device=idx.device)
            lns = torch.zeros(B, 1, dtype=torch.long, device=idx.device)
            for b in range(B):
                y, ml = state["tok_sam"][b].extend(int(idx[b, 0]))
                syms[b, 0] = y; lns[b, 0] = ml
            x = x + self.rosa_tok(idx, syms, lns)
        x = self.ln0(x)
        v_first = None
        for li, blk in enumerate(self.blocks):
            bstates = {k: state[k][li] for k in RWKV8Block._PREV_KEYS}
            bstates["S"] = None
            bstates["sams"] = state["sams"][li]
            x, v_first = blk(x, idx, v_first, bstates)
            # v2 FIX: write the token-shift states back (v1 dropped them, so
            # RNN-mode silently diverged from GPT-mode for RosaLM)
            for k in RWKV8Block._PREV_KEYS:
                state[k][li] = bstates[k]
        state["pos"] += 1
        return self.head(self.ln_out(x)), state

# ==========================================================================
# >>> section from p4_train.py
# ==========================================================================

# ============================================================================
# Section 4 — Data, training, generation, evaluation
#   4.1  enwik8 / text datasets (byte-level) + synthetic showcase tasks
#   4.2  Trainer (warmup+cosine, AMP-optional, async ROSA prefetch pipeline)
#   4.3  Generation (GPT-mode and stateful RNN-mode) + sampling
#   4.4  Evaluation: bpc, MQAR/needle, standalone ROSA compressor ([6],[11])
# ============================================================================


def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ----------------------------------------------------------------------------
# 4.1  Data
# ----------------------------------------------------------------------------

def download_enwik8(root: str = "data") -> str:
    """Download enwik8 (100MB of raw Wikipedia XML bytes) with mirrors."""
    os.makedirs(root, exist_ok=True)
    raw_path = os.path.join(root, "enwik8")
    if os.path.exists(raw_path):
        return raw_path
    zip_path = os.path.join(root, "enwik8.zip")
    urls = ["https://mattmahoney.net/dc/enwik8.zip",
            "https://data.deepai.org/enwik8.zip"]
    import urllib.request
    for url in urls:
        try:
            print(f"downloading enwik8 from {url} ...")
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path) as z:
                name = [n for n in z.namelist() if "enwik8" in n][0]
                with open(raw_path, "wb") as f:
                    f.write(z.read(name))
            os.remove(zip_path)
            return raw_path
        except Exception as e:
            print(f"  mirror failed: {e}")
    raise RuntimeError("could not download enwik8; place data/enwik8 manually")


def download_tinyshakespeare(root: str = "data") -> str:
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, "tinyshakespeare.txt")
    if not os.path.exists(path):
        import urllib.request
        url = ("https://raw.githubusercontent.com/karpathy/char-rnn/master/"
               "data/tinyshakespeare/input.txt")
        urllib.request.urlretrieve(url, path)
    return path


class ByteData:
    """Byte-level LM data over any file.  enwik8 split: first 90M bytes
    train, next 5M val (raw XML bytes — the honest byte-level metric;
    classic ~1.0 bpc numbers use preprocessed text)."""

    def __init__(self, path: str, val_bytes: int = 5_000_000,
                 max_train_bytes: Optional[int] = None):
        raw = np.fromfile(path, dtype=np.uint8)
        print(f"loaded {path}: {len(raw):,} bytes "
              f"({len(np.unique(raw))} distinct)")
        self.vocab = 256
        n_train = min(len(raw) - val_bytes,
                      max_train_bytes or len(raw) - val_bytes)
        self.train = raw[:n_train]
        self.val = raw[n_train:n_train + val_bytes]
        self._rng = np.random.default_rng(42)

    def sample_batch(self, B: int, T: int, device: str,
                     split: str = "train") -> Tuple[Tensor, Tensor]:
        arr = self.train if split == "train" else self.val
        ix = self._rng.integers(0, len(arr) - T - 1, size=B)
        xb = np.stack([arr[i:i + T] for i in ix]).astype(np.int64)
        yb = np.stack([arr[i + 1:i + T + 1] for i in ix]).astype(np.int64)
        return (torch.from_numpy(xb).to(device),
                torch.from_numpy(yb).to(device))

    def sample_np(self, B: int, T: int, split: str = "train") -> np.ndarray:
        arr = self.train if split == "train" else self.val
        ix = self._rng.integers(0, len(arr) - T - 1, size=B)
        return np.stack([arr[i:i + T] for i in ix]).astype(np.int64)


class SequentialData(ByteData):
    """v2: epoch-based, document-ordered streaming batches (complements
    ByteData's random crops).  Each epoch walks the training bytes
    sequentially in non-overlapping T-windows with a shuffled start offset;
    the ROSA suffix automata then see text the way a compressor does.
    `next_batch` cycles automatically across epochs."""

    def __init__(self, path: str, val_bytes: int = 5_000_000,
                 max_train_bytes: Optional[int] = None, seed: int = 42):
        super().__init__(path, val_bytes, max_train_bytes)
        self.seed = seed
        self._cursor = 0
        self._epoch = 0

    def next_batch(self, B: int, T: int, device: str,
                   split: str = "train") -> Tuple[Tensor, Tensor]:
        arr = self.train if split == "train" else self.val
        n = len(arr)
        xb, yb = [], []
        for _ in range(B):
            if self._cursor + T + 1 > n:
                self._epoch += 1
                self._cursor = int(self._rng.integers(0, T))
            xb.append(arr[self._cursor:self._cursor + T])
            yb.append(arr[self._cursor + 1:self._cursor + T + 1])
            self._cursor += T
        return (torch.from_numpy(np.stack(xb).astype(np.int64)).to(device),
                torch.from_numpy(np.stack(yb).astype(np.int64)).to(device))


# ---- synthetic showcase tasks (formats follow BlinkDL's demos [1]) --------

def make_counting_batch(B: int, T: int, start: Optional[List[int]] = None,
                        device: str = "cpu") -> Tensor:
    """'A 123 A 124 A 125 ...' — counting/increment task (251014 in [1]).
    Vocab: 0-9 digits, 10 = 'A' separator."""
    out = []
    for b in range(B):
        if start is None:
            k = random.randint(1, 3)
            lo = 0 if k == 1 else 10 ** (k - 1)
            n = random.randint(lo, 10 ** k - 1)
        else:
            n = start[b]
        a = [10]
        while len(a) < T:
            a += [ord(c) - 48 for c in str(n)] + [10]
            n += 1
        out.append(a[:T])
    return torch.tensor(out, device=device, dtype=torch.long)


def make_reverse_batch(B: int, T: int, digits: Optional[int] = None,
                       device: str = "cpu") -> Tensor:
    """'<digits>,<reversed digits>###...' — the 251105 reverse task [1].
    Vocab: 0-9, ','=10, '#'=11.  Numbers have exactly `digits` digits (no
    leading zeros), matching get_randint in the reference."""
    out = []
    for _ in range(B):
        d = digits if digits is not None else random.randint(1, T // 2 - 1)
        d = max(1, min(d, T // 2 - 1))
        lo = 0 if d == 1 else 10 ** (d - 1)
        n = random.randint(lo, 10 ** d - 1)
        raw = str(n) + "," + str(n)[::-1]
        raw = raw + "#" * (T - len(raw))
        out.append([ord(c) - 48 if c.isdigit() else (10 if c == "," else 11)
                    for c in raw])
    return torch.tensor(out, device=device, dtype=torch.long)


def make_mqar_batch(B: int, T: int, n_pairs: int = 16, vocab: int = 64,
                    device: str = "cpu") -> Tuple[Tensor, Tensor]:
    """Multi-Query Associative Recall.  Sequence: k1 v1 k2 v2 ... kP vP
    followed by queries q1 ? q2 ? ... where qj repeats an earlier key.
    Returns (batch [B,T], answer positions/values) — at a '?' the model must
    produce the value that followed the queried key's first occurrence.
    Keys/values drawn from a shared vocab (random, distinct within sample).
    By construction, token-level ROSA answers every query perfectly."""
    assert vocab >= 2 * n_pairs + 4
    xs, answers = [], []
    for _ in range(B):
        toks = list(np.random.choice(vocab, size=2 * n_pairs, replace=False))
        seq = toks[:]  # k1 v1 k2 v2 ...
        qidx = np.random.choice(n_pairs, size=min(n_pairs, (T - 2 * n_pairs)),
                                replace=False)
        ans = []
        for qi in qidx:
            if len(seq) + 2 > T:
                break
            k, v = toks[2 * qi], toks[2 * qi + 1]
            seq.append(k)          # query key = 2nd occurrence of k
            seq.append(v)          # its value: the target at the query pos
            ans.append((len(seq) - 1, v))
        seq = seq[:T]
        x = np.array(seq, dtype=np.int64)
        xs.append(x)
        answers.append(ans)
    return torch.from_numpy(np.stack(xs)).to(device), answers


# ----------------------------------------------------------------------------
# 4.2  Trainer (v2: schedules, param groups, accumulation, DDP, logging,
#      EMA, sequential data mode, resume-able checkpoints)
# ----------------------------------------------------------------------------

_PREFETCH_POOL: Optional[ThreadPoolExecutor] = None


def _prefetch_pool() -> ThreadPoolExecutor:
    global _PREFETCH_POOL
    if _PREFETCH_POOL is None:
        _PREFETCH_POOL = ThreadPoolExecutor(max_workers=1)
    return _PREFETCH_POOL


class EMA:
    """Exponential moving average of model weights (v2).  `copy_to` swaps the
    EMA weights in for evaluation; `restore` puts the training weights back.
    Tracks unique parameter tensors by identity, so TIED weights (the
    emb/head tying of [11]) are handled correctly — a naive state_dict loop
    double-copies them and corrupts the restore."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: Dict[str, Tensor] = {}
        self._refs: Dict[str, Tensor] = {}
        self.backup: Dict[str, Tensor] = {}
        seen = set()
        for name, t in list(model.named_parameters()) \
                + list(model.named_buffers()):
            if not t.dtype.is_floating_point or id(t) in seen:
                continue
            seen.add(id(t))
            self.shadow[name] = t.detach().clone().float()
            self._refs[name] = t

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, t in self._refs.items():
            self.shadow[name].mul_(self.decay).add_(
                t.detach().float(), alpha=1 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        self.backup = {name: t.detach().clone()
                       for name, t in self._refs.items()}
        for name, t in self._refs.items():
            t.copy_(self.shadow[name].to(t.dtype))

    @torch.no_grad()
    def restore(self, model: nn.Module):
        for name, t in self._refs.items():
            t.copy_(self.backup[name])
        self.backup = {}


class RunLogger:
    """v2: unified logging — always CSV; TensorBoard / wandb when installed
    and requested.  Only rank 0 writes."""

    def __init__(self, out_dir: Optional[str] = None, use_tb: bool = False,
                 use_wandb: bool = False, wandb_cfg: Optional[Dict] = None,
                 enabled: bool = True):
        self.rows: List[Dict] = []
        self.enabled = enabled
        self.out_dir = out_dir
        self.tb = self.wb = None
        if not enabled:
            return
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            self.csv_path = os.path.join(out_dir, "log.csv")
            with open(self.csv_path, "w") as f:
                f.write("step,loss,bpc,lr,tok_s,grad_norm,split\n")
        if use_tb:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb = SummaryWriter(out_dir or "runs/rwkv8")
            except Exception as e:
                print(f"[rwkv8 v2] tensorboard unavailable: {e}")
        if use_wandb:
            try:
                import wandb
                self.wb = wandb
                wandb.init(project="rwkv8-heron", config=wandb_cfg or {},
                           reinit=True)
            except Exception as e:
                print(f"[rwkv8 v2] wandb unavailable: {e}")

    def log(self, step: int, **kv):
        if not self.enabled:
            return
        kv["step"] = step
        self.rows.append(kv)
        if self.out_dir:
            with open(self.csv_path, "a") as f:
                f.write(",".join(str(kv.get(c, "")) for c in
                                 ["step", "loss", "bpc", "lr", "tok_s",
                                  "grad_norm", "split"]) + "\n")
        if self.tb:
            for k, v in kv.items():
                if isinstance(v, (int, float)):
                    self.tb.add_scalar(k, v, step)
        if self.wb:
            self.wb.log({k: v for k, v in kv.items()
                         if isinstance(v, (int, float))}, step=step)

    def close(self):
        if self.tb:
            self.tb.close()
        if self.wb:
            self.wb.finish()


def setup_distributed() -> Tuple[int, int, int]:
    """v2: initialize DDP from the env (torchrun).  Returns (rank,
    world_size, device).  Single-process runs get (0, 1, cuda/cpu)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ \
            and int(os.environ["WORLD_SIZE"]) > 1:
        import torch.distributed as dist
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        rank = dist.get_rank(); ws = dist.get_world_size()
        torch.cuda.set_device(rank % torch.cuda.device_count()) \
            if torch.cuda.is_available() else None
        dev = f"cuda:{rank % max(1, torch.cuda.device_count())}" \
            if torch.cuda.is_available() else "cpu"
        return rank, ws, dev
    return 0, 1, "cuda" if torch.cuda.is_available() else "cpu"


def _param_groups(model: nn.Module, lr: float, weight_decay: float,
                  betas=(0.9, 0.95)):
    """v2: decay only big matrices; norms / mixing scalars / gates / LoRA /
    embeddings stay decay-free (the RWKV convention)."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        leaf = name.split(".")[-1]
        if p.dim() < 2 or leaf in ("x_q", "x_k", "x_v", "x_r", "x_w", "x_g",
                                   "x_a", "w0", "a0", "v0", "k_k", "k_a",
                                   "r_k", "gb", "emb0", "emb1", "len_scale",
                                   "len_bias", "null_emb"):
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}], \
        {"lr": lr, "betas": betas}


def _make_sched(opt, kind: str, steps: int, warmup: int, lr: float,
                min_lr: float, restarts: int = 1):
    """v2 schedules: 'cosine' (default), 'wsd' (warmup-stable-decay),
    'cosine_restarts' (warmup + `restarts` equal cosine cycles)."""
    def cosine(step, total):
        p = step / max(1, total)
        return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * p))

    def fn(step):
        if step < warmup:
            return lr * (step + 1) / warmup
        s = step - warmup
        if kind == "wsd":
            decay_steps = max(1, steps - warmup)
            if s < 0.8 * decay_steps:
                return lr
            p = (s - 0.8 * decay_steps) / (0.2 * decay_steps)
            return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * p))
        if kind == "cosine_restarts":
            cyc = max(1, restarts)
            per = max(1, (steps - warmup)) // cyc
            cs, ci = s % per, s // per
            return cosine(cs, per) * (0.9 ** ci)
        return cosine(s, steps - warmup)                     # "cosine"

    return torch.optim.lr_scheduler.LambdaLR(opt, lambda s: fn(s) / lr)


def _warmup_cosine(step: int, total: int, warmup: int, base: float,
                   minimum: float) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return minimum + 0.5 * (base - minimum) * (1 + math.cos(math.pi * p))


@torch.no_grad()
def eval_bpc(model: nn.Module, data: ByteData, B: int = 8, T: int = 512,
             steps: int = 20, device: str = "cpu",
             deterministic: bool = True, eval_ctx: Optional[int] = None) \
        -> float:
    """v2: bits-per-char on the val split.  deterministic=True scores fixed,
    evenly spaced windows (variance-free comparisons across runs — v1 used
    random windows and a fragile async-flag restore).  eval_ctx overrides T
    if you want a context other than the training length."""
    model.eval()
    T = eval_ctx or T
    was_async = []
    for blk in getattr(model, "blocks", []):
        if getattr(blk, "use_rosa", False) and blk.rosa is not None:
            was_async.append((blk.rosa, blk.rosa.use_async))
            blk.rosa.use_async = False   # eval: no gradient, sync is fine
    try:
        losses = []
        arr = data.val
        if deterministic and len(arr) > T + 1:
            span = len(arr) - T - 1
            idxs = np.linspace(0, span, num=steps * B, endpoint=False)\
                .astype(np.int64)
            for s0 in range(0, len(idxs), B):
                ix = idxs[s0:s0 + B]
                xb = np.stack([arr[i:i + T] for i in ix]).astype(np.int64)
                yb = np.stack([arr[i + 1:i + T + 1] for i in ix])\
                    .astype(np.int64)
                x = torch.from_numpy(xb).to(device)
                y = torch.from_numpy(yb).to(device)
                logits = model(x)
                losses.append(F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
                    .item())
        else:
            for _ in range(steps):
                x, y = data.sample_batch(B, T, device, split="val")
                logits = model(x)
                losses.append(F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
                    .item())
        return float(np.mean(losses) / math.log(2))   # nats -> bits
    finally:
        for rosa, flag in was_async:
            rosa.use_async = flag
        model.train()


def train_model(model: nn.Module, data, steps: int = 1000,
                batch_size: int = 32, ctx_len: Optional[int] = None,
                lr: float = 8e-4, warmup: int = 100, min_lr: float = 1e-5,
                weight_decay: float = 0.01, betas=(0.9, 0.95),
                clip: float = 1.0, amp: bool = False, device: str = "cuda",
                val_every: int = 200, sample_fn=None, sample_every: int = 500,
                checkpoint_path: Optional[str] = None,
                log_every: int = 20, label: str = "model",
                rosa_prefetch: bool = True, quiet: bool = False,
                # ---- v2 ----
                sched: str = "cosine", restarts: int = 1,
                accum_steps: int = 1, ddp: bool = False,
                ema_decay: float = 0.0, sample_mode: str = "random",
                logger: Optional[RunLogger] = None,
                resume: bool = False, rank: int = 0, world: int = 1):
    """v2 trainer: AdamW + warmup-cosine/WSD/restart schedules, weight-decay
    parameter groups, gradient accumulation, optional DDP (call under
    torchrun with ddp=True), EMA, sequential data mode, CSV/TB/wandb
    logging, and RESUME-ABLE checkpoints (optimizer + scheduler + scaler +
    step + RNG states; the v1 checkpoints could not resume).

    Args mostly as v1.  New:
      sched        "cosine" | "wsd" | "cosine_restarts"
      restarts     cycles for cosine_restarts
      accum_steps  gradient accumulation (effective batch = batch_size * acc)
      ddp          wrap in DistributedDataParallel (torchrun sets the env)
      ema_decay    > 0 keeps an EMA; val uses the EMA weights, then restores
      sample_mode  "random" (v1 crops) | "sequential" (SequentialData walk)
      logger       RunLogger instance (csv/tb/wandb)
      resume       load `checkpoint_path` state (model+opt+sched+step) first
    """
    if ctx_len is None:
        ctx_len = getattr(model.cfg, "ctx_len", 256)
    model.to(device).train()
    groups, opt_kw = _param_groups(model, lr, weight_decay, betas)
    opt = torch.optim.AdamW(groups, **opt_kw)
    sched_lam = _make_sched(opt, sched, steps, warmup, lr, min_lr, restarts)
    scaler = torch.amp.GradScaler(enabled=amp and device == "cuda")
    uses_tok = getattr(model, "rosa_tok", None) is not None
    is_seq = sample_mode == "sequential" and hasattr(data, "next_batch")
    ema = EMA(model, ema_decay) if ema_decay > 0 else None

    if ddp and world > 1:
        import torch.distributed as dist
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[rank] if device.startswith("cuda") else None)
    core = model.module if hasattr(model, "module") else model

    start_step = 0
    best_val = float("inf")
    if resume and checkpoint_path and os.path.exists(checkpoint_path):
        ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
        core.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched_lam.load_state_dict(ck["sched"])
        if ck.get("scaler") is not None and amp:
            scaler.load_state_dict(ck["scaler"])
        start_step = ck["step"] + 1
        best_val = ck.get("best_val", float("inf"))
        if rank == 0:
            print(f"[{label}] resumed from {checkpoint_path} "
                  f"at step {start_step}")
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(ck.get("cuda_rng"))

    def sample_job():
        if is_seq:
            x, y = data.next_batch(batch_size, ctx_len, device)
        else:
            x, y = data.sample_batch(batch_size, ctx_len, device, "train")
        fut = None
        if uses_tok and rosa_prefetch and core.cfg.rosa_backend == "cpu":
            xn = x.detach().to("cpu").numpy().astype(np.int32)
            fut = _prefetch_pool().submit(rosa_token_batch_np, xn,
                                          core.cfg.vocab_size)
        return x, y, fut

    cur = sample_job()
    t0 = time.time(); tokens_seen = 0
    gnorm = 0.0
    v = None
    val_ran = False
    for step in range(start_step, steps):
        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for _ in range(max(1, accum_steps)):
            nxt = sample_job()
            x, y, fut = cur
            rosa_tok = None
            if fut is not None:
                sym_np, ln_np = fut.result()
                rosa_tok = (torch.from_numpy(sym_np).to(device),
                            torch.from_numpy(ln_np).to(device))
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=amp and device == "cuda"):
                logits = model(x, rosa_tok=rosa_tok) \
                    if uses_tok else model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), y.reshape(-1)) \
                    / max(1, accum_steps)
            scaler.scale(loss).backward()
            loss_acc += loss.item()
            cur = nxt
        scaler.unscale_(opt)
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip))
        scaler.step(opt)
        scaler.update()
        sched_lam.step()
        tokens_seen += batch_size * ctx_len
        if ema is not None:
            ema.update(core)
        if rank == 0 and (step % log_every == 0 or step == steps - 1):
            bpc = loss_acc / math.log(2)
            dt = time.time() - t0
            if not quiet:
                print(f"[{label}] step {step+1}/{steps} "
                      f"| loss {loss_acc:.4f} | {bpc:.3f} bpc "
                      f"| lr {sched_lam.get_last_lr()[0]:.2e} | gnorm "
                      f"{gnorm:.2f} | {tokens_seen/max(1e-9, dt):,.0f} tok/s")
            if logger:
                logger.log(step, loss=loss_acc, bpc=bpc,
                           lr=sched_lam.get_last_lr()[0], tok_s=tokens_seen
                           / max(1e-9, dt), grad_norm=gnorm, split="train")
        if val_every and (step % val_every == 0 or step == steps - 1) \
                and rank == 0:
            if ema is not None:
                ema.copy_to(core)
            v = eval_bpc(core, data, device=device)
            if ema is not None:
                ema.restore(core)
            best_val = min(best_val, v)
            val_ran = True
            if not quiet:
                print(f"[{label}]     val {v:.4f} bpc (best {best_val:.4f})")
            if logger:
                logger.log(step, bpc=v, split="val")
        if sample_fn and sample_every and (step % sample_every == 0 or
                                           step == steps - 1) and rank == 0:
            if not quiet:
                print(sample_fn())
            model.train()
        if checkpoint_path and rank == 0 and (step % 500 == 0 or
                                              step == steps - 1):
            torch.save({"model": core.state_dict(),
                        "cfg": asdict(core.cfg),
                        "opt": opt.state_dict(),
                        "sched": sched_lam.state_dict(),
                        "scaler": scaler.state_dict() if amp else None,
                        "step": step, "best_val": best_val},
                       checkpoint_path)
            if val_ran and v == best_val:
                # best-val snapshot (weights only, small)
                torch.save({"model": core.state_dict(),
                            "cfg": asdict(core.cfg), "step": step},
                           checkpoint_path + ".best")
                val_ran = False
    if logger:
        logger.close()
    return best_val


# ----------------------------------------------------------------------------
# 4.3  Generation
# ----------------------------------------------------------------------------

def sample_logits(logits: Tensor, temperature: float = 1.0,
                  top_p: float = 1.0, top_k: int = 0,
                  repetition_penalty: float = 1.0,
                  presence_penalty: float = 0.0,
                  recent_ids: Optional[List[int]] = None,
                  min_p: float = 0.0, generator: Optional[torch.Generator] = None
                  ) -> int:
    """v2 sampler (port of the RWKV helper in [1], rewritten).

    Order of operations (the v1 version applied temperature AFTER top-p and
    could raise IndexError when top_p -> 1; both fixed):
      1. penalties on raw logits (repetition / presence, over `recent_ids`)
      2. temperature scaling in LOG space (no underflowing power trick)
      3. top-k filter, then top-p (clamped), then min-p
      4. softmax + multinomial
    """
    if logits.dim() > 1:
        logits = logits.reshape(-1)
    logits = logits.float().clone()
    # 1. penalties -----------------------------------------------------------
    if (repetition_penalty and repetition_penalty != 1.0 or
            presence_penalty and presence_penalty != 0.0) \
            and recent_ids is not None and len(recent_ids) > 0:
        ids = torch.as_tensor(list(recent_ids), dtype=torch.long,
                              device=logits.device)
        ids = ids[ids >= 0]
        if ids.numel() > 0:
            if repetition_penalty and repetition_penalty != 1.0:
                seen, counts = torch.unique(ids, return_counts=True)
                lg = logits[seen]
                logits[seen] = torch.where(
                    lg > 0, lg / repetition_penalty, lg * repetition_penalty)
            if presence_penalty and presence_penalty != 0.0:
                seen = torch.unique(ids)
                logits[seen] -= presence_penalty
    # 2. temperature in log space --------------------------------------------
    if temperature <= 0:
        return int(logits.argmax().item())
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_ids = torch.sort(probs, descending=True)
    # 3. filters ---------------------------------------------------------------
    if top_k and top_k > 0:
        sorted_probs[top_k:] = 0.0
    if top_p < 1.0:
        cum = torch.cumsum(sorted_probs, dim=-1)
        idx = int(torch.searchsorted(cum, torch.tensor(top_p)).item())
        idx = min(idx, sorted_probs.shape[0] - 1)      # v1 IndexError fix
        cutoff = sorted_probs[idx]
        sorted_probs[sorted_probs < cutoff] = 0.0
    if min_p and min_p > 0.0:
        sorted_probs[sorted_probs < min_p * sorted_probs[0]] = 0.0
    # 4. sample ----------------------------------------------------------------
    sorted_probs = sorted_probs / sorted_probs.sum()
    nxt = sorted_ids[torch.multinomial(sorted_probs, 1, generator=generator)]
    return int(nxt.item())


@torch.no_grad()
def generate(model: nn.Module, prompt: List[int], n_new: int = 200,
             temperature: float = 0.8, top_p: float = 0.9, top_k: int = 0,
             device: str = "cpu", mode: str = "rnn",
             repetition_penalty: float = 1.1, presence_penalty: float = 0.0,
             min_p: float = 0.0, seed: Optional[int] = None,
             ctx_limit: Optional[int] = None) -> List[int]:
    """v2 generation.  mode="rnn" (default): O(T) total — prefill the prompt
    once through the stateful path, then one token per step (v1 recomputed
    the whole context every token, O(n^2), and silently capped at 2048).
    mode="gpt": the simple recompute path, kept for validation; ctx_limit
    caps the context explicitly (None = no cap)."""
    model.eval()
    gen = None
    if seed is not None:
        gen = torch.Generator(device="cpu"); gen.manual_seed(int(seed))
    toks = [int(t) for t in prompt]
    sampler = dict(temperature=temperature, top_p=top_p, top_k=top_k,
                   repetition_penalty=repetition_penalty,
                   presence_penalty=presence_penalty, min_p=min_p,
                   generator=gen)
    if mode == "rnn":
        state = model.init_state(1, device)
        for t in toks:
            logits, state = model(torch.tensor([[t]], device=device),
                                  state=state)
        out = list(toks)
        recent = list(toks[-256:])
        for _ in range(n_new):
            nxt = sample_logits(logits[0, -1], recent_ids=recent, **sampler)
            toks.append(nxt); out.append(nxt)
            recent.append(nxt)
            if len(recent) > 256:
                recent.pop(0)
            logits, state = model(torch.tensor([[nxt]], device=device),
                                  state=state)
        return out
    # ---- GPT-mode (recompute; validation reference) ----
    for _ in range(n_new):
        ctx = toks if ctx_limit is None else toks[-ctx_limit:]
        logits = model(torch.tensor([ctx], device=device))
        nxt = sample_logits(logits[0, -1], recent_ids=ctx[-256:], **sampler)
        toks.append(nxt)
    return toks


@torch.no_grad()
def generate_rnn(model: nn.Module, prompt: List[int], n_new: int = 200,
                 temperature: float = 0.8, top_p: float = 0.9, top_k: int = 0,
                 device: str = "cpu", state: Optional[Dict] = None):
    """Stateful RNN-mode generation: O(1) memory per token.  The RWKV-7g
    states, the live token suffix automaton and the per-group QKV automata
    are carried across steps — RWKV's superpower, with ROSA along for the
    ride.  (Kept for backward compatibility; `generate(mode="rnn")` is the
    recommended entry point.)"""
    model.eval()
    if state is None:
        state = model.init_state(1, device)
    toks = [int(t) for t in prompt]
    for t in prompt:
        _, state = model(torch.tensor([[t]], device=device), state=state)
    out = list(prompt)
    for _ in range(n_new):
        logits, state = model(
            torch.tensor([[toks[-1]]], device=device), state=state)
        nxt = sample_logits(logits[0, -1], temperature, top_p, top_k)
        toks.append(nxt)
        out.append(nxt)
    return out, state


# ----------------------------------------------------------------------------
# 4.4  Evaluation extras
# ----------------------------------------------------------------------------

@torch.no_grad()
def mqar_rosa_eval(B: int = 64, T: int = 256, n_pairs: int = 16,
                   vocab: int = 64) -> float:
    """Parameter-free MQAR accuracy of plain token-level ROSA: at each
    query-key position, does ROSA(x)[t] equal the value that followed the
    key's first occurrence?  (Should be ~100% by construction [1].)"""
    xs, answers = make_mqar_batch(B, T, n_pairs, vocab)
    sym, ln = rosa_token_batch_torch(xs)
    good, total = 0, 0
    for b in range(B):
        for (qpos, val) in answers[b]:
            total += 1
            if sym[b, qpos - 1].item() == val:  # ROSA output at query key pos
                good += 1
    return good / max(1, total)


@torch.no_grad()
def needle_rosa_eval(needle: str = "the best thing to do in San Francisco is",
                     filler_len: int = 2000) -> bool:
    """NIAH by design: ROSA retrieves the needle's continuation verbatim,
    regardless of haystack length ([1])."""
    vocab: Dict[str, int] = {}

    def enc(s: str) -> List[int]:
        return [vocab.setdefault(ch, len(vocab)) for ch in s]

    rng = random.Random(0)
    hay = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz ") for _ in
                  range(filler_len))
    text = f"{needle}. {hay} {needle} "
    ids = enc(text)
    sym, _ = rosa_token_batch_np(np.array([ids], np.int32), len(vocab) + 1)
    sym = sym[0]
    pos = len(ids) - 2                # last char of the repeated needle
    truth_pos = len(enc(needle))      # char right after the 1st occurrence
    return int(sym[pos]) == ids[truth_pos]


def standalone_rosa_bpc(data: ByteData, max_chars: int = 300_000,
                        mix_cap: float = 0.97) -> float:
    """ROSA as a standalone, parameter-free probabilistic compressor (the
    ROSA+ idea of [6]): blend ROSA's exact next-symbol prediction with an
    online order-1 (bigram) backoff, weighted by match length (longer
    repeated suffix = more confidence — Witten-Bell-flavoured), and score
    the actual bytes.  Pure CPU, no training — a reference point for the
    neural models."""
    text = data.val[:max_chars].astype(np.int64)
    V = 256
    bigram = np.ones((V, V), dtype=np.float64) * 0.1   # smoothed counts
    sym, ln = rosa_token_batch_np(text[None, :], V)
    sym = sym[0]; ln = ln[0]
    bits = 0.0
    prev = text[0]
    for t in range(len(text) - 1):
        nxt = text[t + 1]
        p_big = bigram[prev] / bigram[prev].sum()
        if sym[t] >= 0:
            lam = min(mix_cap, ln[t] / (ln[t] + 2.0))  # confidence by len
            p = lam * (1.0 if sym[t] == nxt else 0.0) + (1 - lam) * p_big[nxt]
        else:
            p = p_big[nxt]
        p = max(p, 1e-9)
        bits += -math.log2(p)
        bigram[prev, nxt] += 1
        prev = nxt
    return bits / (len(text) - 1)


def artifact_report(model: nn.Module) -> Dict:
    """Parameter-golf-style artifact report ([11]): parameter count and
    checkpoint sizes vs the 16MB budget."""
    n = param_count(model)
    rep = {
        "params": n,
        "fp32_MB": n * 4 / 1e6,
        "fp16_MB": n * 2 / 1e6,
        "int8_MB": n * 1 / 1e6,
        "fits_parameter_golf_16MB_fp16": n * 2 <= 16e6,
    }
    print(f"parameters: {n:,}")
    print(f"artifact size: fp32 {rep['fp32_MB']:.1f} MB | "
          f"fp16 {rep['fp16_MB']:.1f} MB | int8 {rep['int8_MB']:.1f} MB")
    print(f"parameter-golf 16MB budget (fp16): "
          f"{'YES' if rep['fits_parameter_golf_16MB_fp16'] else 'NO'}")
    return rep


# ----------------------------------------------------------------------------
# 4.5  Showcase-task evaluation helpers (used by the notebook demo cells)
# ----------------------------------------------------------------------------

@torch.no_grad()
def task_accuracy(model: nn.Module, batch: Tensor) -> float:
    """Next-token argmax accuracy of a model on a [B,T] token batch."""
    model.eval()
    x, y = batch[:, :-1], batch[:, 1:]
    pred = model(x).argmax(-1)
    model.train()
    return (pred == y).float().mean().item()


def raw_rosa_accuracy(batch: Tensor) -> Tuple[float, float]:
    """Accuracy of the parameter-free token-level ROSA predictor vs the next
    token, and the fraction of positions with any match at all."""
    sym, ln = rosa_token_batch_torch(batch)
    y = batch[:, 1:]
    pred = sym[:, :-1]
    mask = pred >= 0
    acc = (pred[mask] == y[mask]).float().mean().item() if mask.any() else 0.0
    return acc, mask.float().mean().item()


@torch.no_grad()
def reverse_accuracy(model: nn.Module, digits: int, n_samples: int = 64,
                     T: int = 128, device: str = "cpu",
                     digits_max: Optional[int] = None) -> float:
    """Digit accuracy on the reversed part of the 251105-style task [1]."""
    model.eval()
    good, total = 0, 0
    B = 16
    for s0 in range(0, n_samples, B):
        b = min(B, n_samples - s0)
        xb = make_reverse_batch(b, T, digits=digits, device=device)
        pred = model(xb[:, :-1]).argmax(-1)
        for i in range(b):
            row = xb[i].tolist()
            p1 = row.index(10)                    # ','
            p2 = row.index(11)                    # '#'
            # model input up to p2-1 predicts tokens 1..p2-1; the reversed
            # digits occupy p1+1 .. p2
            gold = row[p1 + 1:p2 + 1]
            got = pred[i, p1:p2].tolist()
            good += sum(int(a == b2) for a, b2 in zip(gold, got))
            total += len(gold)
    model.train()
    return good / max(1, total)


# ==========================================================================
# >>> section from p5_serve.py (v2)
# ==========================================================================

# ============================================================================
# Section 5 — Tokenization, chat, state IO, quantization, export, benchmarks
#   5.1  Tokenizer (bytes | tiktoken | HF) + chat bookmark tokens
#   5.2  Streaming chat CLI (RNN-mode)
#   5.3  RNN-state save / load (RWKV state + live suffix automata)
#   5.4  Quantization (int8 dynamic / fp16) + quality report
#   5.5  Export (safetensors / ONNX)
#   5.6  Benchmark harness (full-val bpc, per-position loss, NIAH sweep)
# ============================================================================


# ----------------------------------------------------------------------------
# 5.1  Tokenizer + chat bookmarks
# ----------------------------------------------------------------------------

class Tokenizer:
    """v2: pluggable tokenizer.  Backends:
      * "bytes"   — raw UTF-8 bytes; special tokens get ids >= 256 (so a
                    byte model can still be trained with chat bookmarks).
      * "tiktoken"— any tiktoken BPE by name (e.g. "gpt2"); requires the
                    `tiktoken` package.
      * "hf"      — any HuggingFace tokenizer via transformers.
    The model just sees vocab_size = tokenizer.vocab_size; RosaTokenEmb and
    the ROSA automata are token-id-agnostic and work unchanged."""

    SPECIAL_TOKENS = ["<pad>", "<sys>", "<user>", "<asst>",
                      "<C_begin>", "<C_end>"]

    def __init__(self, backend: str = "bytes", name: Optional[str] = None,
                 special_tokens: bool = True):
        assert backend in ("bytes", "tiktoken", "hf")
        self.backend = backend
        self.name = name
        self.special = list(self.SPECIAL_TOKENS) if special_tokens else []
        self.stoi: Dict[str, int] = {t: 256 + i
                                     for i, t in enumerate(self.special)}
        if backend == "tiktoken":
            import tiktoken
            self.enc = tiktoken.get_encoding(name or "gpt2")
            self._base_vocab = self.enc.n_vocab
        elif backend == "hf":
            from transformers import AutoTokenizer
            self.enc = AutoTokenizer.from_pretrained(name)
            self._base_vocab = len(self.enc)
        else:
            self._base_vocab = 256

    @property
    def vocab_size(self) -> int:
        return self._base_vocab + len(self.special)

    def encode(self, text: str) -> List[int]:
        ids: List[int] = []
        # split out special tokens first (byte backend needs the offset ids)
        if self.special:
            import re
            pat = "(" + "|".join(re.escape(t) for t in self.special) + ")"
            parts = re.split(pat, text)
        else:
            parts = [text]
        for part in parts:
            if not part:
                continue
            if part in self.stoi:
                ids.append(self.stoi[part])
            elif self.backend == "bytes":
                ids.extend(part.encode("utf-8"))
            elif self.backend == "tiktoken":
                ids.extend(self.enc.encode(part,
                                           allowed_special="all"
                                           if self.name == "cl100k_base"
                                           else set()))
            else:
                ids.extend(self.enc.encode(part))
        return ids

    def decode(self, ids: List[int]) -> str:
        inv = {v: k for k, v in self.stoi.items()}
        out = []
        buf: List[int] = []
        for i in ids:
            if i in inv:
                if buf:
                    out.append(self._decode_base(buf)); buf = []
                out.append(inv[i])
            else:
                buf.append(int(i))
        if buf:
            out.append(self._decode_base(buf))
        return "".join(out)

    def _decode_base(self, ids: List[int]) -> str:
        if self.backend == "bytes":
            return bytes(max(0, min(255, i)) for i in ids).decode(
                "utf-8", errors="replace")
        if self.backend == "tiktoken":
            return self.enc.decode([int(i) for i in ids])
        return self.enc.decode([int(i) for i in ids])


def encode_chat(tokenizer: Tokenizer, system: Optional[str],
                rounds: List[Tuple[str, str]], add_bookmarks: bool = True) \
        -> List[int]:
    """v2: encode a chat with BOOKMARK tokens ([1]'s RWKV-8 notes: hardcoded
    <C_begin>/<C_end> markers let ROSA retrieve any round perfectly).  Each
    round is wrapped so the token-level ROSA automaton keys on the single
    bookmark ids — exact, lossless conversational recall."""
    ids: List[int] = []
    if system:
        ids += tokenizer.encode(f"<sys>{system}</sys>") \
            if tokenizer.special and add_bookmarks \
            else tokenizer.encode(system)
    for user, asst in rounds:
        if tokenizer.special and add_bookmarks:
            ids += tokenizer.encode("<C_begin><user>") + \
                tokenizer.encode(user) + tokenizer.encode("<asst>") + \
                tokenizer.encode(asst) + tokenizer.encode("<C_end>")
        else:
            ids += tokenizer.encode(user + asst)
    return ids


# ----------------------------------------------------------------------------
# 5.2  Streaming chat CLI
# ----------------------------------------------------------------------------

@torch.no_grad()
def chat(model: nn.Module, tokenizer: Optional[Tokenizer] = None,
         system: Optional[str] = None, max_new: int = 200,
         temperature: float = 0.8, top_p: float = 0.9, top_k: int = 0,
         min_p: float = 0.0, repetition_penalty: float = 1.1,
         device: str = "cpu", bookmarks: bool = True,
         save_state_path: Optional[str] = None) -> None:
    """v2: interactive REPL over the stateful RNN-mode (O(1)/token,
    unbounded context — ROSA keeps retrieving across the whole conversation).
    Streams token ids to the console as they are sampled.  Ctrl-C / Ctrl-D
    exits; 'save' writes the conversation state to disk."""
    model.eval()
    state = model.init_state(1, device)
    vocab = model.cfg.vocab_size

    def decode(ids: List[int]) -> str:
        return tokenizer.decode(ids) if tokenizer else \
            bytes(i % 256 for i in ids).decode("utf-8", errors="replace")

    def feed(text_ids: List[int]) -> None:
        nonlocal state
        for t in text_ids:
            _, state = model(torch.tensor([[t]], device=device), state=state)

    if system:
        sys_ids = tokenizer.encode(f"<sys>{system}</sys>") \
            if tokenizer and tokenizer.special and bookmarks \
            else (tokenizer.encode(system) if tokenizer
                  else list(system.encode("utf-8")))
        feed(sys_ids)
        print(f"[system loaded: {len(sys_ids)} tokens]")
    print("chat ready — type a message ('save' checkpoints the state, "
          "Ctrl-D exits)")
    while True:
        try:
            line = input("\n> ")
        except (KeyboardInterrupt, EOFError):
            break
        if line.strip() == "save":
            if save_state_path:
                save_rnn_state(state, save_state_path)
                print(f"state saved to {save_state_path}")
            continue
        if not line.strip():
            continue
        msg_ids = tokenizer.encode(f"<C_begin><user>{line}<asst>") \
            if tokenizer and tokenizer.special and bookmarks \
            else (tokenizer.encode(line) if tokenizer
                  else list(line.encode("utf-8")))
        feed(msg_ids)
        gen: List[int] = []
        recent = list(msg_ids[-256:])
        for _ in range(max_new):
            logits, state = model(torch.tensor([[msg_ids[-1]]],
                                               device=device), state=state)
            nxt = sample_logits(logits[0, -1], temperature=temperature,
                                top_p=top_p, top_k=top_k, min_p=min_p,
                                repetition_penalty=repetition_penalty,
                                recent_ids=recent)
            gen.append(nxt); recent.append(nxt)
            if len(recent) > 256:
                recent.pop(0)
            print(decode([nxt]), end="", flush=True)
            if tokenizer is None and nxt == 10:      # byte mode: newline end
                break
            msg_ids = [nxt]
        print()


# ----------------------------------------------------------------------------
# 5.3  RNN-state save / load
# ----------------------------------------------------------------------------

def save_rnn_state(state: Dict, path: str) -> None:
    """v2: serialize a full RNN-mode state — RWKV recurrent states, token
    automata and the live QKV suffix automata — so a conversation can be
    paused, shipped and resumed (or branched)."""
    blob: Dict = {"pos": state.get("pos", 0),
                  "v_first": state.get("v_first")}
    blob["S"] = [s.cpu() for s in state["S"]] if state.get("S") else None
    for key in RWKV8Block._PREV_KEYS:
        vals = []
        for v in state.get(key, []):
            vals.append(v.cpu() if torch.is_tensor(v) else v)
        blob[key] = vals
    if state.get("tok_sam") is not None:
        blob["tok_sam"] = [_token_sam_dump(s) for s in state["tok_sam"]]
    if state.get("sams") is not None:
        dumped = []
        for sams in state["sams"]:
            if isinstance(sams, BatchedQKVSams):
                dumped.append({"batched": sams.dump()})
            else:
                dumped.append({"list": None})
        blob["sams"] = dumped
    torch.save(blob, path)


def load_rnn_state(path: str, model: nn.Module, device: str = "cpu") -> Dict:
    """Inverse of save_rnn_state: rebuild a live state for `model`."""
    blob = torch.load(path, map_location=device, weights_only=False)
    cfg = model.cfg
    batch = 1
    if blob.get("tok_sam"):
        batch = len(blob["tok_sam"])
    state = model.init_state(batch, device)
    state["pos"] = blob.get("pos", 0)
    if blob.get("S") is not None:
        state["S"] = [s.to(device) for s in blob["S"]]
    for key in RWKV8Block._PREV_KEYS:
        vals = []
        for v in blob.get(key, []):
            vals.append(v.to(device) if torch.is_tensor(v) else v)
        if vals:
            state[key] = vals
    if blob.get("tok_sam"):
        for s, d in zip(state["tok_sam"], blob["tok_sam"]):
            _token_sam_load(s, d)
    if blob.get("sams") and cfg.use_qkv_rosa:
        for li, d in enumerate(blob["sams"]):
            if "batched" in d and isinstance(state["sams"][li],
                                             BatchedQKVSams):
                state["sams"][li] = BatchedQKVSams.load(d["batched"])
    return state


def _token_sam_dump(s: "StatefulTokenSam") -> Dict:
    trans_keys, trans_vals = [], []
    for d in s.trans:
        for a, tgt in d.items():
            trans_keys.append((len(trans_keys) // 1, a, tgt))
    # store per-state dicts as flattened triples
    flat = []
    for st, d in enumerate(s.trans):
        for a, tgt in d.items():
            flat.append((st, int(a), int(tgt)))
    return {"vocab": s.vocab, "g": s.g, "n": s.n, "hist": s._hist,
            "link": s.link, "ln": s.ln, "last": s.last, "flat": flat}


def _token_sam_load(s: "StatefulTokenSam", d: Dict) -> None:
    s.trans = [{} for _ in range(len(d["link"]))]
    for st, a, tgt in d["flat"]:
        s.trans[st][a] = tgt
    s.link = d["link"]; s.ln = d["ln"]; s.last = d["last"]
    s.g = d["g"]; s.n = d["n"]; s._hist = d["hist"]


# ----------------------------------------------------------------------------
# 5.4  Quantization + quality report
# ----------------------------------------------------------------------------

def quantize_int8(model: nn.Module) -> nn.Module:
    """v2: dynamic int8 quantization of all Linear layers (CPU inference).
    Returns a NEW quantized copy; the input model is untouched."""
    model = copy.deepcopy(model).cpu().eval()
    ql = torch.ao.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8)
    return ql


def quant_quality_report(model: nn.Module, data: ByteData,
                         device: str = "cpu") -> Dict:
    """v2: measure the quality/space trade of fp16 and int8 vs fp32."""
    rep: Dict = {}
    base = eval_bpc(copy.deepcopy(model).to(device).eval(), data,
                    device=device)
    rep["fp32_bpc"] = base
    n = param_count(model)
    rep["fp32_MB"] = n * 4 / 1e6
    half = copy.deepcopy(model).half().eval()
    rep["fp16_bpc"] = eval_bpc(half, data, device="cpu")
    rep["fp16_MB"] = n * 2 / 1e6
    del half
    q = quantize_int8(model)
    rep["int8_bpc"] = eval_bpc(q, data, device="cpu")
    int8_bytes = sum(p.numel() * p.element_size() for p in q.state_dict()
                     .values() if torch.is_tensor(p))
    for name, mod in q.named_modules():
        if hasattr(mod, "_packed_params"):
            try:
                for pp in mod._packed_params.values() if isinstance(
                        mod._packed_params, dict) else []:
                    if torch.is_tensor(pp):
                        int8_bytes += pp.numel() * pp.element_size()
            except Exception:
                pass
    rep["int8_MB_approx"] = int8_bytes / 1e6
    rep["fp16_delta_bpc"] = rep["fp16_bpc"] - base
    rep["int8_delta_bpc"] = rep["int8_bpc"] - base
    print(f"fp32 {base:.4f} bpc | fp16 {rep['fp16_bpc']:.4f} "
          f"({rep['fp16_delta_bpc']:+.4f}) | int8 {rep['int8_bpc']:.4f} "
          f"({rep['int8_delta_bpc']:+.4f})")
    return rep


# ----------------------------------------------------------------------------
# 5.5  Export
# ----------------------------------------------------------------------------

def export_safetensors(model: nn.Module, path: str,
                       metadata: Optional[Dict] = None) -> str:
    """v2: export weights + config metadata as safetensors (falls back to
    torch.save when the package is missing)."""
    sd = {k: v.contiguous().cpu()
          for k, v in model.state_dict().items()
          if torch.is_tensor(v)}
    try:
        from safetensors.torch import save_file
        save_file(sd, path, metadata=metadata or {})
    except ImportError:
        torch.save({"state_dict": sd, "metadata": metadata}, path)
        path += ".pt (safetensors not installed — wrote torch archive)"
    return path


class _RosaExportWrapper(nn.Module):
    """GPT-mode forward with all ROSA retrieval indices as explicit INPUTS,
    so the whole neural stack exports cleanly to ONNX.  Compute the indices
    with the CPU automata offline, then run this wrapper anywhere."""

    def __init__(self, model: "RWKV8Heron"):
        super().__init__()
        self.model = model
        self.cfg = model.cfg

    def forward(self, idx: Tensor, *rosa_args) -> Tensor:
        m = self.model
        x = m.emb(idx)
        li = 0
        if m.rosa_tok is not None:
            sym, ln = rosa_args[0], rosa_args[1]
            li = 2
            x = x + m.rosa_tok(idx, sym, ln)
        x = m.ln0(x)
        v_first = None
        for b in m.blocks:
            xr = 0.0
            if getattr(b, "use_rosa", False) and b.rosa is not None:
                h = b.rosa.precompute(b.ln_c(x))
                h["fut"] = None
                sym, ln, pos = rosa_args[li], rosa_args[li + 1], \
                    rosa_args[li + 2]
                li += 3
                h["symM"] = h["lnM"] = h["posM"] = None
                if b.rosa.grad_mode == "soft":
                    symM, lnM, posM = rosa_qkv_batch_torch_m(
                        h["qsym"], h["ksym"], h["vsym"],
                        1 << b.rosa.nbits, b.rosa.K, M=b.rosa.soft_M)
                    h["symM"], h["lnM"], h["posM"] = symM, lnM, posM
                xr = b.rosa._assemble(h, sym, ln, pos)
            xx, v_first, _ = b.timemix(b.ln_a(x), v_first)
            x = x + xx + xr if b.use_rosa else x + xx
            if b.ffn_mode == "ffn":
                x = x + b.ffn(b.ln_b(x), idx)
            elif b.ffn_mode == "linear":
                x = x + b.ffn(b.ln_b(x))
        return m.head(m.ln_out(x))


def export_onnx(model: "RWKV8Heron", path: str, vocab_size: Optional[int]
                = None, opset: int = 17) -> str:
    """v2: export the GPT-mode neural core to ONNX.  ROSA retrieval indices
    are exported as INPUTS (the suffix automata themselves are integer
    algorithms, not tensors — run them on CPU and feed the results, which is
    exactly the deployment split the async pipeline already uses)."""
    model.eval()
    V = vocab_size or model.cfg.vocab_size
    C, T = model.cfg.n_embd, 16
    idx = torch.zeros(1, T, dtype=torch.long)
    dyn = {"idx": {0: "batch", 1: "time"}}
    extra = []
    if model.rosa_tok is not None:
        extra += [torch.full((1, T), -1, dtype=torch.long),
                  torch.zeros(1, T, dtype=torch.long)]
        dyn["rosa_sym"] = {0: "batch", 1: "time"}
        dyn["rosa_ln"] = {0: "batch", 1: "time"}
    for b in model.blocks:
        if getattr(b, "use_rosa", False) and b.rosa is not None:
            G = C // b.rosa.nbits
            extra += [torch.full((1, G, T), -1, dtype=torch.long),
                      torch.zeros(1, G, T, dtype=torch.long),
                      torch.zeros(1, G, T, dtype=torch.long)]
    wrapper = _RosaExportWrapper(model)
    for b in model.blocks:                       # no async futures on export
        if getattr(b, "use_rosa", False) and b.rosa is not None:
            b.rosa.use_async = False
    input_names = ["idx"] + [f"rosa_{i}" for i in range(len(extra))]
    dyn = {"idx": {0: "batch", 1: "time"}, "logits": {0: "batch"}}
    for i in range(len(extra)):
        dyn[f"rosa_{i}"] = {0: "batch", 2: "time"}
    torch.onnx.export(wrapper, (idx, *extra), path, opset_version=opset,
                      input_names=input_names, output_names=["logits"],
                      dynamic_axes=dyn)
    return path


# ----------------------------------------------------------------------------
# 5.6  Benchmark harness
# ----------------------------------------------------------------------------

@torch.no_grad()
def eval_bpc_full(model: nn.Module, data: ByteData, T: int = 256,
                  stride: int = 256, device: str = "cpu",
                  max_windows: int = 512) -> float:
    """v2: deterministic SLIDING-WINDOW bpc over the whole val split
    (stride=1 gives the classic compressor protocol; default stride=T for
    speed).  Tokens before each window warm the model up in GPT mode."""
    model.eval()
    arr = data.val
    bits, predicted = 0.0, 0
    starts = list(range(0, max(1, len(arr) - T - 1), stride))[:max_windows]
    for s0 in starts:
        x = torch.from_numpy(arr[s0:s0 + T].astype(np.int64)[None]).to(device)
        y = torch.from_numpy(
            arr[s0 + 1:s0 + T + 1].astype(np.int64)[None]).to(device)
        logits = model(x)
        lp = F.log_softmax(logits[0].float(), dim=-1)
        bits += -lp.gather(1, y[0].unsqueeze(1)).sum().item() / math.log(2)
        predicted += T
    return bits / max(1, predicted)


@torch.no_grad()
def per_position_losses(model: nn.Module, data: ByteData, T: int = 256,
                        n_windows: int = 32, device: str = "cpu") \
        -> np.ndarray:
    """v2: mean loss as a function of position within the context — shows
    how fast the recurrent state warms up (ROSA should make the curve drop
    almost immediately at repeated content)."""
    model.eval()
    arr = data.val
    tot = np.zeros(T); cnt = 0
    rng = np.random.default_rng(0)
    for _ in range(n_windows):
        s0 = int(rng.integers(0, len(arr) - T - 1))
        x = torch.from_numpy(arr[s0:s0 + T].astype(np.int64)[None]).to(device)
        y = torch.from_numpy(
            arr[s0 + 1:s0 + T + 1].astype(np.int64)[None]).to(device)
        logits = model(x)
        lp = F.log_softmax(logits[0].float(), dim=-1)
        nll = -lp.gather(1, y[0].unsqueeze(1)).squeeze(1) / math.log(2)
        tot += nll.cpu().numpy()
        cnt += 1
    return tot / max(1, cnt)


@torch.no_grad()
def niah_sweep(model: nn.Module, filler_lens: Optional[List[int]] = None,
               needle: str = "the best thing to do in San Francisco is",
               device: str = "cpu") -> Dict[int, bool]:
    """v2: needle-in-a-haystack at increasing haystack lengths — the RNN-mode
    superpower demo.  Returns {filler_len: retrieved?}.  ROSA is exact by
    construction, so every entry should be True regardless of length."""
    filler_lens = filler_lens or [100, 500, 1000, 5000, 10000, 50000, 100000]
    out: Dict[int, bool] = {}
    vocab: Dict[str, int] = {}

    def enc(s: str) -> List[int]:
        return [vocab.setdefault(ch, len(vocab)) for ch in s]

    rng = random.Random(0)
    truth_pos = len(enc(needle))
    for L in filler_lens:
        hay = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz ")
                      for _ in range(L))
        ids = enc(f"{needle}. {hay} {needle} ")
        sym, _ln = rosa_token_batch_np(np.array([ids], np.int32),
                                       len(vocab) + 1)
        pos = len(ids) - 2
        out[L] = int(sym[0][pos]) == ids[truth_pos]
    return out


# ----------------------------------------------------------------------------
# 5.7  Command-line entry point
# ----------------------------------------------------------------------------

def _cli() -> None:
    import argparse
    p = argparse.ArgumentParser(
        prog="python rwkv8.py",
        description="RWKV-8 'Heron' v2 — train / chat / benchmark / export")
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train", help="train on a text/byte file")
    pt.add_argument("data")
    pt.add_argument("--steps", type=int, default=1000)
    pt.add_argument("--preset", default="demo")
    pt.add_argument("--ctx", type=int, default=None)
    pt.add_argument("--batch", type=int, default=32)
    pt.add_argument("--lr", type=float, default=8e-4)
    pt.add_argument("--sched", default="cosine",
                    choices=["cosine", "wsd", "cosine_restarts"])
    pt.add_argument("--accum", type=int, default=1)
    pt.add_argument("--ema", type=float, default=0.0)
    pt.add_argument("--sample-mode", default="random",
                    choices=["random", "sequential"])
    pt.add_argument("--amp", action="store_true")
    pt.add_argument("--out", default="ckpt.pt")
    pt.add_argument("--resume", action="store_true")
    pt.add_argument("--val-bytes", type=int, default=1_000_000)
    pt.add_argument("--tokenizer", default=None,
                    help="'tiktoken:gpt2' or 'hf:org/name' — default bytes")

    pc = sub.add_parser("chat", help="interactive RNN-mode chat")
    pc.add_argument("ckpt")
    pc.add_argument("--tokenizer", default=None)
    pc.add_argument("--system", default=None)
    pc.add_argument("--max-new", type=int, default=200)
    pc.add_argument("--temp", type=float, default=0.8)
    pc.add_argument("--device", default=None)

    pg = sub.add_parser("generate", help="one-shot generation")
    pg.add_argument("ckpt")
    pg.add_argument("--prompt", default="First Citizen")
    pg.add_argument("--n", type=int, default=200)
    pg.add_argument("--mode", default="rnn", choices=["rnn", "gpt"])
    pg.add_argument("--device", default=None)

    pb = sub.add_parser("benchmark", help="bpc + per-position loss + NIAH")
    pb.add_argument("ckpt")
    pb.add_argument("data")
    pb.add_argument("--device", default=None)

    pe = sub.add_parser("export", help="export weights")
    pe.add_argument("ckpt")
    pe.add_argument("out")
    pe.add_argument("--format", default="safetensors",
                    choices=["safetensors", "onnx"])

    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def load_tok(spec):
        if not spec:
            return None
        kind, _, name = spec.partition(":")
        return Tokenizer(kind, name or None)

    if a.cmd == "train":
        tok = load_tok(a.tokenizer)
        data = (ByteData(a.data, val_bytes=a.val_bytes)
                if a.sample_mode == "random"
                else SequentialData(a.data, val_bytes=a.val_bytes))
        cfg = RWKV8Config.preset(a.preset, vocab_size=tok.vocab_size
                                 if tok else 256)
        model = RWKV8Heron(cfg)
        print(cfg.summary(), "|", param_count(model), "params")
        train_model(model, data, steps=a.steps, batch_size=a.batch,
                    ctx_len=a.ctx or cfg.ctx_len, lr=a.lr, sched=a.sched,
                    accum_steps=a.accum, ema_decay=a.ema, amp=a.amp,
                    device=dev, checkpoint_path=a.out, resume=a.resume,
                    logger=RunLogger(out_dir="runs/" + a.preset))
    else:
        ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
        model = RWKV8Heron(RWKV8Config(**{
            k: v for k, v in ck["cfg"].items() if k != "layer_rosa_bits"
            or v is not None}))
        model.load_state_dict(ck["model"])
        model.to(dev).eval()
        if a.cmd == "chat":
            chat(model, load_tok(a.tokenizer), system=a.system,
                 max_new=a.max_new, temperature=a.temp, device=dev)
        elif a.cmd == "generate":
            tok = load_tok(a.tokenizer)
            ids = tok.encode(a.prompt) if tok else \
                list(a.prompt.encode("utf-8"))
            out = generate(model, ids, n_new=a.n, device=dev, mode=a.mode,
                           temperature=0.8)
            new = out[len(ids):]
            print((tok.decode(new) if tok else
                   bytes(new).decode("utf-8", errors="replace")))
        elif a.cmd == "benchmark":
            data = ByteData(a.data, val_bytes=1_000_000)
            print("windowed bpc:", eval_bpc(model, data, device=dev))
            print("full-val bpc:", eval_bpc_full(model, data, device=dev))
            pp = per_position_losses(model, data, device=dev)
            print("per-position loss (first/last 8):",
                  pp[:8].round(3).tolist(), "...", pp[-8:].round(3).tolist())
            print("NIAH:", niah_sweep(model, filler_lens=[100, 1000, 10000]))
        elif a.cmd == "export":
            if a.format == "onnx":
                print("wrote", export_onnx(model, a.out))
            else:
                print("wrote", export_safetensors(
                    model, a.out, metadata={"arch": "rwkv8-heron-v2"}))


if __name__ == "__main__":
    _cli()
