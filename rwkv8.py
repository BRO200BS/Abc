# ==========================================================================
# >>> section from p1_engine.py
# ==========================================================================

# ============================================================================
# rwkv8.py — RWKV-8 "Heron" LLM with ROSA, completely in Python (PyTorch)
#
# A from-scratch, GPU-ready (Google Colab T4) implementation of:
#   * RWKV-7g time-mix  (exact port of BlinkDL's cuda/wkv7_cuda.cu to PyTorch)
#   * ROSA              (Rapid Online Suffix Automaton — RWKV-8's neurosymbolic
#                        retrieval mechanism that goes "beyond attention")
#   * DeepEmbed         (RWKV-8's per-token FFN modulation)
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
    embedding (how confident the retrieval is)."""

    def __init__(self, vocab: int, dim: int, len_buckets: int = 32):
        super().__init__()
        self.vocab = vocab
        self.len_buckets = len_buckets
        self.emb = nn.Embedding(vocab, dim)
        self.null_emb = nn.Parameter(torch.zeros(dim))
        self.len_emb = nn.Embedding(len_buckets, dim)
        nn.init.zeros_(self.emb.weight)          # start silent; learn to use
        nn.init.zeros_(self.len_emb.weight)

    def forward(self, idx: Tensor, rosa_sym: Optional[Tensor] = None,
                rosa_ln: Optional[Tensor] = None) -> Tensor:
        if rosa_sym is None or rosa_ln is None:
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
                 gate_bias: float = 2.0):
        super().__init__()
        assert dim % nbits == 0
        self.dim = dim
        self.nbits = nbits
        self.K = K
        self.grad_mode = grad_mode
        self.tau = tau
        self.use_async = bool(use_async and NUMBA_OK)
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

    def finalize(self, h: dict) -> Tensor:
        if h["fut"] is not None:
            sym_np, ln_np, pos_np = h["fut"].result()
            dev = h["q"].device
            B, G, T = h["qsym"].shape
            sym = torch.from_numpy(sym_np).to(dev).view(B, G, T)
            ln = torch.from_numpy(ln_np).to(dev).view(B, G, T)
            pos = torch.from_numpy(pos_np).to(dev).view(B, G, T)
        else:
            sym, ln, pos = rosa_qkv_batch_torch(
                h["qsym"], h["ksym"], h["vsym"], 1 << self.nbits, self.K)
        return self._assemble(h, sym, ln, pos)

    def forward(self, xn: Tensor) -> Tensor:
        return self.finalize(self.precompute(xn))

    # ---- RNN-mode (single token, live automata) ---------------------------
    def forward_step(self, xn1: Tensor, q: Tensor, k: Tensor, v: Tensor,
                     sams: List[StatefulQKVSam]) -> Tuple[Tensor, List[StatefulQKVSam]]:
        """xn1: [1,1,C] normalized input (for the read gate); q,k,v: [1,1,C]
        already-projected streams; sams: one StatefulQKVSam per group
        (extended in place)."""
        B, T, C = q.shape
        G = C // self.nbits
        qsym = pack_symbols(q, self.nbits)[0, 0]                # [G]
        ksym = pack_symbols(k, self.nbits)[0, 0]
        vsym = pack_symbols(v, self.nbits)[0, 0]
        syms = torch.full((G, 1), -1, dtype=torch.long, device=q.device)
        lns = torch.zeros(G, 1, dtype=torch.long, device=q.device)
        poss = torch.full((G, 1), -1, dtype=torch.long, device=q.device)
        for g in range(G):
            r_, ml_, rp_ = sams[g].extend(int(qsym[g]), int(ksym[g]),
                                          int(vsym[g]))
            syms[g, 0] = r_; lns[g, 0] = ml_; poss[g, 0] = rp_
        sym = syms.unsqueeze(0)                                 # [1,G,1]
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

def rwkv7g_scan(r: Tensor, w: Tensor, k: Tensor, v: Tensor, kk: Tensor,
                agate: Tensor, state: Optional[Tensor] = None,
                chunk: int = 32) -> Tuple[Tensor, Tensor]:
    """r,w,k,v,kk,agate: [B,T,H,N]; state: [B,H,N,N].  Returns (y [B,T,H,N],
    state_out).  Runs in float32 for accumulation stability (state kept in
    fp32 like the CUDA kernel); optional per-chunk gradient checkpointing to
    bound activation memory for long contexts."""
    B, T, H, N = r.shape
    orig_dtype = r.dtype
    with torch.autocast(device_type=r.device.type, enabled=False):
        r = r.float(); w = w.float(); k = k.float(); v = v.float()
        kk = kk.float(); agate = agate.float()
        ww = torch.exp(-torch.exp(w))                     # [B,T,H,N]
        bb = kk * agate                                   # [B,T,H,N]
        if state is None:
            state = r.new_zeros(B, H, N, N)

        def run_chunk(S_in, r_c, ww_c, k_c, v_c, kk_c, bb_c):
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

        use_ckpt = (chunk is not None and chunk > 0 and T > chunk
                    and torch.is_grad_enabled())
        if use_ckpt:
            ys = []
            for s0 in range(0, T, chunk):
                s1 = min(s0 + chunk, T)
                y_c, state = torch.utils.checkpoint.checkpoint(
                    run_chunk, state, r[:, s0:s1], ww[:, s0:s1], k[:, s0:s1],
                    v[:, s0:s1], kk[:, s0:s1], bb[:, s0:s1],
                    use_reentrant=False)
                ys.append(y_c)
            y = torch.cat(ys, dim=1)
        else:
            y, state = run_chunk(state, r, ww, k, v, kk, bb)
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
                 head_size: int = 16, scan_chunk: int = 32):
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
            state, self.scan_chunk)
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
    """

    def __init__(self, dim: int, dim_ffn: int, vocab: int,
                 deepembed: str = "1x"):
        super().__init__()
        self.deep_mode = deepembed            # "none" | "1x" | "4x"
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.x_k = nn.Parameter(torch.zeros(1, 1, dim))
        self.key = nn.Linear(dim, dim_ffn, bias=False)
        self.value = nn.Linear(dim_ffn, dim, bias=False)
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
            return self.value(h * self.deepemb(idx))
        out = self.value(h)
        if self.deep_mode == "1x":
            out = out * self.deepemb(idx)
        return out

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
    rosa_grad: str = "ste"         # "ste" | "exact" (counterfactual) | "none"
    rosa_tau: float = 1e-3
    rosa_o_zero: bool = True       # zero-init the branch output projection
    rosa_gate_bias: float = 2.0    # read gate starts open (~0.88)
    use_len_emb: bool = True
    len_buckets: int = 32
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
        assert self.rosa_grad in ("ste", "exact", "none")
        assert self.ffn_mode in ("ffn", "linear", "none")
        if self.ffn_mode != "ffn":
            self.deepembed = "none"   # DeepEmbed lives inside the FFN

    def summary(self) -> str:
        return (f"RWKV8Config(L={self.n_layer}, D={self.n_embd}, "
                f"V={self.vocab_size}, ffn={self.dim_ffn}, "
                f"qkv_rosa={self.use_qkv_rosa}/{self.rosa_bits}bit/"
                f"{self.rosa_grad}, token_rosa={self.use_token_rosa}, "
                f"deepembed={self.deepembed}, tie={self.tie_head})")


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
        ln = nn.LayerNorm if cfg.block_ln else nn.Identity
        self.ln_a = ln(C)
        self.ln_b = ln(C)
        self.ln_c = ln(C)
        if use_timemix:
            self.timemix = RWKV_Tmix_x070(
                C, cfg.n_layer, layer_id, cfg.head_size, cfg.scan_chunk)
        if self.use_rosa:
            self.rosa = RosaQKVBranch(
                C, nbits=cfg.rosa_bits, K=cfg.rosa_K,
                grad_mode=cfg.rosa_grad, tau=cfg.rosa_tau,
                use_async=cfg.async_rosa, use_len_emb=cfg.use_len_emb,
                len_buckets=cfg.len_buckets,
                o_init_zero=cfg.rosa_o_zero, gate_bias=cfg.rosa_gate_bias)
        if self.ffn_mode == "ffn":
            self.ffn = FFNDeepEmbed(C, cfg.dim_ffn, cfg.vocab_size,
                                    cfg.deepembed)
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
            x = x + xx + xr if self.use_rosa else x + xx
            if self.ffn_mode == "ffn":
                x = x + self.ffn(self.ln_b(x), idx)
            elif self.ffn_mode == "linear":
                x = x + self.ffn(self.ln_b(x))
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
        self.rosa_tok = (RosaTokenEmb(V, C, cfg.len_buckets)
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

    def forward(self, idx: Tensor,
                rosa_tok: Optional[Tuple[Tensor, Tensor]] = None,
                state: Optional[Dict] = None):
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

    # ---------------- RNN-mode ----------------
    def init_state(self, batch: int = 1, device=None) -> Dict:
        cfg = self.cfg
        dev = device or next(self.parameters()).device
        H = cfg.n_embd // cfg.head_size
        state: Dict = {"pos": 0, "v_first": None}
        state["S"] = [torch.zeros(batch, H, cfg.head_size, cfg.head_size,
                                  device=dev) for _ in range(cfg.n_layer)]
        state["sams"] = ([blk.rosa.init_sams() for blk in self.blocks]
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
            "sams": [blk.rosa.init_sams() for blk in self.blocks],
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
# 4.2  Trainer
# ----------------------------------------------------------------------------

_PREFETCH_POOL: Optional[ThreadPoolExecutor] = None


def _prefetch_pool() -> ThreadPoolExecutor:
    global _PREFETCH_POOL
    if _PREFETCH_POOL is None:
        _PREFETCH_POOL = ThreadPoolExecutor(max_workers=1)
    return _PREFETCH_POOL


def _warmup_cosine(step: int, total: int, warmup: int, base: float,
                   minimum: float) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return minimum + 0.5 * (base - minimum) * (1 + math.cos(math.pi * p))


@torch.no_grad()
def eval_bpc(model: nn.Module, data: ByteData, B: int = 8, T: int = 512,
             steps: int = 20, device: str = "cpu") -> float:
    model.eval()
    losses = []
    was_async = None
    for blk in getattr(model, "blocks", []):
        if getattr(blk, "use_rosa", False) and blk.rosa is not None:
            was_async = blk.rosa.use_async
            blk.rosa.use_async = False   # eval: no gradient, sync is fine
    for _ in range(steps):
        x, y = data.sample_batch(B, T, device, split="val")
        logits = model(x)
        losses.append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                      y.reshape(-1)).item())
    for blk in getattr(model, "blocks", []):
        if getattr(blk, "use_rosa", False) and blk.rosa is not None \
                and was_async is not None:
            blk.rosa.use_async = was_async
    model.train()
    return float(np.mean(losses) / math.log(2))   # nats -> bits


def train_model(model: nn.Module, data: ByteData, steps: int = 1000,
                batch_size: int = 32, ctx_len: int = 256, lr: float = 8e-4,
                warmup: int = 100, min_lr: float = 1e-5,
                weight_decay: float = 0.01, betas=(0.9, 0.95),
                clip: float = 1.0, amp: bool = False, device: str = "cuda",
                val_every: int = 200, sample_fn=None, sample_every: int = 500,
                checkpoint_path: Optional[str] = None,
                log_every: int = 20, label: str = "model",
                rosa_prefetch: bool = True, quiet: bool = False):
    """Train with AdamW + warmup-cosine.  The token-level ROSA indices for
    the NEXT batch are computed on a background CPU thread while the GPU
    trains the current batch (async pipeline [5]); the QKV-ROSA retrieval of
    each block overlaps its time-mix on the GPU."""
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas,
                            weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: _warmup_cosine(s, steps, warmup, lr, min_lr) / lr)
    scaler = torch.amp.GradScaler(enabled=amp and device == "cuda")
    uses_tok = getattr(model, "rosa_tok", None) is not None

    def sample_job():
        x, y = data.sample_batch(batch_size, ctx_len, device)
        fut = None
        if uses_tok and rosa_prefetch:
            xn = x.detach().to("cpu").numpy().astype(np.int32)
            fut = _prefetch_pool().submit(rosa_token_batch_np, xn,
                                          model.cfg.vocab_size)
        return x, y, fut

    cur = sample_job()
    t0 = time.time(); tokens_seen = 0; best_val = float("inf")
    for step in range(steps):
        nxt = sample_job()
        x, y, fut = cur
        rosa_tok = None
        if fut is not None:
            sym_np, ln_np = fut.result()
            sym = torch.from_numpy(sym_np).to(device)
            ln = torch.from_numpy(ln_np).to(device)
            rosa_tok = (sym, ln)
        y_in = y
        x_in = x
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=amp and device == "cuda"):
            logits = model(x_in, rosa_tok=rosa_tok) \
                if uses_tok else model(x_in)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   y_in.reshape(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(opt)
        scaler.update()
        sched.step()
        cur = nxt
        tokens_seen += batch_size * ctx_len
        if not quiet and (step % log_every == 0 or step == steps - 1):
            bpc = loss.item() / math.log(2)
            dt = time.time() - t0
            print(f"[{label}] step {step+1}/{steps} | loss {loss.item():.4f} "
                  f"| {bpc:.3f} bpc | lr {sched.get_last_lr()[0]:.2e} "
                  f"| {tokens_seen/dt:,.0f} tok/s")
        if val_every and (step % val_every == 0 or step == steps - 1):
            v = eval_bpc(model, data, device=device)
            best_val = min(best_val, v)
            if not quiet:
                print(f"[{label}]     val {v:.4f} bpc "
                      f"(best {best_val:.4f})")
        if sample_fn and sample_every and \
                (step % sample_every == 0 or step == steps - 1):
            if not quiet:
                print(sample_fn())
            model.train()
        if checkpoint_path and (step % 500 == 0 or step == steps - 1):
            torch.save({"model": model.state_dict(), "cfg":
                        asdict(model.cfg), "step": step},
                       checkpoint_path)
    return best_val


# ----------------------------------------------------------------------------
# 4.3  Generation
# ----------------------------------------------------------------------------

def sample_logits(logits: Tensor, temperature: float = 1.0,
                  top_p: float = 1.0, top_k: int = 0) -> int:
    """Top-k / top-p sampling (port of the RWKV helper in [1])."""
    probs = F.softmax(logits.float(), dim=-1)
    sorted_probs, sorted_ids = torch.sort(probs, descending=True)
    if top_k > 0:
        probs[sorted_ids[top_k:]] = 0
    if top_p < 1.0:
        cum = torch.cumsum(sorted_probs, dim=-1)
        cutoff = sorted_probs[torch.searchsorted(cum, top_p)]
        probs[probs < cutoff] = 0
    if temperature != 1.0:
        probs = probs ** (1.0 / temperature)
        probs = probs / probs.sum()
    return torch.multinomial(probs, num_samples=1).item()


@torch.no_grad()
def generate(model: nn.Module, prompt: List[int], n_new: int = 200,
             temperature: float = 0.8, top_p: float = 0.9, top_k: int = 0,
             device: str = "cpu") -> List[int]:
    """GPT-mode generation (recomputes the context; simple, always correct)."""
    model.eval()
    toks = [int(t) for t in prompt]
    for _ in range(n_new):
        ctx = toks[-2048:]
        logits = model(torch.tensor([ctx], device=device))
        nxt = sample_logits(logits[0, -1], temperature, top_p, top_k)
        toks.append(nxt)
    return toks


@torch.no_grad()
def generate_rnn(model: nn.Module, prompt: List[int], n_new: int = 200,
                 temperature: float = 0.8, top_p: float = 0.9, top_k: int = 0,
                 device: str = "cpu", state: Optional[Dict] = None):
    """Stateful RNN-mode generation: O(1) memory per token.  The RWKV-7g
    states, the live token suffix automaton and the per-group QKV automata
    are carried across steps — RWKV's superpower, with ROSA along for the
    ride."""
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
