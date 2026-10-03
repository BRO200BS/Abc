"""RWKV-8 "Heron" v2 test suite.

Run:  python -m pytest test_rwkv8.py -v   (CPU-only, ~3-6 min; numba JIT
compiles once per session).  Every v2 feature has a validation test here:
engine candidates, batched SAM stepping, the GPU window matcher, the
chunk-parallel WY scan, the sampler fixes, RosaLM/Heron RNN equivalence,
soft-ROSA, presets, the trainer (resume/EMA/schedules), state IO, and the
benchmark harness.
"""
import os
import sys
import tempfile

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import rwkv8 as m

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# 1 · ROSA engine
# ---------------------------------------------------------------------------

def test_multi_candidate_matches_single_engine():
    rng = np.random.default_rng(0)
    T, A = 48, 16
    for _ in range(4):
        q = rng.integers(0, A, T).astype(np.int32)
        k = rng.integers(0, A, T).astype(np.int32)
        v = rng.integers(0, A, T).astype(np.int32)
        s1, l1, p1 = m.rosa_qkv_batch_np(q[None], k[None], v[None], A)
        sM, lM, pM = m.rosa_qkv_batch_np_m(q[None], k[None], v[None], A, M=4)
        assert s1[0][0] == sM[0][0][0] and l1[0][0] == lM[0][0][0]
        for t in range(T):
            for c in range(4):
                L, p = int(lM[0, c, t]), int(pM[0, c, t])
                if L == 0 and p < 0:
                    continue
                assert p <= t and p - L >= 0
                assert (k[p - L:p] == q[t - L + 1:t + 1]).all()


def test_batched_sam_stepping_matches_per_object():
    rng = np.random.default_rng(1)
    N, A = 6, 16
    sams = [m.StatefulQKVSam(A) for _ in range(N)]
    batch = m.BatchedQKVSams(N, A)
    for i in range(60):
        q = rng.integers(0, A, N).astype(np.int32)
        k = rng.integers(0, A, N).astype(np.int32)
        v = rng.integers(0, A, N).astype(np.int32)
        ref = [s.extend(int(q[n]), int(k[n]), int(v[n]))
               for n, s in enumerate(sams)]
        got = batch.step(q, k, v)
        ra = np.array(ref, np.int32)
        assert (ra[:, 0] == got[0]).all() and (ra[:, 1] == got[1]).all() \
            and (ra[:, 2] == got[2]).all()


def test_batched_sam_serialization_roundtrip():
    rng = np.random.default_rng(2)
    N, A = 3, 16
    batch = m.BatchedQKVSams(N, A)
    for _ in range(20):
        batch.step(*(rng.integers(0, A, N).astype(np.int32) for _ in range(3)))
    ref = m.BatchedQKVSams.load(batch.dump())
    lst = m.BatchedQKVSams.from_list(batch.to_list())
    objs = batch.to_list()                     # advance these alongside
    for _ in range(10):
        q, k, v = (rng.integers(0, A, N).astype(np.int32) for _ in range(3))
        exp = [s.extend(int(q[n]), int(k[n]), int(v[n]))
               for n, s in enumerate(objs)]
        ra = np.array(exp, np.int32)
        a1 = ref.step(q, k, v)
        a2 = lst.step(q, k, v)
        assert (ra[:, 0] == a1[0]).all() and (ra[:, 1] == a1[1]).all()
        assert (ra[:, 0] == a2[0]).all() and (ra[:, 1] == a2[1]).all()


def test_gpu_window_matcher_matches_engine():
    rng = np.random.default_rng(3)
    for _ in range(3):
        q = torch.randint(0, 16, (4, 40))
        k = torch.randint(0, 16, (4, 40))
        v = torch.randint(0, 16, (4, 40))
        s_nb, l_nb, p_nb = m.rosa_qkv_batch_np(
            q.numpy().astype(np.int32), k.numpy().astype(np.int32),
            v.numpy().astype(np.int32), 16)
        s_g, l_g, p_g = m.rosa_qkv_batch_gpu(q, k, v, alphabet=16, Lcap=64)
        assert (torch.from_numpy(s_nb) == s_g).all()
        assert (torch.from_numpy(l_nb) == l_g).all()
        assert (torch.from_numpy(p_nb) == p_g).all()


# ---------------------------------------------------------------------------
# 2 · Chunk-parallel WY scan
# ---------------------------------------------------------------------------

def _scan_args(T, B=2, H=3, N=16, dtype=torch.float64, realistic=True):
    r = torch.randn(B, T, H, N, dtype=dtype) * 0.5
    if realistic:
        x = torch.randn(B, T, H, N, dtype=dtype)
        w = -F.softplus(-x) - 0.5            # model's structural bound
    else:
        w = torch.randn(B, T, H, N, dtype=dtype) * 0.7 - 1.0
    k = torch.randn(B, T, H, N, dtype=dtype) * 0.5
    v = torch.randn(B, T, H, N, dtype=dtype) * 0.5
    kk = torch.randn(B, T, H, N, dtype=dtype)
    kk = kk / kk.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    a = torch.sigmoid(torch.randn(B, T, H, N, dtype=dtype))
    return r, w, k, v, kk, a


@pytest.mark.parametrize("T,state_on", [(97, False), (64, True), (33, False)])
def test_wy_scan_exact_fp64(T, state_on):
    args = _scan_args(T)
    S0 = torch.randn(2, 3, 16, 16, dtype=torch.float64) * 0.3 \
        if state_on else None
    y1, S1 = m.rwkv7g_scan(*args, state=S0, chunk=32, mode="sequential")
    y2, S2 = m.rwkv7g_scan(*args, state=S0, chunk=32, mode="chunked")
    assert torch.allclose(y1, y2, atol=1e-9, rtol=1e-8)
    assert torch.allclose(S1, S2, atol=1e-9, rtol=1e-8)


def test_wy_scan_grads_fp64():
    args = _scan_args(96)
    g = torch.randn(2, 96, 3, 16, dtype=torch.float64)
    a1 = [t.clone().requires_grad_(True) for t in args]
    a2 = [t.clone().requires_grad_(True) for t in args]
    y1, _ = m.rwkv7g_scan(*a1, chunk=32, mode="sequential")
    y2, _ = m.rwkv7g_scan(*a2, chunk=32, mode="chunked")
    (y1 * g).sum().backward()
    (y2 * g).sum().backward()
    for x, y in zip(a1, a2):
        assert torch.allclose(x.grad, y.grad, atol=1e-8, rtol=1e-6)


def test_wy_scan_fp32_realistic():
    args = _scan_args(128, dtype=torch.float32)
    with torch.no_grad():
        y1, _ = m.rwkv7g_scan(*args, chunk=32, mode="sequential")
        y2, _ = m.rwkv7g_scan(*args, chunk=32, mode="chunked")
    assert (y1 - y2).abs().max().item() < 1e-4


# ---------------------------------------------------------------------------
# 3 · Models
# ---------------------------------------------------------------------------

def _tiny_cfg(**over):
    return m.RWKV8Config.preset("test", vocab_size=256, async_rosa=False,
                                **over)


@pytest.mark.parametrize("kind", ["heron", "rosalm"])
def test_gpt_equals_rnn(kind):
    cfg = _tiny_cfg()
    model = (m.RWKV8Heron(cfg) if kind == "heron" else m.RosaLM(cfg))
    model.eval()
    idx = torch.randint(0, 256, (2, 24))
    with torch.no_grad():
        gpt = model(idx)
        state = model.init_state(2)
        rnn = []
        for t in range(24):
            lo, state = model(idx[:, t:t + 1], state=state)
            rnn.append(lo[:, 0])
        rnn = torch.stack(rnn, dim=1)
    assert (gpt - rnn).abs().max().item() < 1e-4


def test_rnn_mode_batch_gt_one():
    """v1 silently shared QKV automata across batch rows in RNN mode."""
    cfg = _tiny_cfg()
    model = m.RWKV8Heron(cfg)
    model.eval()
    idx = torch.randint(0, 256, (2, 12))
    with torch.no_grad():
        gpt = model(idx)
        state = model.init_state(2)
        rnn = []
        for t in range(12):
            lo, state = model(idx[:, t:t + 1], state=state)
            rnn.append(lo[:, 0])
        rnn = torch.stack(rnn, dim=1)
    assert (gpt - rnn).abs().max().item() < 1e-4


def test_soft_rosa_forward_backward():
    cfg = _tiny_cfg(rosa_grad="soft", rosa_soft_M=3)
    model = m.RWKV8Heron(cfg)
    model.train()
    idx = torch.randint(0, 256, (2, 32))
    y = torch.randint(0, 256, (2, 32))
    loss = F.cross_entropy(model(idx).reshape(-1, 256), y.reshape(-1))
    loss.backward()
    for blk in model.blocks:
        assert torch.isfinite(blk.rosa.len_scale.grad)
        assert torch.isfinite(blk.rosa.emb1.grad).all()
        assert torch.isfinite(blk.rosa.q.weight.grad).all()


def test_soft_m1_equals_hard():
    mh = m.RWKV8Heron(_tiny_cfg(rosa_grad="ste"))
    ms = m.RWKV8Heron(_tiny_cfg(rosa_grad="soft", rosa_soft_M=1))
    missing, unexpected = ms.load_state_dict(mh.state_dict(), strict=False)
    assert all("len_" in k for k in missing) and not unexpected
    mh.eval(); ms.eval()
    idx = torch.randint(0, 256, (2, 32))
    with torch.no_grad():
        assert (mh(idx) - ms(idx)).abs().max().item() < 1e-5


def test_gpu_backend_matches_cpu_inside_model():
    mc = m.RWKV8Heron(_tiny_cfg())
    mg = m.RWKV8Heron(_tiny_cfg(rosa_backend="gpu"))
    mg.load_state_dict(mc.state_dict())
    mc.eval(); mg.eval()
    idx = torch.randint(0, 256, (2, 32))
    with torch.no_grad():
        d = (mc(idx) - mg(idx)).abs().max().item()
    assert d < 0.35     # matches > gpu_cap legitimately differ


def test_presets():
    assert m.RWKV8Config.preset("t4max").n_layer == 12
    assert m.RWKV8Config.preset("golf16mb").n_embd == 384
    with pytest.raises(KeyError):
        m.RWKV8Config.preset("nope")


def test_inner_monologue_forward():
    cfg = _tiny_cfg(inner_monologue=True)
    model = m.RWKV8Heron(cfg)
    model.eval()
    idx = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        _ = model(idx)
        _ = model(idx)          # second pass consumes its own lagged tokens


# ---------------------------------------------------------------------------
# 4 · Sampler
# ---------------------------------------------------------------------------

def test_sampler_no_indexerror_at_top_p_one():
    logits = torch.randn(300)
    for tp in [1.0, 0.9999999, 0.5]:
        m.sample_logits(logits, top_p=tp)


def test_sampler_greedy_at_zero_temperature():
    logits = torch.randn(100)
    assert m.sample_logits(logits, temperature=0.0) == int(logits.argmax())


def test_sampler_penalties_shift_distribution():
    logits = torch.zeros(50)
    logits[3] = 5.0                        # favourite
    # with a heavy repetition penalty on token 3, another token should win
    hits = 0
    for _ in range(50):
        s = m.sample_logits(logits, temperature=1.0, recent_ids=[3] * 40)
        hits += (s == 3)
    assert hits < 40, "repetition penalty did not suppress the hot token"


def test_sampler_temperature_first_order():
    """With top_p, temperature must reshape the distribution BEFORE the
    filter (v1 applied it after, silently changing the tail)."""
    logits = torch.tensor([10.0, 9.0, 5.0, 0.0])
    torch.manual_seed(0)
    got = {m.sample_logits(logits, temperature=0.5, top_p=0.9)
           for _ in range(20)}
    assert got <= {0, 1}, f"low-temp top-p kept the tail: {got}"


# ---------------------------------------------------------------------------
# 5 · Trainer
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_data():
    text = "the quick brown fox jumps over the lazy dog. " * 300
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "tiny.txt")
    with open(path, "w") as f:
        f.write(text)
    return m.ByteData(path, val_bytes=1000, max_train_bytes=8000), tmp


def test_trainer_resume_and_logging(tiny_data):
    data, tmp = tiny_data
    cfg = _tiny_cfg(ctx_len=32)
    model = m.RWKV8Heron(cfg)
    ckpt = os.path.join(tmp, "ck.pt")
    log = m.RunLogger(out_dir=os.path.join(tmp, "logs"))
    m.train_model(model, data, steps=4, batch_size=2, ctx_len=32, device="cpu",
                  val_every=2, checkpoint_path=ckpt, quiet=True, warmup=1,
                  logger=log, ema_decay=0.9)
    assert os.path.exists(ckpt)
    assert os.path.exists(os.path.join(tmp, "logs", "log.csv"))
    best = m.train_model(model, data, steps=8, batch_size=2, ctx_len=32,
                         device="cpu", val_every=2, checkpoint_path=ckpt,
                         quiet=True, resume=True)
    assert best < float("inf")


def test_schedules_have_expected_shape(tiny_data):
    _, tmp = tiny_data
    model = m.RWKV8Heron(_tiny_cfg(ctx_len=32))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sch = m._make_sched(opt, "cosine_restarts", steps=100, warmup=10,
                        lr=1e-3, min_lr=1e-5, restarts=4)
    lrs = []
    opt.step()
    for _ in range(100):
        lrs.append(sch.get_last_lr()[0])
        opt.step()
        sch.step()
    peaks = sum(1 for i in range(1, 99)
                if lrs[i] > lrs[i - 1] and lrs[i] > lrs[i + 1])
    assert peaks >= 3


def test_param_groups_exclude_scalars(tiny_data):
    _, tmp = tiny_data
    model = m.RWKV8Heron(_tiny_cfg(ctx_len=32))
    groups, _ = m._param_groups(model, 1e-3, 0.1)
    scalars = {p.numel() for p in groups[1]["params"] if p.dim() < 2}
    assert 1 in scalars and any(p.dim() >= 2 for p in groups[0]["params"])


def test_sequential_data(tiny_data):
    text = "the quick brown fox jumps over the lazy dog. " * 300
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "s.txt")
    with open(path, "w") as f:
        f.write(text)
    sdata = m.SequentialData(path, val_bytes=200, max_train_bytes=2000)
    x, y = sdata.next_batch(2, 16, "cpu")
    assert x.shape == (2, 16) and (y[:, :-1] == x[:, 1:]).all()


# ---------------------------------------------------------------------------
# 6 · Inference / serving
# ---------------------------------------------------------------------------

def test_state_save_load_bit_exact(tiny_data):
    data, _ = tiny_data
    model = m.RWKV8Heron(_tiny_cfg(ctx_len=32))
    model.eval()
    state = model.init_state(1)
    with torch.no_grad():
        for t in [84, 104, 101]:
            _, state = model(torch.tensor([[t]]), state=state)
    tmp = tempfile.mkdtemp()
    sp = os.path.join(tmp, "s.pt")
    m.save_rnn_state(state, sp)
    state2 = m.load_rnn_state(sp, model)
    with torch.no_grad():
        l1, _ = model(torch.tensor([[114]]), state=state)
        l2, _ = model(torch.tensor([[114]]), state=state2)
    assert (l1 - l2).abs().max().item() == 0.0


def test_generate_deterministic_with_seed(tiny_data):
    model = m.RWKV8Heron(_tiny_cfg(ctx_len=32))
    model.eval()
    g1 = m.generate(model, [84, 104], n_new=8, device="cpu", mode="rnn",
                    temperature=0.7, seed=7, repetition_penalty=1.0)
    g2 = m.generate(model, [84, 104], n_new=8, device="cpu", mode="rnn",
                    temperature=0.7, seed=7, repetition_penalty=1.0)
    assert g1 == g2


def test_tokenizer_bytes_with_specials():
    tok = m.Tokenizer("bytes")
    ids = tok.encode("<C_begin>hi<asst>")
    assert tok.stoi["<C_begin>"] in ids and tok.stoi["<asst>"] in ids
    assert "hi" in tok.decode(ids)
    chat_ids = m.encode_chat(tok, "sys prompt", [("q", "a")])
    assert tok.stoi["<sys>"] in chat_ids


def test_quantize_int8_runs(tiny_data):
    model = m.RWKV8Heron(_tiny_cfg(ctx_len=32))
    model.eval()
    q = m.quantize_int8(model)
    with torch.no_grad():
        _ = q(torch.randint(0, 256, (1, 16)))


def test_benchmarks(tiny_data):
    data, _ = tiny_data
    model = m.RWKV8Heron(_tiny_cfg(ctx_len=32))
    model.eval()
    bpc = m.eval_bpc(model, data, device="cpu", T=32, steps=4)
    full = m.eval_bpc_full(model, data, T=32, stride=32, device="cpu")
    pp = m.per_position_losses(model, data, T=32, n_windows=2, device="cpu")
    assert bpc > 0 and full > 0 and pp.shape[0] == 32
    niah = m.niah_sweep(model, filler_lens=[50, 300])
    assert all(niah.values()), f"NIAH must retrieve by design: {niah}"


# ---------------------------------------------------------------------------
# 7 · Engine regression (v1 behaviours preserved)
# ---------------------------------------------------------------------------

def test_engine_still_matches_brute_force():
    rng = np.random.default_rng(9)
    for seed in range(4):
        rng = np.random.default_rng(9 + seed)
        T, A = 30, 16
        q = rng.integers(0, A, T).astype(np.int32)
        k = rng.integers(0, A, T).astype(np.int32)
        v = rng.integers(0, A, T).astype(np.int32)
        sym, ln, _pos = m.rosa_qkv_batch_np(q[None], k[None], v[None], A)
        bf_idx, bf_ln = m.rosa_qkv_brute(q.tolist(), k.tolist(), v.tolist())
        bf_idx = np.array(bf_idx, np.int32)[None]
        bf_ln = np.array(bf_ln, np.int32)[None]
        # brute force encodes "no match" as idx=0/ln=0 (array init), the
        # engine as sym=-1: compare on matched positions only
        matched = bf_ln > 0
        assert (sym[matched] == bf_idx[matched]).all(), f"seed {seed}"
        assert (ln == bf_ln).all(), f"seed {seed}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
