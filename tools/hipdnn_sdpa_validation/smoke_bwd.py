"""Smoke test: hipDNN SDPA backward, checked against float64 autograd.

Run with this tree's PyTorch installed editable (pip install -e .), e.g. from the root:
    python tools/hipdnn_sdpa_validation/smoke_bwd.py
Each line is one case: PASS/FAIL and the worst relative gradient error.
Exits non-zero if any case fails.
"""

import math
import os
import sys

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smoke_fwd import band, DEV, failures, rand, reference, rel_err, report


# Relative to the largest reference gradient (as the P0 probe does).
GRAD_TOL = {torch.float16: 2e-2, torch.bfloat16: 6e-2, torch.float32: 1e-3}


def leaf(*shape, dtype):
    return rand(*shape, dtype=dtype).requires_grad_()


def as_ref(*ts):
    return [t.detach().double().requires_grad_() for t in ts]


def check_grads(name, got, ref, dtype, extra=""):
    errs = [rel_err(g, r) for g, r in zip(got, ref)]
    ok = all(e <= GRAD_TOL[dtype] for e in errs) and all(torch.isfinite(g).all() for g in got)
    report(name, ok, "grad " + " ".join(f"{e:.2e}" for e in errs) + extra)


def bhsd(t):  # (B, S, H, D) -> (B, H, S, D)
    return t.transpose(1, 2)


def flash_dense(name, b, h, hk, sq, sk, d, dtype, causal=False, wl=-1, wr=-1):
    q, k, v = leaf(b, sq, h, d, dtype=dtype), leaf(b, sk, hk, d, dtype=dtype), leaf(b, sk, hk, d, dtype=dtype)
    out = torch.ops.aten._flash_attention_forward(
        q, k, v, None, None, sq, sk, 0.0, causal, False,
        window_size_left=wl if wl >= 0 else None, window_size_right=wr if wr >= 0 else None)[0]
    g = torch.randn_like(out)
    out.backward(g)
    q64, k64, v64 = as_ref(q, k, v)
    ref, _ = reference(bhsd(q64), bhsd(k64), bhsd(v64), 1 / math.sqrt(d), keep=band(sq, sk, wl, 0 if causal else wr))
    ref.backward(bhsd(g.double()))
    check_grads(name, (q.grad, k.grad, v.grad), (q64.grad, k64.grad, v64.grad), dtype)


def flash_dropout(p=0.2):
    b, h, s, d, dtype = 2, 4, 256, 64, torch.float16
    q, k, v = (leaf(b, s, h, d, dtype=dtype) for _ in range(3))
    out, _, _, _, dm = torch.ops.aten._flash_attention_forward(q, k, v, None, None, s, s, p, False, True)
    g = torch.randn_like(out)
    out.backward(g)
    mult = (dm >= 0).double() / (1 - p)  # ROCm debug mask: +0.5 kept, -0.5 dropped
    q64, k64, v64 = as_ref(q, k, v)
    ref, _ = reference(bhsd(q64), bhsd(k64), bhsd(v64), 1 / math.sqrt(d), mult=mult)
    ref.backward(bhsd(g.double()))
    # The backward recomputes the mask from the forward's seed and offset; the
    # gradients only match the reference if it draws the same mask.
    check_grads("flash dropout p=0.2: fwd mask == bwd mask", (q.grad, k.grad, v.grad),
                (q64.grad, k64.grad, v64.grad), dtype)


def flash_varlen(name, lens, h, hk, d, dtype, causal=False, wl=-1, wr=-1):
    t = sum(lens)
    q, k, v = leaf(t, h, d, dtype=dtype), leaf(t, hk, d, dtype=dtype), leaf(t, hk, d, dtype=dtype)
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), device=DEV, dtype=torch.int32)
    out = torch.ops.aten._flash_attention_forward(
        q, k, v, cu, cu, max(lens), max(lens), 0.0, causal, False,
        window_size_left=wl if wl >= 0 else None, window_size_right=wr if wr >= 0 else None)[0]
    g = torch.randn_like(out)
    out.backward(g)
    q64, k64, v64 = as_ref(q, k, v)
    for i, n in enumerate(lens):
        o = cu[i].item()
        seq = [x[o:o + n].transpose(0, 1)[None] for x in (q64, k64, v64)]
        ref, _ = reference(*seq, 1 / math.sqrt(d), keep=band(n, n, wl, 0 if causal else wr))
        ref.backward(g[o:o + n].transpose(0, 1)[None].double())
    check_grads(name, (q.grad, k.grad, v.grad), (q64.grad, k64.grad, v64.grad), dtype)


def mem_eff(name, b, h, sq, sk, d, dv, dtype, mask_type=0, bias_shape=None, bias_grad=False):
    q, k, v = leaf(b, sq, h, d, dtype=dtype), leaf(b, sk, h, d, dtype=dtype), leaf(b, sk, h, dv, dtype=dtype)
    base = rand(*bias_shape, dtype=dtype).requires_grad_(bias_grad) if bias_shape else None
    bias = base.expand(b, h, sq, sk) if base is not None else None
    out = torch.ops.aten._efficient_attention_forward(q, k, v, bias, None, None, None, None, 0.0, mask_type, True)[0]
    g = torch.randn_like(out)
    out.backward(g)
    q64, k64, v64 = as_ref(q, k, v)
    base64 = as_ref(base)[0] if base is not None else None
    if base64 is not None and not bias_grad:
        base64.requires_grad_(False)
    keep = band(sq, sk, -1, 0, bottom_right=mask_type == 2) if mask_type else None
    ref, _ = reference(bhsd(q64), bhsd(k64), bhsd(v64), 1 / math.sqrt(d), keep=keep,
                       bias=base64.expand(b, h, sq, sk) if base64 is not None else None)
    ref.backward(bhsd(g.double()))
    got, want = [q.grad, k.grad, v.grad], [q64.grad, k64.grad, v64.grad]
    if bias_grad:
        got.append(base.grad)
        want.append(base64.grad)
    check_grads(name, got, want, dtype)


def mem_eff_varlen(name, lens, h, d, dtype):
    t = sum(lens)
    q, k, v = (leaf(1, t, h, d, dtype=dtype) for _ in range(3))
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), device=DEV, dtype=torch.int32)
    out = torch.ops.aten._efficient_attention_forward(q, k, v, None, cu, cu, max(lens), max(lens), 0.0, 0, True)[0]
    g = torch.randn_like(out)
    out.backward(g)
    q64, k64, v64 = as_ref(q, k, v)
    for i, n in enumerate(lens):
        o = cu[i].item()
        seq = [x[0, o:o + n].transpose(0, 1)[None] for x in (q64, k64, v64)]
        ref, _ = reference(*seq, 1 / math.sqrt(d))
        ref.backward(g[0, o:o + n].transpose(0, 1)[None].double())
    check_grads(name, (q.grad, k.grad, v.grad), (q64.grad, k64.grad, v64.grad), dtype)


def sdpa_training():
    """SDPA end to end: selection, forward and backward on hipDNN."""
    q, k, v = (leaf(2, 4, 128, 64, dtype=torch.bfloat16) for _ in range(3))
    report("_fused_sdp_choice with grad -> FLASH",
           torch._fused_sdp_choice(q, k, v) == SDPBackend.FLASH_ATTENTION.value, "")
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    g = torch.randn_like(out)
    out.backward(g)
    q64, k64, v64 = as_ref(q, k, v)
    ref, _ = reference(q64, k64, v64, 1 / 8, keep=band(128, 128, -1, 0))
    ref.backward(g.double())
    check_grads("SDPA FLASH causal training", (q.grad, k.grad, v.grad), (q64.grad, k64.grad, v64.grad),
                torch.bfloat16)

    q, k, v = (leaf(2, 4, 128, 64, dtype=torch.float32) for _ in range(3))
    mask = rand(2, 4, 128, 128, dtype=torch.float32).requires_grad_()
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    g = torch.randn_like(out)
    out.backward(g)
    q64, k64, v64, m64 = as_ref(q, k, v, mask)
    ref, _ = reference(q64, k64, v64, 1 / 8, bias=m64)
    ref.backward(g.double())
    check_grads("SDPA EFFICIENT fp32 float mask training", (q.grad, k.grad, v.grad, mask.grad),
                (q64.grad, k64.grad, v64.grad, m64.grad), torch.float32)


def determinism():
    q, k, v = (leaf(2, 256, 4, 64, dtype=torch.bfloat16) for _ in range(3))
    grads = []
    for _ in range(2):
        for t in (q, k, v):
            t.grad = None
        out = torch.ops.aten._flash_attention_forward(q, k, v, None, None, 256, 256, 0.0, True, False)[0]
        out.backward(torch.ones_like(out))
        grads.append([t.grad.clone() for t in (q, k, v)])
    report("backward is bitwise repeatable", all(torch.equal(a, b) for a, b in zip(*grads)), "")

    from torch.backends.cuda import SDPAParams, can_use_flash_attention, can_use_efficient_attention
    qg = leaf(2, 4, 128, 64, dtype=torch.float16)
    params = SDPAParams(qg, qg, qg, None, 0.0, False, False)
    torch.use_deterministic_algorithms(True)
    try:
        report("deterministic mode: flash declines training",
               not can_use_flash_attention(params, False), "")
        report("deterministic mode: mem-eff declines training",
               not can_use_efficient_attention(params, False), "")
    finally:
        torch.use_deterministic_algorithms(False)
    report("default mode: flash takes training", can_use_flash_attention(params, False), "")


def main():
    print(torch.__version__, torch.version.hip, torch.cuda.get_device_properties(0).gcnArchName)
    torch.backends.cuda.preferred_rocm_fa_library("hipdnn")
    print("preferred:", torch.backends.cuda.preferred_rocm_fa_library())
    torch.manual_seed(0)

    f16, bf16, f32 = torch.float16, torch.bfloat16, torch.float32
    flash_dense("flash bwd f16 d64", 2, 4, 4, 256, 256, 64, f16)
    flash_dense("flash bwd bf16 d128 causal Sq<Sk", 2, 4, 4, 100, 256, 128, bf16, causal=True)
    flash_dense("flash bwd bf16 d128 GQA 8/2", 2, 8, 2, 256, 256, 128, bf16)
    flash_dense("flash bwd bf16 d64 window (63,0)", 2, 4, 4, 256, 256, 64, bf16, wl=63, wr=0)
    flash_dense("flash bwd bf16 d64 window (32,16)", 2, 4, 4, 256, 256, 64, bf16, wl=32, wr=16)
    flash_dense("flash bwd bf16 d256", 1, 4, 4, 256, 256, 256, bf16)
    flash_dropout()
    flash_varlen("flash varlen bwd bf16 d64", [37, 128, 5, 200], 4, 4, 64, bf16)
    flash_varlen("flash varlen bwd bf16 causal GQA", [64, 17, 130], 8, 2, 128, bf16, causal=True)
    mem_eff("mem-eff bwd f16 d64", 2, 4, 256, 256, 64, 64, f16)
    mem_eff("mem-eff bwd f32 d64", 2, 4, 128, 128, 64, 64, f32)
    mem_eff("mem-eff bwd bf16 causal top-left Sq<Sk", 2, 4, 100, 256, 64, 64, bf16, mask_type=1)
    mem_eff("mem-eff bwd bf16 Dv != D (128/64)", 2, 4, 128, 128, 128, 64, bf16)
    mem_eff("mem-eff bwd bf16 bias, no grad", 2, 4, 128, 128, 64, 64, bf16, bias_shape=(2, 4, 128, 128))
    mem_eff("mem-eff bwd bf16 dbias full", 2, 4, 128, 128, 64, 64, bf16, bias_shape=(2, 4, 128, 128), bias_grad=True)
    mem_eff("mem-eff bwd bf16 dbias over heads", 2, 4, 128, 128, 64, 64, bf16, bias_shape=(1, 4, 128, 128), bias_grad=True)
    mem_eff("mem-eff bwd bf16 dbias row-broadcast", 2, 4, 128, 128, 64, 64, bf16, bias_shape=(2, 4, 1, 128), bias_grad=True)
    mem_eff_varlen("mem-eff varlen bwd bf16 d64", [37, 128, 5], 4, 64, bf16)
    sdpa_training()
    determinism()
    print(f"\n{len(failures)} failure(s)" + (": " + ", ".join(failures) if failures else ""))
    # See smoke_fwd.py: wait for GPU work before exiting.
    torch.cuda.synchronize()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
