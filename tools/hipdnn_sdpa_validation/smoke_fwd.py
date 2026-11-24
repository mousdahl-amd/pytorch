"""Smoke test: the hipDNN SDPA forward arms, checked against a float64 reference.

Run with this tree's PyTorch installed editable (pip install -e .), e.g. from the root:
    python tools/hipdnn_sdpa_validation/smoke_fwd.py
Each line is one case: PASS/FAIL, the relative output error and the LSE error.
Exits non-zero if any case fails.
"""

import math
import sys

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend


DEV = "cuda"
TOL = {torch.float16: 1e-2, torch.bfloat16: 3e-2, torch.float32: 2e-3}
LSE_TOL = 2e-3
failures = []


def rel_err(got, ref, valid=None):
    got = got.double()
    ref = ref.double()
    if valid is not None:
        got = got[valid]
        ref = ref[valid]
    denom = max(ref.abs().max().item(), 1e-2)
    return (got - ref).abs().max().item() / denom


def report(name, ok, detail):
    print(f"{'PASS' if ok else 'FAIL'}  {name:44s} {detail}")
    if not ok:
        failures.append(name)


def reference(q, k, v, scale, keep=None, bias=None, mult=None):
    """q (B,H,Sq,D), k/v (B,Hk,Sk,D) -> out (B,H,Sq,Dv), lse (B,H,Sq), float64."""
    q, k, v = q.double(), k.double(), v.double()
    h, hk = q.shape[1], k.shape[1]
    if hk != h:
        k = k.repeat_interleave(h // hk, dim=1)
        v = v.repeat_interleave(h // hk, dim=1)
    s = (q @ k.transpose(-1, -2)) * scale
    if bias is not None:
        s = s + bias.double()
    if keep is not None:
        s = s.masked_fill(~keep, float("-inf"))
    lse = torch.logsumexp(s, dim=-1)
    p = torch.softmax(s, dim=-1)
    if mult is not None:
        p = p * mult.double()
    return p @ v, lse


def band(sq, sk, left, right, bottom_right=True):
    """FlashAttention band: key j kept for query i in [i+d-left, i+d+right]."""
    i = torch.arange(sq, device=DEV)[:, None]
    j = torch.arange(sk, device=DEV)[None, :]
    d = sk - sq if bottom_right else 0
    keep = torch.ones(sq, sk, dtype=torch.bool, device=DEV)
    if left >= 0:
        keep &= j >= i + d - left
    if right >= 0:
        keep &= j <= i + d + right
    return keep


def rand(*shape, dtype):
    return torch.randn(*shape, device=DEV, dtype=dtype)


def flash_dense(name, b, h, hk, sq, sk, d, dtype, causal=False, wl=-1, wr=-1):
    q, k, v = rand(b, sq, h, d, dtype=dtype), rand(b, sk, hk, d, dtype=dtype), rand(b, sk, hk, d, dtype=dtype)
    out, lse, _, _, _ = torch.ops.aten._flash_attention_forward(
        q, k, v, None, None, sq, sk, 0.0, causal, False,
        window_size_left=wl if wl >= 0 else None, window_size_right=wr if wr >= 0 else None)
    keep = band(sq, sk, wl, 0 if causal else wr)
    ref, ref_lse = reference(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), 1 / math.sqrt(d), keep=keep)
    e, el = rel_err(out.transpose(1, 2), ref), (lse.double() - ref_lse).abs().max().item()
    report(name, e <= TOL[dtype] and el <= LSE_TOL, f"out {e:.2e}  lse {el:.2e}")


def flash_varlen(name, lens_q, lens_k, h, hk, d, dtype, causal=False, wl=-1, wr=-1, seqused=False):
    tq, tk = sum(lens_q), sum(lens_k) + (8 * len(lens_k) if seqused else 0)
    q, k, v = rand(tq, h, d, dtype=dtype), rand(tk, hk, d, dtype=dtype), rand(tk, hk, d, dtype=dtype)
    cu_q = torch.tensor([0] + list(torch.tensor(lens_q).cumsum(0)), device=DEV, dtype=torch.int32)
    slots_k = [n + (8 if seqused else 0) for n in lens_k]
    cu_k = torch.tensor([0] + list(torch.tensor(slots_k).cumsum(0)), device=DEV, dtype=torch.int32)
    used = torch.tensor(lens_k, device=DEV, dtype=torch.int32) if seqused else None
    out, lse, _, _, _ = torch.ops.aten._flash_attention_forward(
        q, k, v, cu_q, cu_k, max(lens_q), max(slots_k), 0.0, causal, False,
        window_size_left=wl if wl >= 0 else None, window_size_right=wr if wr >= 0 else None,
        seqused_k=used)
    e = el = 0.0
    for i, (lq, lk) in enumerate(zip(lens_q, lens_k)):
        oq, ok_ = cu_q[i].item(), cu_k[i].item()
        qs, ks, vs = q[oq:oq + lq], k[ok_:ok_ + lk], v[ok_:ok_ + lk]
        keep = band(lq, lk, wl, 0 if causal else wr)
        ref, ref_lse = reference(qs.transpose(0, 1)[None], ks.transpose(0, 1)[None], vs.transpose(0, 1)[None],
                                 1 / math.sqrt(d), keep=keep)
        e = max(e, rel_err(out[oq:oq + lq].transpose(0, 1)[None], ref))
        el = max(el, (lse[:, oq:oq + lq].double() - ref_lse[0]).abs().max().item())
    report(name, e <= TOL[dtype] and el <= LSE_TOL and tuple(lse.shape) == (h, tq),
           f"out {e:.2e}  lse {el:.2e}  lse shape {tuple(lse.shape)}")


def mem_eff(name, b, h, sq, sk, d, dv, dtype, mask_type=0, bias_shape=None):
    q, k, v = rand(b, sq, h, d, dtype=dtype), rand(b, sk, h, d, dtype=dtype), rand(b, sk, h, dv, dtype=dtype)
    bias = None
    if bias_shape is not None:
        bias = rand(*bias_shape, dtype=dtype).expand(b, h, sq, sk)
    out, lse, _, _, _, _ = torch.ops.aten._efficient_attention_forward(
        q, k, v, bias, None, None, None, None, 0.0, mask_type, True)
    keep = None
    if mask_type:
        keep = band(sq, sk, -1, 0, bottom_right=mask_type == 2)
    ref, ref_lse = reference(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), 1 / math.sqrt(d),
                             keep=keep, bias=bias)
    e, el = rel_err(out.transpose(1, 2), ref), (lse.double() - ref_lse).abs().max().item()
    report(name, e <= TOL[dtype] and el <= LSE_TOL, f"out {e:.2e}  lse {el:.2e}")


def mem_eff_varlen(name, lens, h, d, dtype):
    t = sum(lens)
    q, k, v = rand(1, t, h, d, dtype=dtype), rand(1, t, h, d, dtype=dtype), rand(1, t, h, d, dtype=dtype)
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), device=DEV, dtype=torch.int32)
    out, lse, _, _, _, _ = torch.ops.aten._efficient_attention_forward(
        q, k, v, None, cu, cu, max(lens), max(lens), 0.0, 0, True)
    e = el = 0.0
    for i, n in enumerate(lens):
        o = cu[i].item()
        ref, ref_lse = reference(*(x[0, o:o + n].transpose(0, 1)[None] for x in (q, k, v)), 1 / math.sqrt(d))
        e = max(e, rel_err(out[0, o:o + n].transpose(0, 1)[None], ref))
        el = max(el, (lse[i, :, :n].double() - ref_lse[0]).abs().max().item())
    report(name, e <= TOL[dtype] and el <= LSE_TOL and tuple(lse.shape) == (len(lens), h, max(lens)),
           f"out {e:.2e}  lse {el:.2e}  lse shape {tuple(lse.shape)}")


def dropout_check(name, out, lse, dmask, q, k, v, p):
    # ROCm debug mask contract (AOTriton's encoded softmax): +0.5 kept, -0.5 dropped.
    ok_values = bool(((dmask == 0.5) | (dmask == -0.5)).all())
    keep_frac = (dmask >= 0).double().mean().item()
    mult = (dmask >= 0).double() / (1 - p)
    ref, ref_lse = reference(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), 1 / math.sqrt(q.shape[-1]),
                             mult=mult)
    e, el = rel_err(out.transpose(1, 2), ref), (lse.double() - ref_lse).abs().max().item()
    ok = e <= TOL[q.dtype] and el <= LSE_TOL and abs(keep_frac - (1 - p)) < 0.02 and ok_values
    report(name, ok, f"out {e:.2e}  lse {el:.2e}  keep {keep_frac:.3f}")


def flash_dropout(p=0.2):
    b, h, s, d = 2, 4, 256, 64
    q, k, v = (rand(b, s, h, d, dtype=torch.float16) for _ in range(3))
    out, lse, _, _, dm = torch.ops.aten._flash_attention_forward(q, k, v, None, None, s, s, p, False, True)
    dropout_check("flash dropout p=0.2 (rng_dump)", out, lse, dm, q, k, v, p)
    out2, _, _, _, dm2 = torch.ops.aten._flash_attention_forward(q, k, v, None, None, s, s, p, False, True)
    report("flash dropout: next call draws a new mask", not torch.equal(dm, dm2), "")

    # Graph capture: the seed/offset are resolved on the device per replay.
    args = (q, k, v, None, None, s, s, p, False, True)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        torch.ops.aten._flash_attention_forward(*args)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        g_out, g_lse, _, _, g_dm = torch.ops.aten._flash_attention_forward(*args)
    masks = []
    for i in range(2):
        graph.replay()
        torch.cuda.synchronize()
        masks.append(g_dm.clone())
        dropout_check(f"flash dropout under graph replay {i}", g_out, g_lse, g_dm, q, k, v, p)
    report("graph replays draw different masks", not torch.equal(masks[0], masks[1]), "")


def gates():
    # Training-path checks are in smoke_bwd.py.
    q = rand(2, 4, 128, 64, dtype=torch.float16)
    report("_fused_sdp_choice without grad -> FLASH",
           torch._fused_sdp_choice(q, q, q) == SDPBackend.FLASH_ATTENTION.value, "")
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        out = F.scaled_dot_product_attention(q, q, q, is_causal=True)
    ref, _ = reference(q, q, q, 1 / 8, keep=band(128, 128, -1, 0))
    report("SDPA FLASH causal, no grad", rel_err(out, ref) <= TOL[torch.float16], "")
    mask = torch.rand(2, 1, 128, 128, device=DEV) > 0.3
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
        out = F.scaled_dot_product_attention(q, q, q, attn_mask=mask)
    ref, _ = reference(q, q, q, 1 / 8, keep=mask)
    report("SDPA EFFICIENT bool mask, no grad", rel_err(out, ref) <= TOL[torch.float16], "")


def main():
    print(torch.__version__, torch.version.hip, torch.cuda.get_device_properties(0).gcnArchName)
    torch.backends.cuda.preferred_rocm_fa_library("hipdnn")
    print("preferred:", torch.backends.cuda.preferred_rocm_fa_library())
    torch.manual_seed(0)

    f16, bf16, f32 = torch.float16, torch.bfloat16, torch.float32
    flash_dense("flash f16 d64", 2, 4, 4, 256, 256, 64, f16)
    flash_dense("flash bf16 d128 causal Sq<Sk (bottom-right)", 2, 4, 4, 100, 256, 128, bf16, causal=True)
    flash_dense("flash bf16 d128 GQA 8/2", 2, 8, 2, 256, 256, 128, bf16)
    flash_dense("flash bf16 d64 window (63,0)", 2, 4, 4, 256, 256, 64, bf16, wl=63, wr=0)
    flash_dense("flash bf16 d64 window (0,0)", 2, 4, 4, 256, 256, 64, bf16, wl=0, wr=0)
    flash_dense("flash bf16 d64 window (32,16)", 2, 4, 4, 256, 256, 64, bf16, wl=32, wr=16)
    flash_dense("flash bf16 d256", 1, 4, 4, 256, 256, 256, bf16)
    flash_dense("flash bf16 decode Sq=1", 4, 8, 8, 1, 512, 128, bf16, causal=True)
    flash_varlen("flash varlen bf16 d64", [37, 128, 5, 200], [37, 128, 5, 200], 4, 4, 64, bf16)
    flash_varlen("flash varlen bf16 causal GQA", [64, 17, 130], [64, 17, 130], 8, 2, 128, bf16, causal=True)
    flash_varlen("flash varlen bf16 window (16,0)", [64, 100], [64, 100], 4, 4, 64, bf16, wl=16, wr=0)
    flash_varlen("flash varlen bf16 seqused_k", [1, 1, 1], [40, 7, 90], 4, 4, 64, bf16, seqused=True)
    mem_eff("mem-eff f16 d64", 2, 4, 256, 256, 64, 64, f16)
    mem_eff("mem-eff f32 d64", 2, 4, 128, 128, 64, 64, f32)
    mem_eff("mem-eff bf16 causal top-left Sq<Sk", 2, 4, 100, 256, 64, 64, bf16, mask_type=1)
    mem_eff("mem-eff bf16 causal bottom-right Sq<Sk", 2, 4, 100, 256, 64, 64, bf16, mask_type=2)
    mem_eff("mem-eff bf16 Dv != D (128/64)", 2, 4, 128, 128, 128, 64, bf16)
    mem_eff("mem-eff bf16 bias full", 2, 4, 128, 128, 64, 64, bf16, bias_shape=(2, 4, 128, 128))
    mem_eff("mem-eff bf16 bias over heads", 2, 4, 128, 128, 64, 64, bf16, bias_shape=(1, 4, 128, 128))
    mem_eff("mem-eff bf16 bias row-broadcast", 2, 4, 128, 128, 64, 64, bf16, bias_shape=(2, 4, 1, 128))
    mem_eff_varlen("mem-eff varlen bf16 d64", [37, 128, 5], 4, 64, bf16)
    flash_dropout()
    gates()
    print(f"\n{len(failures)} failure(s)" + (": " + ", ".join(failures) if failures else ""))
    # On Windows, a process that exits with GPU work in flight hangs in HIP's
    # fat-binary unregistration (independent of hipDNN), so wait first.
    torch.cuda.synchronize()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
