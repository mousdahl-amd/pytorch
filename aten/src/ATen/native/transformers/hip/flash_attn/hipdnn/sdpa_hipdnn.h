#pragma once

// SDPA on ROCm through hipDNN's native frontend: the flash-attention entry
// points (same contracts as their AOTriton counterparts in flash_api.h) and
// the memory-efficient attention forward and backward.

#include <ATen/HipDNNConfig.h>

#if AT_HIPDNN_ENABLED()

#include <ATen/core/Generator.h>
#include <ATen/core/Tensor.h>

#include <optional>
#include <tuple>

namespace pytorch_flash {

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
mha_fwd_hipdnn(
    const at::Tensor& q, // batch_size x seqlen_q x num_heads x head_size
    const at::Tensor& k, // batch_size x seqlen_k x num_heads_k x head_size
    const at::Tensor& v, // batch_size x seqlen_k x num_heads_k x head_size
    std::optional<at::Tensor>& out_, // batch_size x seqlen_q x num_heads x head_size
    std::optional<at::Tensor>& alibi_slopes_,
    const float p_dropout,
    const float softmax_scale,
    bool is_causal,
    std::optional<int64_t> window_size_left,
    std::optional<int64_t> window_size_right,
    const bool return_softmax,
    const std::optional<at::Generator>& gen_);

// The logsumexp is returned as (num_heads, total_q).
std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
mha_varlen_fwd_hipdnn(
    const at::Tensor& q, // total_q x num_heads x head_size
    const at::Tensor& k, // total_k x num_heads_k x head_size
    const at::Tensor& v, // total_k x num_heads_k x head_size
    std::optional<at::Tensor>& out_, // total_q x num_heads x head_size
    const at::Tensor& cu_seqlens_q, // b+1
    const at::Tensor& cu_seqlens_k, // b+1
    std::optional<at::Tensor>& seqused_k, // b
    std::optional<at::Tensor>& block_table_,
    std::optional<at::Tensor>& alibi_slopes_,
    int max_seqlen_q,
    const int max_seqlen_k,
    const float p_dropout,
    const float softmax_scale,
    const bool zero_tensors,
    bool is_causal,
    std::optional<int64_t> window_size_left,
    std::optional<int64_t> window_size_right,
    const bool return_softmax,
    const std::optional<at::Generator>& gen_);

// Backward of mha_fwd_hipdnn: returns (dq, dk, dv, undefined).
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> mha_bwd_hipdnn(
    const at::Tensor& dout, // batch_size x seqlen_q x num_heads x head_size
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& out,
    const at::Tensor& softmax_lse, // b x h x seqlen_q
    std::optional<at::Tensor>& dq_,
    std::optional<at::Tensor>& dk_,
    std::optional<at::Tensor>& dv_,
    std::optional<at::Tensor>& alibi_slopes_,
    const float p_dropout,
    const float softmax_scale,
    bool is_causal,
    std::optional<int64_t> window_size_left,
    std::optional<int64_t> window_size_right,
    const bool deterministic,
    const at::Tensor& philox_seed,
    const at::Tensor& philox_offset);

// Backward of mha_varlen_fwd_hipdnn; softmax_lse is (num_heads, total_q).
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> mha_varlen_bwd_hipdnn(
    const at::Tensor& dout, // total_q x num_heads x head_size
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& out,
    const at::Tensor& softmax_lse,
    std::optional<at::Tensor>& dq_,
    std::optional<at::Tensor>& dk_,
    std::optional<at::Tensor>& dv_,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    std::optional<at::Tensor>& alibi_slopes_,
    const int max_seqlen_q,
    const int max_seqlen_k,
    const float p_dropout,
    const float softmax_scale,
    const bool zero_tensors,
    bool is_causal,
    std::optional<int64_t> window_size_left,
    std::optional<int64_t> window_size_right,
    const bool deterministic,
    const at::Tensor& philox_seed,
    const at::Tensor& philox_offset);

// Memory-efficient attention forward. Writes `out` (B, M, H, Kv), or
// (1, total_q, H, Kv) with seqstart_q/seqstart_k, and returns the logsumexp
// (lse_batch, H, compute_logsumexp ? max_seqlen_q : 0). With dropout,
// philox_seed and philox_offset must already hold the seed and offset on the
// device.
at::Tensor me_fwd_hipdnn(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& bias,
    const std::optional<at::Tensor>& seqstart_q,
    const std::optional<at::Tensor>& seqstart_k,
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    double dropout_p,
    int64_t custom_mask_type,
    bool compute_logsumexp,
    float softmax_scale,
    const at::Tensor& philox_seed,
    const at::Tensor& philox_offset,
    at::Tensor& out);

// Memory-efficient attention backward. Writes grad_q, grad_k and grad_v and
// returns the bias gradient (full shape, packed), or an undefined tensor
// without bias_requires_grad.
at::Tensor me_bwd_hipdnn(
    const at::Tensor& grad_out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& bias,
    const at::Tensor& out,
    const std::optional<at::Tensor>& cu_seqlens_q,
    const std::optional<at::Tensor>& cu_seqlens_k,
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    const at::Tensor& logsumexp,
    double dropout_p,
    const at::Tensor& philox_seed,
    const at::Tensor& philox_offset,
    int64_t custom_mask_type,
    bool bias_requires_grad,
    float softmax_scale,
    at::Tensor& grad_q,
    at::Tensor& grad_k,
    at::Tensor& grad_v);

} // namespace pytorch_flash

#endif // AT_HIPDNN_ENABLED()
