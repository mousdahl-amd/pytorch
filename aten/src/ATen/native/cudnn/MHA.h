#pragma once
#include <ATen/core/Tensor.h>

namespace at::native {

// Check the pointer and stride alignment cuDNN requires for varlen tensors.
bool has_aligned_varlen_layout(const Tensor& tensor);

// The cuDNN SDPA graph takes a 4D bias. Broadcastable 2D/3D masks are expanded
// (no copy) to [b, 1 or h, s_q, s_kv].
inline std::optional<Tensor> expand_attn_bias_for_cudnn(
    const std::optional<Tensor>& attn_bias,
    int64_t b,
    int64_t s_q,
    int64_t s_kv) {
  if (!attn_bias.has_value()) {
    return std::nullopt;
  }
  const Tensor& bias = attn_bias.value();
  TORCH_CHECK(
      bias.dim() >= 2 && bias.dim() <= 4,
      "cuDNN SDPA expects either a 2D, 3D, or 4D attn_bias but got ",
      bias.dim(),
      "D");
  const int64_t h = bias.dim() == 4 ? bias.size(1) : 1;
  return bias.expand({b, h, s_q, s_kv});
}

// Whether hipDNN can serve this attention configuration (ROCm only; defined in
// MHA.cpp when AT_HIPDNN_ENABLED()).
bool cudnn_sdpa_is_supported(
    int64_t b,
    int64_t h,
    int64_t s_q,
    int64_t s_kv,
    int64_t d_qk,
    int64_t d_v,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const std::optional<Tensor>& attn_mask,
    double dropout_probability,
    bool is_causal,
    bool requires_grad);

void run_cudnn_SDP_fprop(
    int64_t b,
    int64_t h,
    int64_t s_q,
    int64_t s_kv,
    int64_t d_k,
    int64_t d_v,
    float scaling_factor,
    bool isTraining,
    bool is_causal,
    double dropout_probability,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const std::optional<Tensor>& attn_bias,
    Tensor& softmaxstats,
    Tensor& o,
    Tensor& dropoutseed,
    Tensor& dropoutoffset);

void run_cudnn_SDP_fprop_nestedtensor(
    int64_t b,
    int64_t h_q,
    int64_t h_k,
    int64_t h_v,
    int64_t max_s_q,
    int64_t max_s_kv,
    int64_t d_k,
    int64_t d_v,
    float scaling_factor,
    bool isTraining,
    bool is_causal,
    double dropout_probability,
    const Tensor& cum_seqlen_q,
    const Tensor& cum_seqlen_kv,
    const std::optional<Tensor>& seqused_k,
    const std::optional<Tensor>& page_table,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const std::optional<Tensor>& attn_bias,
    Tensor& softmaxstats,
    Tensor& o,
    Tensor& dropoutseed,
    Tensor& dropoutoffset);

void run_cudnn_SDP_bprop(
    int64_t b,
    int64_t h,
    int64_t s_q,
    int64_t s_kv,
    int64_t d_k,
    int64_t d_v,
    float scaling_factor,
    bool is_causal,
    float dropout_probability,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const std::optional<Tensor>& attn_bias,
    const Tensor& o,
    const Tensor& dO,
    const Tensor& softmaxstats,
    Tensor& dQ,
    Tensor& dK,
    Tensor& dV,
    const Tensor& dropoutseed,
    const Tensor& dropoutoffset);

void run_cudnn_SDP_bprop_nestedtensor(
    int64_t b,
    int64_t h_q,
    int64_t h_k,
    int64_t h_v,
    int64_t s_q,
    int64_t s_kv,
    int64_t d_qk,
    int64_t d_v,
    float scaling_factor,
    bool is_causal,
    float dropout_probability,
    const Tensor& cum_seqlen_q,
    const Tensor& cum_seqlen_kv,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const std::optional<Tensor>& attn_bias,
    const Tensor& o,
    const Tensor& dO,
    const Tensor& softmaxstats,
    Tensor& dQ,
    Tensor& dK,
    Tensor& dV,
    const Tensor& dropoutseed,
    const Tensor& dropoutoffset);

} // namespace at::native
