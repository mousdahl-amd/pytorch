#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/transformers/hip/flash_attn/hipdnn/sdpa_hipdnn.h>

#if AT_HIPDNN_ENABLED()

#include <ATen/TensorIndexing.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/HIPGeneratorImpl.h>
#include <ATen/hip/PhiloxCudaState.h>
#include <ATen/hipdnn/Handle.h>
#include <ATen/hipdnn/hipdnn_frontend.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>
#include <ATen/native/utils/ParamsHash.h>
#include <c10/hip/HIPGuard.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/arange.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/searchsorted.h>
#include <ATen/ops/zeros.h>
#endif

#include <limits>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace at::cuda::philox {
// Defined in attention.cu. Writes the Philox seed and offset to device
// scalars, resolving a graph-captured state on the device.
void unpack_cudnn_wrapper(
    at::PhiloxCudaState arg,
    int64_t* seed_ptr,
    int64_t* offset_ptr,
    hipStream_t stream);
} // namespace at::cuda::philox

namespace pytorch_flash {
namespace {

namespace fe = hipdnn_frontend;
namespace gr = hipdnn_frontend::graph;

enum Uid : int64_t {
  kQ = 1,
  kK,
  kV,
  kO,
  kStats,
  kBias,
  kSeed,
  kOffset,
  kRngDump,
  kSeqLenQ,
  kSeqLenKV,
  kOffsetsQ,
  kOffsetsKV,
};

enum class Alignment : int8_t { None, TopLeft, BottomRight };

// hipDNN's diagonal band. A bound of -1 is unbounded.
struct Band {
  int64_t left = -1;
  int64_t right = -1;
  Alignment alignment = Alignment::None;
};

// FlashAttention's window semantics (also vLLM's): query i keeps keys
// [i + Sk - Sq - left, i + Sk - Sq + right], with a negative or missing size
// unbounded and causal meaning right = 0. hipDNN's left bound counts the
// diagonal itself, so it is left + 1.
Band flash_band(
    bool is_causal,
    std::optional<int64_t> window_left,
    std::optional<int64_t> window_right) {
  Band band;
  const int64_t left = window_left.value_or(-1);
  const int64_t right = window_right.value_or(-1);
  if (left >= 0) {
    band.left = left + 1;
  }
  if (is_causal) {
    band.right = 0;
  } else if (right >= 0) {
    band.right = right;
  }
  if (band.left >= 0 || band.right >= 0) {
    band.alignment = Alignment::BottomRight;
  }
  return band;
}

Band mem_efficient_band(int64_t custom_mask_type) {
  Band band;
  switch (static_cast<sdp::CustomMaskType>(custom_mask_type)) {
    case sdp::CustomMaskType::NoCustomMask:
      break;
    case sdp::CustomMaskType::CausalFromTopLeft:
      band.right = 0;
      band.alignment = Alignment::TopLeft;
      break;
    case sdp::CustomMaskType::CausalFromBottomRight:
      band.right = 0;
      band.alignment = Alignment::BottomRight;
      break;
    default:
      TORCH_CHECK(false, "Unsupported custom_mask_type ", custom_mask_type);
  }
  return band;
}

// Everything the forward graph depends on. Strides are hipDNN's (B, H, S)
// strides of a (B, H, S, D) view whose last dimension is contiguous; for a
// ragged (THD) operand, S is the token stride and B is S times the maximum
// sequence length.
struct FwdKey {
  c10::DeviceIndex device;
  at::ScalarType dtype;
  bool ragged;
  bool seq_len_q; // ragged: per-batch query lengths plus a padding mask
  bool bias;
  bool dropout;
  bool rng_dump;
  Alignment alignment;
  int64_t left_bound;
  int64_t right_bound;
  int64_t b, hq, hk, sq, sk, d, dv;
  int64_t q[3], k[3], v[3], o[3];
  int64_t bias_dim[4];
  int64_t bias_stride[4];
  float scale;
  float dropout_p;
};

using FwdKeyWrapper = at::native::ParamsWrapper<FwdKey>;

struct FwdPlan {
  std::shared_ptr<gr::Graph> graph;
  int64_t workspace = 0;
};

fe::DataType to_hipdnn(at::ScalarType t) {
  switch (t) {
    case at::kHalf:
      return fe::DataType::HALF;
    case at::kBFloat16:
      return fe::DataType::BFLOAT16;
    case at::kFloat:
      return fe::DataType::FLOAT;
    default:
      TORCH_CHECK(false, "hipDNN SDPA does not support dtype ", t);
  }
}

std::shared_ptr<gr::TensorAttributes> make_tensor(
    gr::Graph& graph,
    int64_t uid,
    const char* name,
    fe::DataType dt,
    std::vector<int64_t> dim,
    std::vector<int64_t> stride) {
  return graph.tensor(gr::TensorAttributes()
                          .set_uid(uid)
                          .set_name(name)
                          .set_data_type(dt)
                          .set_dim(std::move(dim))
                          .set_stride(std::move(stride)));
}

std::shared_ptr<gr::TensorAttributes> make_scalar(
    gr::Graph& graph,
    int64_t uid,
    const char* name,
    fe::DataType dt,
    int64_t n) {
  return make_tensor(graph, uid, name, dt, {n, 1, 1, 1}, {1, 1, 1, 1});
}

FwdPlan build_plan(const FwdKey& key, hipdnnHandle_t handle) {
  const auto dt = to_hipdnn(key.dtype);
  auto graph = std::make_shared<gr::Graph>();
  graph->set_io_data_type(dt)
      .set_intermediate_data_type(fe::DataType::FLOAT)
      .set_compute_data_type(fe::DataType::FLOAT);

  auto q = make_tensor(
      *graph, kQ, "Q", dt, {key.b, key.hq, key.sq, key.d},
      {key.q[0], key.q[1], key.q[2], 1});
  auto k = make_tensor(
      *graph, kK, "K", dt, {key.b, key.hk, key.sk, key.d},
      {key.k[0], key.k[1], key.k[2], 1});
  auto v = make_tensor(
      *graph, kV, "V", dt, {key.b, key.hk, key.sk, key.dv},
      {key.v[0], key.v[1], key.v[2], 1});

  gr::SdpaAttributes attrs;
  // Statistics are always generated: that is the form validated on every
  // engine, and the logsumexp is needed for training anyway.
  attrs.set_name("sdpa_fwd").set_attn_scale(key.scale).set_generate_stats(true);
  if (key.left_bound >= 0) {
    attrs.set_diagonal_band_left_bound(key.left_bound);
  }
  if (key.right_bound >= 0) {
    attrs.set_diagonal_band_right_bound(key.right_bound);
  }
  if (key.alignment != Alignment::None) {
    attrs.set_diagonal_alignment(
        key.alignment == Alignment::BottomRight
            ? fe::DiagonalAlignment::BOTTOM_RIGHT
            : fe::DiagonalAlignment::TOP_LEFT);
  }

  std::shared_ptr<gr::TensorAttributes> offsets_q;
  if (key.ragged) {
    offsets_q = make_scalar(
        *graph, kOffsetsQ, "OffsetsQ", fe::DataType::INT32, key.b + 1);
    auto offsets_kv = make_scalar(
        *graph, kOffsetsKV, "OffsetsKV", fe::DataType::INT32, key.b + 1);
    q->set_ragged_offset(offsets_q).set_ragged_offset_multiplier(key.q[2]);
    k->set_ragged_offset(offsets_kv).set_ragged_offset_multiplier(key.k[2]);
    v->set_ragged_offset(offsets_kv).set_ragged_offset_multiplier(key.v[2]);
    attrs.set_seq_len_kv(make_scalar(
        *graph, kSeqLenKV, "SeqLenKV", fe::DataType::INT32, key.b));
    if (key.seq_len_q) {
      attrs
          .set_seq_len_q(make_scalar(
              *graph, kSeqLenQ, "SeqLenQ", fe::DataType::INT32, key.b))
          .set_padding_mask(true);
    }
  }
  if (key.bias) {
    attrs.set_bias(make_tensor(
        *graph, kBias, "Bias", dt,
        {key.bias_dim[0], key.bias_dim[1], key.bias_dim[2], key.bias_dim[3]},
        {key.bias_stride[0], key.bias_stride[1], key.bias_stride[2],
         key.bias_stride[3]}));
  }
  if (key.dropout) {
    attrs.set_dropout(
        key.dropout_p,
        make_scalar(*graph, kSeed, "Seed", fe::DataType::INT64, 1),
        make_scalar(*graph, kOffset, "Offset", fe::DataType::INT64, 1));
    if (key.rng_dump) {
      auto rng = make_tensor(
          *graph, kRngDump, "RngDump", fe::DataType::FLOAT,
          {key.b, key.hq, key.sq, key.sk},
          {key.hq * key.sq * key.sk, key.sq * key.sk, key.sk, 1});
      rng->set_output(true);
      attrs.set_rng_dump(rng);
    }
  }

  auto [o, stats] = graph->sdpa(q, k, v, attrs);
  o->set_uid(kO)
      .set_output(true)
      .set_data_type(dt)
      .set_dim({key.b, key.hq, key.sq, key.dv})
      .set_stride({key.o[0], key.o[1], key.o[2], 1});
  stats->set_uid(kStats)
      .set_output(true)
      .set_data_type(fe::DataType::FLOAT)
      .set_dim({key.b, key.hq, key.sq, 1});
  if (key.ragged) {
    o->set_ragged_offset(offsets_q).set_ragged_offset_multiplier(key.o[2]);
    // Token-major (total_q, H) through the query offsets.
    stats->set_stride({key.sq * key.hq, 1, key.hq, 1})
        .set_ragged_offset(offsets_q)
        .set_ragged_offset_multiplier(key.hq);
  } else {
    stats->set_stride({key.hq * key.sq, key.sq, 1, 1});
  }

  fe::Error err = graph->validate();
  if (err.is_good()) {
    err = graph->is_supported_ext(handle);
  }
  if (err.is_good()) {
    err = graph->create_execution_plans();
  }
  if (err.is_good()) {
    err = graph->check_support();
  }
  if (err.is_good()) {
    err = graph->build_plans();
  }
  FwdPlan plan;
  if (err.is_good()) {
    err = graph->get_workspace_size(plan.workspace);
  }
  TORCH_CHECK(
      err.is_good(),
      "hipDNN has no SDPA forward engine for this configuration (dtype ",
      key.dtype, ", batch ", key.b, ", heads ", key.hq, "/", key.hk,
      ", seqlen ", key.sq, "/", key.sk, ", head dims ", key.d, "/", key.dv,
      key.ragged ? ", varlen" : "", key.bias ? ", bias" : "",
      key.dropout ? ", dropout" : "", "): ", err.get_message());
  plan.graph = std::move(graph);
  return plan;
}

// Plans are built per thread, as hipDNN handles are, and kept for the life of
// the thread.
const FwdPlan& get_plan(const FwdKeyWrapper& key, hipdnnHandle_t handle) {
  thread_local std::unordered_map<
      FwdKeyWrapper,
      FwdPlan,
      at::native::ParamsWrapperHash<FwdKeyWrapper>>
      cache;
  auto it = cache.find(key);
  if (it == cache.end()) {
    it = cache.emplace(key, build_plan(key.pod, handle)).first;
  }
  return it->second;
}

void execute(
    const FwdPlan& plan,
    hipdnnHandle_t handle,
    std::unordered_map<int64_t, void*>& variant_pack,
    const at::Tensor& like) {
  at::Tensor workspace;
  if (plan.workspace > 0) {
    workspace = at::empty({plan.workspace}, like.options().dtype(at::kByte));
  }
  const fe::Error err = plan.graph->execute(
      handle,
      variant_pack,
      workspace.defined() ? workspace.data_ptr() : nullptr);
  TORCH_CHECK(err.is_good(), "hipDNN SDPA forward failed: ", err.get_message());
}

void set_strides(int64_t (&dst)[3], const at::Tensor& t, int64_t seq_dim, int64_t max_seqlen) {
  // seq_dim is the token dimension: 1 for BSHD, 0 for THD. Heads follow it.
  const int64_t token = t.stride(seq_dim);
  dst[0] = seq_dim == 0 ? max_seqlen * token : t.stride(0);
  dst[1] = t.stride(seq_dim + 1);
  dst[2] = token;
}

// ParamsWrapper zeroes the key (padding included) and cannot be moved, so keys
// are initialized in place.
void init_key(
    FwdKeyWrapper& key,
    const at::Tensor& q,
    const Band& band,
    float scale,
    float dropout_p,
    bool rng_dump) {
  key.pod.device = q.get_device();
  key.pod.dtype = q.scalar_type();
  key.pod.alignment = band.alignment;
  key.pod.left_bound = band.left;
  key.pod.right_bound = band.right;
  key.pod.scale = scale;
  key.pod.dropout = dropout_p > 0.f;
  key.pod.dropout_p = key.pod.dropout ? dropout_p : 0.f;
  key.pod.rng_dump = key.pod.dropout && rng_dump;
}

// Reserves one Philox counter per attention-matrix element and writes the
// seed and offset to device scalars, which is also what keeps them valid
// under graph capture. [INFERENCE] hipDNN's dropout (SEM-015) draws one
// counter per element, as AOTriton does.
std::tuple<at::Tensor, at::Tensor> philox_seed_offset(
    float p_dropout,
    int64_t counter_offset,
    const std::optional<at::Generator>& gen_,
    const at::Tensor& like) {
  const auto options = like.options().dtype(at::kLong);
  at::Tensor seed_t = at::empty({}, options);
  at::Tensor offset_t = at::empty({}, options);
  if (p_dropout > 0.f) {
    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
        gen_, at::cuda::detail::getDefaultCUDAGenerator());
    at::PhiloxCudaState state;
    {
      // See Note [Acquire lock when using random generators]
      std::lock_guard<std::mutex> lock(gen->mutex_);
      state = gen->philox_cuda_state(counter_offset);
    }
    at::cuda::philox::unpack_cudnn_wrapper(
        state,
        seed_t.data_ptr<int64_t>(),
        offset_t.data_ptr<int64_t>(),
        at::cuda::getCurrentCUDAStream().stream());
  }
  return {seed_t, offset_t};
}

void check_flash_inputs(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v) {
  const auto dtype = q.scalar_type();
  TORCH_CHECK(
      dtype == at::kHalf || dtype == at::kBFloat16,
      "FlashAttention only support fp16 and bf16 data type");
  TORCH_CHECK(k.scalar_type() == dtype, "query and key must have the same dtype");
  TORCH_CHECK(v.scalar_type() == dtype, "query and value must have the same dtype");
  for (const at::Tensor* t : {&q, &k, &v}) {
    TORCH_CHECK(t->is_cuda(), "FlashAttention inputs must be on a GPU");
    TORCH_CHECK(
        t->stride(-1) == 1, "Input tensor must have contiguous last dimension");
  }
}

at::Tensor check_or_make_out(
    std::optional<at::Tensor>& out_,
    const at::Tensor& q,
    at::IntArrayRef shape) {
  if (!out_.has_value()) {
    return at::empty_like(q);
  }
  at::Tensor out = out_.value();
  TORCH_CHECK(out.scalar_type() == q.scalar_type(), "Output must have the same dtype as inputs");
  TORCH_CHECK(out.is_cuda(), "Output must be on a GPU");
  TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
  TORCH_CHECK(out.sizes() == shape, "Output has shape ", out.sizes(), ", expected ", shape);
  return out;
}

// Per-batch lengths from cumulative offsets: cu[1:] - cu[:-1].
at::Tensor lengths_from_offsets(const at::Tensor& cu) {
  const int64_t n = cu.size(0) - 1;
  return cu.slice(0, 1, n + 1).sub(cu.slice(0, 0, n));
}

void check_offsets(const at::Tensor& t, int64_t n, const char* name) {
  TORCH_CHECK(t.scalar_type() == at::kInt, name, " must have dtype int32");
  TORCH_CHECK(t.is_cuda(), name, " must be on a GPU");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(t.dim() == 1 && t.size(0) == n, name, " must have shape (", n, ")");
}

} // namespace

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
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    std::optional<at::Tensor>& out_,
    std::optional<at::Tensor>& alibi_slopes_,
    const float p_dropout,
    const float softmax_scale,
    bool is_causal,
    std::optional<int64_t> window_size_left,
    std::optional<int64_t> window_size_right,
    const bool return_softmax,
    const std::optional<at::Generator>& gen_) {
  TORCH_CHECK(
      !alibi_slopes_.has_value(),
      "FlashAttention through hipDNN does not support alibi_slopes");
  check_flash_inputs(q, k, v);
  c10::cuda::CUDAGuard device_guard(q.device());

  const int64_t batch_size = q.size(0);
  const int64_t seqlen_q = q.size(1);
  const int64_t num_heads = q.size(2);
  const int64_t head_size = q.size(3);
  const int64_t seqlen_k = k.size(1);
  const int64_t num_heads_k = k.size(2);
  TORCH_CHECK(head_size % 8 == 0, "head_size must be a multiple of 8, this is ensured by padding!");
  TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");
  TORCH_CHECK(k.sizes() == at::IntArrayRef({batch_size, seqlen_k, num_heads_k, head_size}), "key has shape ", k.sizes());
  TORCH_CHECK(v.sizes() == k.sizes(), "value has shape ", v.sizes(), ", expected ", k.sizes());

  if (seqlen_q == 1) {
    is_causal = false; // the same as causal with one query
  }

  at::Tensor out = check_or_make_out(out_, q, q.sizes());
  at::Tensor softmax_lse = at::empty(
      {batch_size, num_heads, seqlen_q}, q.options().dtype(at::kFloat));
  auto [seed_t, offset_t] = philox_seed_offset(
      p_dropout, batch_size * num_heads * seqlen_q * seqlen_k, gen_, q);
  // The debug mask is hipDNN's dropout keep mask (rng_dump), in fp32.
  const bool dump_mask = return_softmax && p_dropout > 0.f;
  at::Tensor debug_mask = dump_mask
      ? at::empty({batch_size, num_heads, seqlen_q, seqlen_k}, q.options().dtype(at::kFloat))
      : at::empty({0, 0, 0, 0}, q.options());
  if (out.numel() == 0) {
    return {out, q, k, v, softmax_lse, seed_t, offset_t, debug_mask};
  }

  FwdKeyWrapper key;
  init_key(
      key, q, flash_band(is_causal, window_size_left, window_size_right),
      softmax_scale, p_dropout, dump_mask);
  key.pod.b = batch_size;
  key.pod.hq = num_heads;
  key.pod.hk = num_heads_k;
  key.pod.sq = seqlen_q;
  key.pod.sk = seqlen_k;
  key.pod.d = head_size;
  key.pod.dv = head_size;
  set_strides(key.pod.q, q, 1, seqlen_q);
  set_strides(key.pod.k, k, 1, seqlen_k);
  set_strides(key.pod.v, v, 1, seqlen_k);
  set_strides(key.pod.o, out, 1, seqlen_q);

  auto handle = at::native::getHipdnnHandle();
  const FwdPlan& plan = get_plan(key, handle);
  std::unordered_map<int64_t, void*> pack{
      {kQ, q.data_ptr()},
      {kK, k.data_ptr()},
      {kV, v.data_ptr()},
      {kO, out.data_ptr()},
      {kStats, softmax_lse.data_ptr()}};
  if (key.pod.dropout) {
    pack[kSeed] = seed_t.data_ptr();
    pack[kOffset] = offset_t.data_ptr();
  }
  if (key.pod.rng_dump) {
    pack[kRngDump] = debug_mask.data_ptr();
  }
  execute(plan, handle, pack, q);
  return {out, q, k, v, softmax_lse, seed_t, offset_t, debug_mask};
}

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
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    std::optional<at::Tensor>& out_,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    std::optional<at::Tensor>& seqused_k,
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
    const std::optional<at::Generator>& gen_) {
  TORCH_CHECK(!block_table_.has_value(), "[ROCm] mha_varlen_fwd: block_table_ must be nullopt");
  TORCH_CHECK(!alibi_slopes_.has_value(), "[ROCm] mha_varlen_fwd: alibi_slopes_ must be nullopt");
  // hipDNN has no dropout mask dump for packed sequences.
  TORCH_CHECK(
      !return_softmax,
      "FlashAttention through hipDNN cannot return the dropout mask for variable-length inputs");
  check_flash_inputs(q, k, v);
  c10::cuda::CUDAGuard device_guard(q.device());

  const int64_t batch_size = cu_seqlens_q.numel() - 1;
  const int64_t total_q = q.size(0);
  const int64_t num_heads = q.size(1);
  const int64_t head_size = q.size(2);
  const int64_t total_k = k.size(0);
  const int64_t num_heads_k = k.size(1);
  TORCH_CHECK(batch_size > 0, "batch size must be positive");
  TORCH_CHECK(head_size % 8 == 0, "head_size must be a multiple of 8, this is ensured by padding!");
  TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");
  TORCH_CHECK(k.sizes() == at::IntArrayRef({total_k, num_heads_k, head_size}), "key has shape ", k.sizes());
  TORCH_CHECK(v.sizes() == k.sizes(), "value has shape ", v.sizes(), ", expected ", k.sizes());
  check_offsets(cu_seqlens_q, batch_size + 1, "cu_seqlens_q");
  check_offsets(cu_seqlens_k, batch_size + 1, "cu_seqlens_k");
  if (seqused_k.has_value()) {
    check_offsets(seqused_k.value(), batch_size, "seqused_k");
  }

  if (max_seqlen_q == 1) {
    is_causal = false; // the same as causal with one query
  }

  at::Tensor out = check_or_make_out(out_, q, q.sizes());
  // PyTorch's varlen logsumexp is head-major (H, total_q); hipDNN writes it
  // token-major, so it goes through a temporary and one transposing copy.
  at::Tensor softmax_lse =
      at::empty({num_heads, total_q}, q.options().dtype(at::kFloat));
  at::Tensor debug_mask = at::empty({0, 0, 0, 0}, q.options());
  if (zero_tensors) {
    out.zero_();
    softmax_lse.fill_(-std::numeric_limits<float>::infinity());
  }
  auto [seed_t, offset_t] = philox_seed_offset(
      p_dropout,
      batch_size * num_heads * int64_t{max_seqlen_q} * int64_t{max_seqlen_k},
      gen_,
      q);

  if (max_seqlen_k <= 0 || total_q == 0) {
    // No keys: an empty attention, as the AOTriton backend reports it.
    out.zero_();
    softmax_lse.fill_(std::numeric_limits<float>::infinity());
    return {out, q, k, v, softmax_lse, seed_t, offset_t, debug_mask};
  }

  FwdKeyWrapper key;
  init_key(
      key, q, flash_band(is_causal, window_size_left, window_size_right),
      softmax_scale, p_dropout, false);
  key.pod.ragged = true;
  // seqused_k is hipDNN's "ragged queries with seq_len_kv alone" form.
  key.pod.seq_len_q = !seqused_k.has_value();
  key.pod.b = batch_size;
  key.pod.hq = num_heads;
  key.pod.hk = num_heads_k;
  key.pod.sq = max_seqlen_q;
  key.pod.sk = max_seqlen_k;
  key.pod.d = head_size;
  key.pod.dv = head_size;
  set_strides(key.pod.q, q, 0, max_seqlen_q);
  set_strides(key.pod.k, k, 0, max_seqlen_k);
  set_strides(key.pod.v, v, 0, max_seqlen_k);
  set_strides(key.pod.o, out, 0, max_seqlen_q);

  at::Tensor lse_token_major =
      at::empty({total_q, num_heads}, q.options().dtype(at::kFloat));
  at::Tensor seq_len_kv = seqused_k.has_value()
      ? seqused_k.value()
      : lengths_from_offsets(cu_seqlens_k);
  at::Tensor seq_len_q;
  if (key.pod.seq_len_q) {
    seq_len_q = lengths_from_offsets(cu_seqlens_q);
  }

  auto handle = at::native::getHipdnnHandle();
  const FwdPlan& plan = get_plan(key, handle);
  std::unordered_map<int64_t, void*> pack{
      {kQ, q.data_ptr()},
      {kK, k.data_ptr()},
      {kV, v.data_ptr()},
      {kO, out.data_ptr()},
      {kStats, lse_token_major.data_ptr()},
      {kOffsetsQ, cu_seqlens_q.data_ptr()},
      {kOffsetsKV, cu_seqlens_k.data_ptr()},
      {kSeqLenKV, seq_len_kv.data_ptr()}};
  if (key.pod.seq_len_q) {
    pack[kSeqLenQ] = seq_len_q.data_ptr();
  }
  if (key.pod.dropout) {
    pack[kSeed] = seed_t.data_ptr();
    pack[kOffset] = offset_t.data_ptr();
  }
  execute(plan, handle, pack, q);
  softmax_lse.copy_(lse_token_major.t());
  return {out, q, k, v, softmax_lse, seed_t, offset_t, debug_mask};
}

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
    at::Tensor& out) {
  const auto dtype = query.scalar_type();
  TORCH_CHECK(
      dtype == at::kHalf || dtype == at::kBFloat16 || dtype == at::kFloat,
      "Efficient attention through hipDNN supports fp16, bf16 and fp32, got ", dtype);
  TORCH_CHECK(key.scalar_type() == dtype && value.scalar_type() == dtype,
      "query, key and value must have the same dtype");
  const bool varlen = seqstart_q.has_value();
  TORCH_CHECK(
      !(varlen && bias.has_value()),
      "Efficient attention through hipDNN does not support a bias with variable-length inputs");

  const int64_t num_heads = query.size(2);
  const int64_t num_heads_k = key.size(2);
  const int64_t batch_size = varlen ? seqstart_q->size(0) - 1 : query.size(0);
  const int64_t seqlen_q = varlen ? max_seqlen_q : query.size(1);
  const int64_t seqlen_k = varlen ? max_seqlen_k : key.size(1);
  const int64_t seq_dim = 1; // (B or 1, tokens, H, D)

  FwdKeyWrapper k;
  init_key(
      k, query, mem_efficient_band(custom_mask_type), softmax_scale,
      static_cast<float>(dropout_p), false);
  k.pod.b = batch_size;
  k.pod.hq = num_heads;
  k.pod.hk = num_heads_k;
  k.pod.sq = seqlen_q;
  k.pod.sk = seqlen_k;
  k.pod.d = query.size(3);
  k.pod.dv = value.size(3);
  // A varlen operand is (1, total, H, D): hipDNN sees its token dimension as
  // ragged, so it takes the THD strides with the leading 1 dropped.
  auto strides = [&](int64_t (&dst)[3], const at::Tensor& t, int64_t max_seqlen) {
    if (varlen) {
      set_strides(dst, t.squeeze(0), 0, max_seqlen);
    } else {
      set_strides(dst, t, seq_dim, max_seqlen);
    }
  };
  strides(k.pod.q, query, seqlen_q);
  strides(k.pod.k, key, seqlen_k);
  strides(k.pod.v, value, seqlen_k);
  strides(k.pod.o, out, seqlen_q);

  // A broadcast bias dimension (size 1, or an expanded stride of 0) is size 1
  // to hipDNN. The last dimension must be dense.
  at::Tensor bias_t;
  if (bias.has_value()) {
    bias_t = bias->stride(-1) == 1 ? *bias : bias->contiguous();
    TORCH_CHECK(bias_t.dim() == 4, "attn_bias must be 4-D, got ", bias_t.dim(), "-D");
    TORCH_CHECK(bias_t.scalar_type() == dtype, "attn_bias must have the dtype of query");
    k.pod.bias = true;
    for (int64_t i = 3; i >= 0; --i) {
      const bool broadcast = bias_t.size(i) == 1 || bias_t.stride(i) == 0;
      k.pod.bias_dim[i] = broadcast ? 1 : bias_t.size(i);
      k.pod.bias_stride[i] = !broadcast
          ? bias_t.stride(i)
          : (i == 3 ? 1 : k.pod.bias_dim[i + 1] * k.pod.bias_stride[i + 1]);
    }
  }

  const auto float_options = query.options().dtype(at::kFloat);
  if (out.numel() == 0) {
    return at::zeros(
        {batch_size, num_heads, compute_logsumexp ? seqlen_q : 0}, float_options);
  }

  at::Tensor stats;
  at::Tensor seq_len_q;
  at::Tensor seq_len_kv;
  if (varlen) {
    check_offsets(*seqstart_q, batch_size + 1, "seqstart_q");
    check_offsets(*seqstart_k, batch_size + 1, "seqstart_k");
    k.pod.ragged = true;
    k.pod.seq_len_q = true;
    stats = at::empty({query.size(1), num_heads}, float_options);
    seq_len_q = lengths_from_offsets(*seqstart_q);
    seq_len_kv = lengths_from_offsets(*seqstart_k);
  } else {
    stats = at::empty({batch_size, num_heads, seqlen_q}, float_options);
  }

  auto handle = at::native::getHipdnnHandle();
  const FwdPlan& plan = get_plan(k, handle);
  std::unordered_map<int64_t, void*> pack{
      {kQ, query.data_ptr()},
      {kK, key.data_ptr()},
      {kV, value.data_ptr()},
      {kO, out.data_ptr()},
      {kStats, stats.data_ptr()}};
  if (varlen) {
    pack[kOffsetsQ] = seqstart_q->data_ptr();
    pack[kOffsetsKV] = seqstart_k->data_ptr();
    pack[kSeqLenQ] = seq_len_q.data_ptr();
    pack[kSeqLenKV] = seq_len_kv.data_ptr();
  }
  if (k.pod.bias) {
    pack[kBias] = bias_t.data_ptr();
  }
  if (k.pod.dropout) {
    pack[kSeed] = philox_seed.data_ptr();
    pack[kOffset] = philox_offset.data_ptr();
  }
  execute(plan, handle, pack, query);

  if (!compute_logsumexp) {
    return at::empty({batch_size, num_heads, 0}, float_options);
  }
  if (!varlen) {
    return stats;
  }
  // Scatter the token-major (total_q, H) statistics into the padded
  // (B, H, max_seqlen_q) logsumexp of the memory-efficient contract.
  at::Tensor lse = at::zeros({batch_size, num_heads, seqlen_q}, float_options);
  const at::Tensor starts = seqstart_q->to(at::kLong);
  const at::Tensor token = at::arange(query.size(1), starts.options());
  const at::Tensor batch = at::searchsorted(
      starts.slice(0, 1, batch_size + 1), token, /*out_int32=*/false, /*right=*/true);
  const at::Tensor position = token.sub(starts.index_select(0, batch));
  lse.permute({0, 2, 1}).index_put_({batch, position}, stats);
  return lse;
}

} // namespace pytorch_flash

#endif // AT_HIPDNN_ENABLED()
