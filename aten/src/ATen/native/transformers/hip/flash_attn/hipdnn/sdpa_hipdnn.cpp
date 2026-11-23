#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/transformers/hip/flash_attn/hipdnn/sdpa_hipdnn.h>

#if AT_HIPDNN_ENABLED()

#include <ATen/Context.h>
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

#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
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
  kDO,
  kDQ,
  kDK,
  kDV,
  kDBias,
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

// Everything a forward or backward graph depends on. Strides are hipDNN's
// (B, H, S) strides of a (B, H, S, D) view whose last dimension is
// contiguous; for a ragged (THD) operand, S is the token stride and B is S
// times the maximum sequence length.
struct GraphKey {
  c10::DeviceIndex device;
  at::ScalarType dtype;
  bool backward;
  bool ragged;
  bool seq_len_q; // ragged: per-batch query lengths plus a padding mask
  bool bias;
  bool dbias;
  bool dropout;
  bool rng_dump;
  Alignment alignment;
  int64_t left_bound;
  int64_t right_bound;
  int64_t b, hq, hk, sq, sk, d, dv;
  // Backward, ragged: the head stride of the head-major (H, total_q) stats.
  int64_t total_q;
  int64_t q[3], k[3], v[3], o[3];
  int64_t dout[3], dq[3], dk[3], dv_[3];
  int64_t bias_dim[4];
  int64_t bias_stride[4];
  float scale;
  float dropout_p;
};

using GraphKeyWrapper = at::native::ParamsWrapper<GraphKey>;

struct Plan {
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

std::vector<int64_t> bhsd_strides(const int64_t (&s)[3]) {
  return {s[0], s[1], s[2], 1};
}

// Attributes shared by SdpaAttributes and SdpaBackwardAttributes: the band,
// the bias, dropout and, for ragged inputs, the sequence lengths. Returns
// the query and key/value offset tables (null unless ragged).
template <typename Attributes>
std::pair<std::shared_ptr<gr::TensorAttributes>, std::shared_ptr<gr::TensorAttributes>>
set_common_attributes(
    gr::Graph& graph,
    const GraphKey& key,
    fe::DataType dt,
    Attributes& attrs) {
  attrs.set_attn_scale(key.scale);
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
  std::shared_ptr<gr::TensorAttributes> offsets_q, offsets_kv;
  if (key.ragged) {
    offsets_q = make_scalar(
        graph, kOffsetsQ, "OffsetsQ", fe::DataType::INT32, key.b + 1);
    offsets_kv = make_scalar(
        graph, kOffsetsKV, "OffsetsKV", fe::DataType::INT32, key.b + 1);
    attrs.set_seq_len_kv(make_scalar(
        graph, kSeqLenKV, "SeqLenKV", fe::DataType::INT32, key.b));
    if (key.seq_len_q) {
      attrs
          .set_seq_len_q(make_scalar(
              graph, kSeqLenQ, "SeqLenQ", fe::DataType::INT32, key.b))
          .set_padding_mask(true);
    }
  }
  if (key.bias) {
    const std::vector<int64_t> dim{
        key.bias_dim[0], key.bias_dim[1], key.bias_dim[2], key.bias_dim[3]};
    const std::vector<int64_t> stride{
        key.bias_stride[0], key.bias_stride[1], key.bias_stride[2],
        key.bias_stride[3]};
    attrs.set_bias(make_tensor(graph, kBias, "Bias", dt, dim, stride));
  }
  if (key.dropout) {
    attrs.set_dropout(
        key.dropout_p,
        make_scalar(graph, kSeed, "Seed", fe::DataType::INT64, 1),
        make_scalar(graph, kOffset, "Offset", fe::DataType::INT64, 1));
  }
  return {offsets_q, offsets_kv};
}

void make_ragged(
    const std::shared_ptr<gr::TensorAttributes>& t,
    const std::shared_ptr<gr::TensorAttributes>& offsets,
    const int64_t (&strides)[3]) {
  t->set_ragged_offset(offsets).set_ragged_offset_multiplier(strides[2]);
}

void set_output(
    const std::shared_ptr<gr::TensorAttributes>& t,
    int64_t uid,
    fe::DataType dt,
    std::vector<int64_t> dim,
    const int64_t (&strides)[3]) {
  t->set_uid(uid)
      .set_output(true)
      .set_data_type(dt)
      .set_dim(std::move(dim))
      .set_stride(bhsd_strides(strides));
}

void build_forward(gr::Graph& graph, const GraphKey& key, fe::DataType dt) {
  auto q = make_tensor(graph, kQ, "Q", dt, {key.b, key.hq, key.sq, key.d}, bhsd_strides(key.q));
  auto k = make_tensor(graph, kK, "K", dt, {key.b, key.hk, key.sk, key.d}, bhsd_strides(key.k));
  auto v = make_tensor(graph, kV, "V", dt, {key.b, key.hk, key.sk, key.dv}, bhsd_strides(key.v));

  gr::SdpaAttributes attrs;
  // Statistics are always generated: that is the form validated on every
  // engine, and the logsumexp is needed for training anyway.
  attrs.set_name("sdpa_fwd").set_generate_stats(true);
  auto [offsets_q, offsets_kv] = set_common_attributes(graph, key, dt, attrs);
  if (key.ragged) {
    make_ragged(q, offsets_q, key.q);
    make_ragged(k, offsets_kv, key.k);
    make_ragged(v, offsets_kv, key.v);
  }
  if (key.rng_dump) {
    auto rng = make_tensor(
        graph, kRngDump, "RngDump", fe::DataType::FLOAT,
        {key.b, key.hq, key.sq, key.sk},
        {key.hq * key.sq * key.sk, key.sq * key.sk, key.sk, 1});
    rng->set_output(true);
    attrs.set_rng_dump(rng);
  }

  auto [o, stats] = graph.sdpa(q, k, v, attrs);
  set_output(o, kO, dt, {key.b, key.hq, key.sq, key.dv}, key.o);
  stats->set_uid(kStats)
      .set_output(true)
      .set_data_type(fe::DataType::FLOAT)
      .set_dim({key.b, key.hq, key.sq, 1});
  if (key.ragged) {
    make_ragged(o, offsets_q, key.o);
    // Token-major (total_q, H) through the query offsets.
    stats->set_stride({key.sq * key.hq, 1, key.hq, 1})
        .set_ragged_offset(offsets_q)
        .set_ragged_offset_multiplier(key.hq);
  } else {
    stats->set_stride({key.hq * key.sq, key.sq, 1, 1});
  }
}

void build_backward(gr::Graph& graph, const GraphKey& key, fe::DataType dt) {
  auto q = make_tensor(graph, kQ, "Q", dt, {key.b, key.hq, key.sq, key.d}, bhsd_strides(key.q));
  auto k = make_tensor(graph, kK, "K", dt, {key.b, key.hk, key.sk, key.d}, bhsd_strides(key.k));
  auto v = make_tensor(graph, kV, "V", dt, {key.b, key.hk, key.sk, key.dv}, bhsd_strides(key.v));
  auto o = make_tensor(graph, kO, "O", dt, {key.b, key.hq, key.sq, key.dv}, bhsd_strides(key.o));
  auto dout = make_tensor(graph, kDO, "dO", dt, {key.b, key.hq, key.sq, key.dv}, bhsd_strides(key.dout));
  // Ragged statistics are head-major (H, total_q) through the query offsets:
  // PyTorch's varlen logsumexp layout, as validated in P0.
  auto stats = key.ragged
      ? make_tensor(graph, kStats, "Stats", fe::DataType::FLOAT,
                    {key.b, key.hq, key.sq, 1}, {key.sq, key.total_q, 1, 1})
      : make_tensor(graph, kStats, "Stats", fe::DataType::FLOAT,
                    {key.b, key.hq, key.sq, 1}, {key.hq * key.sq, key.sq, 1, 1});

  gr::SdpaBackwardAttributes attrs;
  attrs.set_name("sdpa_bwd");
  auto [offsets_q, offsets_kv] = set_common_attributes(graph, key, dt, attrs);
  if (key.ragged) {
    make_ragged(q, offsets_q, key.q);
    make_ragged(o, offsets_q, key.o);
    make_ragged(dout, offsets_q, key.dout);
    make_ragged(k, offsets_kv, key.k);
    make_ragged(v, offsets_kv, key.v);
    stats->set_ragged_offset(offsets_q).set_ragged_offset_multiplier(1);
  }
  if (key.dbias) {
    auto dbias = make_tensor(
        graph, kDBias, "dBias", dt,
        {key.bias_dim[0], key.bias_dim[1], key.bias_dim[2], key.bias_dim[3]},
        {key.bias_stride[0], key.bias_stride[1], key.bias_stride[2],
         key.bias_stride[3]});
    dbias->set_output(true);
    attrs.set_dbias(dbias);
  }

  auto [dq, dk, dv] = graph.sdpa_backward(q, k, v, o, dout, stats, attrs);
  set_output(dq, kDQ, dt, {key.b, key.hq, key.sq, key.d}, key.dq);
  set_output(dk, kDK, dt, {key.b, key.hk, key.sk, key.d}, key.dk);
  set_output(dv, kDV, dt, {key.b, key.hk, key.sk, key.dv}, key.dv_);
  if (key.ragged) {
    make_ragged(dq, offsets_q, key.dq);
    make_ragged(dk, offsets_kv, key.dk);
    make_ragged(dv, offsets_kv, key.dv_);
  }
}

Plan build_plan(const GraphKey& key, hipdnnHandle_t handle) {
  const auto dt = to_hipdnn(key.dtype);
  auto graph = std::make_shared<gr::Graph>();
  graph->set_io_data_type(dt)
      .set_intermediate_data_type(fe::DataType::FLOAT)
      .set_compute_data_type(fe::DataType::FLOAT);
  if (key.backward) {
    build_backward(*graph, key, dt);
  } else {
    build_forward(*graph, key, dt);
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
  Plan plan;
  if (err.is_good()) {
    err = graph->get_workspace_size(plan.workspace);
  }
  TORCH_CHECK(
      err.is_good(),
      "hipDNN has no SDPA ", key.backward ? "backward" : "forward",
      " engine for this configuration (dtype ", key.dtype, ", batch ", key.b,
      ", heads ", key.hq, "/", key.hk, ", seqlen ", key.sq, "/", key.sk,
      ", head dims ", key.d, "/", key.dv, key.ragged ? ", varlen" : "",
      key.bias ? ", bias" : "", key.dbias ? ", dbias" : "",
      key.dropout ? ", dropout" : "", "): ", err.get_message());
  plan.graph = std::move(graph);
  return plan;
}

// Plans are built per thread, as hipDNN handles are, and never destroyed: a
// thread's cache is deliberately leaked, as ATen/hipdnn/Handle.cpp leaks its
// pooled handles. Threads such as the autograd workers exit while the process
// tears down, and destroying a hipDNN graph then races with the teardown of
// hipDNN and the HIP runtime on the main thread (seen as segfaults in
// hipdnn_frontend::graph::Graph::~Graph at exit on gfx950).
const Plan& get_plan(const GraphKeyWrapper& key, hipdnnHandle_t handle) {
  using Cache = std::unordered_map<
      GraphKeyWrapper,
      Plan,
      at::native::ParamsWrapperHash<GraphKeyWrapper>>;
  thread_local Cache* cache = new Cache();
  auto it = cache->find(key);
  if (it == cache->end()) {
    it = cache->emplace(key, build_plan(key.pod, handle)).first;
  }
  return it->second;
}

void execute(
    const GraphKeyWrapper& key,
    std::unordered_map<int64_t, void*>& variant_pack,
    const at::Tensor& like) {
  auto handle = at::native::getHipdnnHandle();
  const Plan& plan = get_plan(key, handle);
  at::Tensor workspace;
  if (plan.workspace > 0) {
    workspace = at::empty({plan.workspace}, like.options().dtype(at::kByte));
  }
  const fe::Error err = plan.graph->execute(
      handle,
      variant_pack,
      workspace.defined() ? workspace.data_ptr() : nullptr);
  TORCH_CHECK(
      err.is_good(),
      "hipDNN SDPA ", key.pod.backward ? "backward" : "forward", " failed: ",
      err.get_message());
}

void set_strides(int64_t (&dst)[3], const at::Tensor& t, int64_t seq_dim, int64_t max_seqlen) {
  // seq_dim is the token dimension: 1 for BSHD, 0 for THD. Heads follow it.
  const int64_t token = t.stride(seq_dim);
  dst[0] = seq_dim == 0 ? max_seqlen * token : t.stride(0);
  dst[1] = t.stride(seq_dim + 1);
  dst[2] = token;
}

// Memory-efficient operands are (B, S, H, D), or (1, total, H, D) for packed
// sequences, which hipDNN sees as THD with the leading 1 dropped.
void me_strides(int64_t (&dst)[3], const at::Tensor& t, bool varlen, int64_t max_seqlen) {
  if (varlen) {
    set_strides(dst, t.squeeze(0), 0, max_seqlen);
  } else {
    set_strides(dst, t, 1, max_seqlen);
  }
}

// ParamsWrapper zeroes the key (padding included) and cannot be moved, so keys
// are initialized in place.
void init_key(
    GraphKeyWrapper& key,
    bool backward,
    const at::Tensor& q,
    const Band& band,
    float scale,
    float dropout_p,
    bool rng_dump) {
  key.pod.device = q.get_device();
  key.pod.dtype = q.scalar_type();
  key.pod.backward = backward;
  key.pod.alignment = band.alignment;
  key.pod.left_bound = band.left;
  key.pod.right_bound = band.right;
  key.pod.scale = scale;
  key.pod.dropout = dropout_p > 0.f;
  key.pod.dropout_p = key.pod.dropout ? dropout_p : 0.f;
  key.pod.rng_dump = key.pod.dropout && rng_dump;
}

void set_shape(
    GraphKey& key,
    int64_t b,
    int64_t hq,
    int64_t hk,
    int64_t sq,
    int64_t sk,
    int64_t d,
    int64_t dv) {
  key.b = b;
  key.hq = hq;
  key.hk = hk;
  key.sq = sq;
  key.sk = sk;
  key.d = d;
  key.dv = dv;
}

// A broadcast bias dimension (size 1, or an expanded stride of 0) is size 1
// to hipDNN. The last dimension must be dense. With a bias gradient, the bias
// is materialized instead: PyTorch's grad_bias has the expanded shape, and
// hipDNN then never sees a broadcast dBias.
at::Tensor set_bias(GraphKey& key, const at::Tensor& bias, bool with_grad) {
  TORCH_CHECK(bias.dim() == 4, "attn_bias must be 4-D, got ", bias.dim(), "-D");
  TORCH_CHECK(bias.scalar_type() == key.dtype, "attn_bias must have the dtype of query");
  at::Tensor t = with_grad || bias.stride(-1) != 1 ? bias.contiguous() : bias;
  key.bias = true;
  key.dbias = with_grad;
  for (int64_t i = 3; i >= 0; --i) {
    const bool broadcast = t.size(i) == 1 || t.stride(i) == 0;
    key.bias_dim[i] = broadcast ? 1 : t.size(i);
    key.bias_stride[i] = !broadcast
        ? t.stride(i)
        : (i == 3 ? 1 : key.bias_dim[i + 1] * key.bias_stride[i + 1]);
  }
  return t;
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

// The backward reads the forward's seed and offset; they live on the device
// when the forward ran through hipDNN.
void add_dropout(
    std::unordered_map<int64_t, void*>& pack,
    at::Tensor& seed,
    at::Tensor& offset,
    const at::Tensor& like) {
  if (!seed.is_cuda()) {
    seed = seed.to(like.device());
    offset = offset.to(like.device());
  }
  pack[kSeed] = seed.data_ptr();
  pack[kOffset] = offset.data_ptr();
}

// The backward reads its inputs with 16-byte loads.
at::Tensor aligned_contiguous(const at::Tensor& t) {
  at::Tensor c = t.contiguous();
  return reinterpret_cast<uintptr_t>(c.const_data_ptr()) % 16 == 0 ? c : c.clone();
}

// hipDNN has no deterministic mode yet: its backward is not asked to be
// deterministic until it can promise it (plan H6).
void check_not_deterministic(bool deterministic) {
  TORCH_CHECK(
      !deterministic,
      "SDPA backward through hipDNN does not offer a deterministic algorithm yet");
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

// An optional output (out, dq, dk, dv): checked if given, else like `like`.
at::Tensor check_or_make(
    std::optional<at::Tensor>& given,
    const at::Tensor& like,
    const char* name) {
  if (!given.has_value()) {
    return at::empty_like(like);
  }
  at::Tensor t = given.value();
  TORCH_CHECK(t.scalar_type() == like.scalar_type(), name, " must have the dtype of the inputs");
  TORCH_CHECK(t.is_cuda(), name, " must be on a GPU");
  TORCH_CHECK(t.stride(-1) == 1, name, " must have contiguous last dimension");
  TORCH_CHECK(t.sizes() == like.sizes(), name, " has shape ", t.sizes(), ", expected ", like.sizes());
  return t;
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

void add_ragged(
    std::unordered_map<int64_t, void*>& pack,
    const at::Tensor& offsets_q,
    const at::Tensor& offsets_kv,
    const at::Tensor& seq_len_q,
    const at::Tensor& seq_len_kv) {
  pack[kOffsetsQ] = offsets_q.data_ptr();
  pack[kOffsetsKV] = offsets_kv.data_ptr();
  pack[kSeqLenKV] = seq_len_kv.data_ptr();
  if (seq_len_q.defined()) {
    pack[kSeqLenQ] = seq_len_q.data_ptr();
  }
}

// For each packed query token: its batch index and its position in it.
std::pair<at::Tensor, at::Tensor> token_coordinates(
    const at::Tensor& cu_seqlens,
    int64_t total) {
  const int64_t batch_size = cu_seqlens.size(0) - 1;
  const at::Tensor starts = cu_seqlens.to(at::kLong);
  const at::Tensor token = at::arange(total, starts.options());
  at::Tensor batch = at::searchsorted(
      starts.slice(0, 1, batch_size + 1), token, /*out_int32=*/false, /*right=*/true);
  at::Tensor position = token.sub(starts.index_select(0, batch));
  return {std::move(batch), std::move(position)};
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

  at::Tensor out = check_or_make(out_, q, "out");
  at::Tensor softmax_lse = at::empty(
      {batch_size, num_heads, seqlen_q}, q.options().dtype(at::kFloat));
  auto [seed_t, offset_t] = philox_seed_offset(
      p_dropout, batch_size * num_heads * seqlen_q * seqlen_k, gen_, q);
  // The debug mask comes from hipDNN's dropout keep mask (rng_dump: 1 kept,
  // 0 dropped), re-encoded after the call to the ROCm contract set by
  // AOTriton's encoded softmax: +0.5 kept, -0.5 dropped.
  const bool dump_mask = return_softmax && p_dropout > 0.f;
  at::Tensor debug_mask = dump_mask
      ? at::empty({batch_size, num_heads, seqlen_q, seqlen_k}, q.options().dtype(at::kFloat))
      : at::empty({0, 0, 0, 0}, q.options());
  if (out.numel() == 0) {
    return {out, q, k, v, softmax_lse, seed_t, offset_t, debug_mask};
  }

  GraphKeyWrapper key;
  init_key(
      key, false, q, flash_band(is_causal, window_size_left, window_size_right),
      softmax_scale, p_dropout, dump_mask);
  set_shape(key.pod, batch_size, num_heads, num_heads_k, seqlen_q, seqlen_k, head_size, head_size);
  set_strides(key.pod.q, q, 1, seqlen_q);
  set_strides(key.pod.k, k, 1, seqlen_k);
  set_strides(key.pod.v, v, 1, seqlen_k);
  set_strides(key.pod.o, out, 1, seqlen_q);

  std::unordered_map<int64_t, void*> pack{
      {kQ, q.data_ptr()},
      {kK, k.data_ptr()},
      {kV, v.data_ptr()},
      {kO, out.data_ptr()},
      {kStats, softmax_lse.data_ptr()}};
  if (key.pod.dropout) {
    add_dropout(pack, seed_t, offset_t, q);
  }
  if (key.pod.rng_dump) {
    pack[kRngDump] = debug_mask.data_ptr();
  }
  execute(key, pack, q);
  if (key.pod.rng_dump) {
    debug_mask.sub_(0.5);
  }
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

  at::Tensor out = check_or_make(out_, q, "out");
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

  GraphKeyWrapper key;
  init_key(
      key, false, q, flash_band(is_causal, window_size_left, window_size_right),
      softmax_scale, p_dropout, false);
  key.pod.ragged = true;
  // seqused_k is hipDNN's "ragged queries with seq_len_kv alone" form.
  key.pod.seq_len_q = !seqused_k.has_value();
  set_shape(key.pod, batch_size, num_heads, num_heads_k, max_seqlen_q, max_seqlen_k, head_size, head_size);
  set_strides(key.pod.q, q, 0, max_seqlen_q);
  set_strides(key.pod.k, k, 0, max_seqlen_k);
  set_strides(key.pod.v, v, 0, max_seqlen_k);
  set_strides(key.pod.o, out, 0, max_seqlen_q);

  at::Tensor lse_token_major =
      at::empty({total_q, num_heads}, q.options().dtype(at::kFloat));
  const at::Tensor seq_len_kv = seqused_k.has_value()
      ? seqused_k.value()
      : lengths_from_offsets(cu_seqlens_k);
  const at::Tensor seq_len_q = key.pod.seq_len_q
      ? lengths_from_offsets(cu_seqlens_q)
      : at::Tensor();

  std::unordered_map<int64_t, void*> pack{
      {kQ, q.data_ptr()},
      {kK, k.data_ptr()},
      {kV, v.data_ptr()},
      {kO, out.data_ptr()},
      {kStats, lse_token_major.data_ptr()}};
  add_ragged(pack, cu_seqlens_q, cu_seqlens_k, seq_len_q, seq_len_kv);
  if (key.pod.dropout) {
    add_dropout(pack, seed_t, offset_t, q);
  }
  execute(key, pack, q);
  softmax_lse.copy_(lse_token_major.t());
  return {out, q, k, v, softmax_lse, seed_t, offset_t, debug_mask};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> mha_bwd_hipdnn(
    const at::Tensor& dout,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& out,
    const at::Tensor& softmax_lse,
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
    const at::Tensor& philox_offset) {
  TORCH_CHECK(
      !alibi_slopes_.has_value(),
      "FlashAttention through hipDNN does not support alibi_slopes");
  check_not_deterministic(deterministic);
  check_flash_inputs(q, k, v);
  TORCH_CHECK(out.scalar_type() == q.scalar_type() && dout.scalar_type() == q.scalar_type(),
      "out and dout must have the dtype of query");
  c10::cuda::CUDAGuard device_guard(q.device());

  const int64_t batch_size = q.size(0);
  const int64_t seqlen_q = q.size(1);
  const int64_t num_heads = q.size(2);
  const int64_t head_size = q.size(3);
  const int64_t seqlen_k = k.size(1);
  const int64_t num_heads_k = k.size(2);
  TORCH_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
  TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");
  TORCH_CHECK(out.sizes() == q.sizes() && dout.sizes() == q.sizes(),
      "out and dout must have the shape of query");
  TORCH_CHECK(softmax_lse.sizes() == at::IntArrayRef({batch_size, num_heads, seqlen_q}),
      "softmax_lse has shape ", softmax_lse.sizes());

  if (seqlen_q == 1) {
    is_causal = false; // the same as causal with one query
  }

  at::Tensor dq = check_or_make(dq_, q, "dq");
  at::Tensor dk = check_or_make(dk_, k, "dk");
  at::Tensor dv = check_or_make(dv_, v, "dv");
  if (q.numel() == 0 || k.numel() == 0) {
    dq.zero_();
    dk.zero_();
    dv.zero_();
    return {dq, dk, dv, at::Tensor()};
  }
  const at::Tensor dout_c = aligned_contiguous(dout);
  const at::Tensor lse = aligned_contiguous(softmax_lse);

  GraphKeyWrapper key;
  init_key(
      key, true, q, flash_band(is_causal, window_size_left, window_size_right),
      softmax_scale, p_dropout, false);
  set_shape(key.pod, batch_size, num_heads, num_heads_k, seqlen_q, seqlen_k, head_size, head_size);
  set_strides(key.pod.q, q, 1, seqlen_q);
  set_strides(key.pod.k, k, 1, seqlen_k);
  set_strides(key.pod.v, v, 1, seqlen_k);
  set_strides(key.pod.o, out, 1, seqlen_q);
  set_strides(key.pod.dout, dout_c, 1, seqlen_q);
  set_strides(key.pod.dq, dq, 1, seqlen_q);
  set_strides(key.pod.dk, dk, 1, seqlen_k);
  set_strides(key.pod.dv_, dv, 1, seqlen_k);

  std::unordered_map<int64_t, void*> pack{
      {kQ, q.data_ptr()},
      {kK, k.data_ptr()},
      {kV, v.data_ptr()},
      {kO, out.data_ptr()},
      {kDO, dout_c.data_ptr()},
      {kStats, lse.data_ptr()},
      {kDQ, dq.data_ptr()},
      {kDK, dk.data_ptr()},
      {kDV, dv.data_ptr()}};
  at::Tensor seed = philox_seed;
  at::Tensor offset = philox_offset;
  if (key.pod.dropout) {
    add_dropout(pack, seed, offset, q);
  }
  execute(key, pack, q);
  return {dq, dk, dv, at::Tensor()};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> mha_varlen_bwd_hipdnn(
    const at::Tensor& dout,
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
    const at::Tensor& philox_offset) {
  TORCH_CHECK(!alibi_slopes_.has_value(), "[ROCm] mha_varlen_bwd: alibi_slopes_ must be nullopt");
  check_not_deterministic(deterministic);
  check_flash_inputs(q, k, v);
  TORCH_CHECK(out.scalar_type() == q.scalar_type() && dout.scalar_type() == q.scalar_type(),
      "out and dout must have the dtype of query");
  c10::cuda::CUDAGuard device_guard(q.device());

  const int64_t batch_size = cu_seqlens_q.numel() - 1;
  const int64_t total_q = q.size(0);
  const int64_t num_heads = q.size(1);
  const int64_t head_size = q.size(2);
  const int64_t num_heads_k = k.size(1);
  TORCH_CHECK(batch_size > 0, "batch size must be positive");
  TORCH_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
  TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");
  TORCH_CHECK(out.sizes() == q.sizes() && dout.sizes() == q.sizes(),
      "out and dout must have the shape of query");
  TORCH_CHECK(softmax_lse.sizes() == at::IntArrayRef({num_heads, total_q}),
      "softmax_lse has shape ", softmax_lse.sizes(), ", expected (num_heads, total_q)");
  check_offsets(cu_seqlens_q, batch_size + 1, "cu_seqlens_q");
  check_offsets(cu_seqlens_k, batch_size + 1, "cu_seqlens_k");

  if (max_seqlen_q == 1) {
    is_causal = false; // the same as causal with one query
  }

  at::Tensor dq = check_or_make(dq_, q, "dq");
  at::Tensor dk = check_or_make(dk_, k, "dk");
  at::Tensor dv = check_or_make(dv_, v, "dv");
  if (zero_tensors) {
    dq.zero_();
    dk.zero_();
    dv.zero_();
  }
  if (max_seqlen_k <= 0 || total_q == 0) {
    dq.zero_();
    dk.zero_();
    dv.zero_();
    return {dq, dk, dv, at::Tensor()};
  }
  const at::Tensor dout_c = aligned_contiguous(dout);
  const at::Tensor lse = aligned_contiguous(softmax_lse);

  GraphKeyWrapper key;
  init_key(
      key, true, q, flash_band(is_causal, window_size_left, window_size_right),
      softmax_scale, p_dropout, false);
  key.pod.ragged = true;
  key.pod.seq_len_q = true;
  key.pod.total_q = total_q;
  set_shape(key.pod, batch_size, num_heads, num_heads_k, max_seqlen_q, max_seqlen_k, head_size, head_size);
  set_strides(key.pod.q, q, 0, max_seqlen_q);
  set_strides(key.pod.k, k, 0, max_seqlen_k);
  set_strides(key.pod.v, v, 0, max_seqlen_k);
  set_strides(key.pod.o, out, 0, max_seqlen_q);
  set_strides(key.pod.dout, dout_c, 0, max_seqlen_q);
  set_strides(key.pod.dq, dq, 0, max_seqlen_q);
  set_strides(key.pod.dk, dk, 0, max_seqlen_k);
  set_strides(key.pod.dv_, dv, 0, max_seqlen_k);

  const at::Tensor seq_len_q = lengths_from_offsets(cu_seqlens_q);
  const at::Tensor seq_len_kv = lengths_from_offsets(cu_seqlens_k);
  std::unordered_map<int64_t, void*> pack{
      {kQ, q.data_ptr()},
      {kK, k.data_ptr()},
      {kV, v.data_ptr()},
      {kO, out.data_ptr()},
      {kDO, dout_c.data_ptr()},
      {kStats, lse.data_ptr()},
      {kDQ, dq.data_ptr()},
      {kDK, dk.data_ptr()},
      {kDV, dv.data_ptr()}};
  add_ragged(pack, cu_seqlens_q, cu_seqlens_k, seq_len_q, seq_len_kv);
  at::Tensor seed = philox_seed;
  at::Tensor offset = philox_offset;
  if (key.pod.dropout) {
    add_dropout(pack, seed, offset, q);
  }
  execute(key, pack, q);
  return {dq, dk, dv, at::Tensor()};
}

namespace {

// The parts of a memory-efficient key shared by forward and backward: dtype,
// band, shape, and the strides of query, key, value and out.
void init_me_key(
    GraphKeyWrapper& key,
    bool backward,
    const at::Tensor& query,
    const at::Tensor& k,
    const at::Tensor& value,
    const at::Tensor& out,
    bool varlen,
    int64_t batch_size,
    int64_t seqlen_q,
    int64_t seqlen_k,
    double dropout_p,
    int64_t custom_mask_type,
    float softmax_scale) {
  const auto dtype = query.scalar_type();
  TORCH_CHECK(
      dtype == at::kHalf || dtype == at::kBFloat16 || dtype == at::kFloat,
      "Efficient attention through hipDNN supports fp16, bf16 and fp32, got ", dtype);
  TORCH_CHECK(k.scalar_type() == dtype && value.scalar_type() == dtype,
      "query, key and value must have the same dtype");
  init_key(
      key, backward, query, mem_efficient_band(custom_mask_type), softmax_scale,
      static_cast<float>(dropout_p), false);
  set_shape(
      key.pod, batch_size, query.size(2), k.size(2), seqlen_q, seqlen_k,
      query.size(3), value.size(3));
  key.pod.ragged = varlen;
  key.pod.seq_len_q = varlen;
  me_strides(key.pod.q, query, varlen, seqlen_q);
  me_strides(key.pod.k, k, varlen, seqlen_k);
  me_strides(key.pod.v, value, varlen, seqlen_k);
  me_strides(key.pod.o, out, varlen, seqlen_q);
}

} // namespace

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
  const bool varlen = seqstart_q.has_value();
  TORCH_CHECK(
      !(varlen && bias.has_value()),
      "Efficient attention through hipDNN does not support a bias with variable-length inputs");
  const int64_t num_heads = query.size(2);
  const int64_t batch_size = varlen ? seqstart_q->size(0) - 1 : query.size(0);
  const int64_t seqlen_q = varlen ? max_seqlen_q : query.size(1);
  const int64_t seqlen_k = varlen ? max_seqlen_k : key.size(1);

  GraphKeyWrapper k;
  init_me_key(
      k, false, query, key, value, out, varlen, batch_size, seqlen_q, seqlen_k,
      dropout_p, custom_mask_type, softmax_scale);
  at::Tensor bias_t;
  if (bias.has_value()) {
    bias_t = set_bias(k.pod, *bias, /*with_grad=*/false);
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
    stats = at::empty({query.size(1), num_heads}, float_options);
    seq_len_q = lengths_from_offsets(*seqstart_q);
    seq_len_kv = lengths_from_offsets(*seqstart_k);
  } else {
    stats = at::empty({batch_size, num_heads, seqlen_q}, float_options);
  }

  std::unordered_map<int64_t, void*> pack{
      {kQ, query.data_ptr()},
      {kK, key.data_ptr()},
      {kV, value.data_ptr()},
      {kO, out.data_ptr()},
      {kStats, stats.data_ptr()}};
  if (varlen) {
    add_ragged(pack, *seqstart_q, *seqstart_k, seq_len_q, seq_len_kv);
  }
  if (k.pod.bias) {
    pack[kBias] = bias_t.data_ptr();
  }
  at::Tensor seed = philox_seed;
  at::Tensor offset = philox_offset;
  if (k.pod.dropout) {
    add_dropout(pack, seed, offset, query);
  }
  execute(k, pack, query);

  if (!compute_logsumexp) {
    return at::empty({batch_size, num_heads, 0}, float_options);
  }
  if (!varlen) {
    return stats;
  }
  // Scatter the token-major (total_q, H) statistics into the padded
  // (B, H, max_seqlen_q) logsumexp of the memory-efficient contract.
  at::Tensor lse = at::zeros({batch_size, num_heads, seqlen_q}, float_options);
  const auto [batch, position] = token_coordinates(*seqstart_q, query.size(1));
  lse.permute({0, 2, 1}).index_put_({batch, position}, stats);
  return lse;
}

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
    at::Tensor& grad_v) {
  auto& ctx = at::globalContext();
  check_not_deterministic(
      ctx.deterministicAlgorithms() && !ctx.deterministicAlgorithmsWarnOnly());
  const bool varlen = cu_seqlens_q.has_value();
  TORCH_CHECK(
      !(varlen && bias.has_value()),
      "Efficient attention through hipDNN does not support a bias with variable-length inputs");
  const int64_t num_heads = query.size(2);
  const int64_t batch_size = varlen ? cu_seqlens_q->size(0) - 1 : query.size(0);
  const int64_t seqlen_q = varlen ? max_seqlen_q : query.size(1);
  const int64_t seqlen_k = varlen ? max_seqlen_k : key.size(1);

  GraphKeyWrapper k;
  init_me_key(
      k, true, query, key, value, out, varlen, batch_size, seqlen_q, seqlen_k,
      dropout_p, custom_mask_type, softmax_scale);
  const at::Tensor dout = aligned_contiguous(grad_out);
  me_strides(k.pod.dout, dout, varlen, seqlen_q);
  me_strides(k.pod.dq, grad_q, varlen, seqlen_q);
  me_strides(k.pod.dk, grad_k, varlen, seqlen_k);
  me_strides(k.pod.dv_, grad_v, varlen, seqlen_k);

  at::Tensor bias_t;
  at::Tensor grad_bias;
  if (bias.has_value()) {
    bias_t = set_bias(k.pod, *bias, bias_requires_grad);
    if (bias_requires_grad) {
      grad_bias = at::empty_like(bias_t);
    }
  }

  // The statistics, head-major (H, total_q) for packed sequences: the
  // forward's padded (B, H, max_seqlen_q) logsumexp is gathered into it.
  at::Tensor lse;
  at::Tensor seq_len_q;
  at::Tensor seq_len_kv;
  if (varlen) {
    check_offsets(*cu_seqlens_q, batch_size + 1, "cu_seqlens_q");
    check_offsets(*cu_seqlens_k, batch_size + 1, "cu_seqlens_k");
    const int64_t total_q = query.size(1);
    k.pod.total_q = total_q;
    const auto [batch, position] = token_coordinates(*cu_seqlens_q, total_q);
    lse = aligned_contiguous(
        logsumexp.permute({0, 2, 1}).index({batch, position}).t());
    seq_len_q = lengths_from_offsets(*cu_seqlens_q);
    seq_len_kv = lengths_from_offsets(*cu_seqlens_k);
  } else {
    lse = aligned_contiguous(logsumexp);
  }
  TORCH_CHECK(
      lse.numel() == batch_size * num_heads * seqlen_q || varlen,
      "logsumexp has shape ", logsumexp.sizes(),
      "; the forward must compute the logsumexp for the backward");

  std::unordered_map<int64_t, void*> pack{
      {kQ, query.data_ptr()},
      {kK, key.data_ptr()},
      {kV, value.data_ptr()},
      {kO, out.data_ptr()},
      {kDO, dout.data_ptr()},
      {kStats, lse.data_ptr()},
      {kDQ, grad_q.data_ptr()},
      {kDK, grad_k.data_ptr()},
      {kDV, grad_v.data_ptr()}};
  if (varlen) {
    add_ragged(pack, *cu_seqlens_q, *cu_seqlens_k, seq_len_q, seq_len_kv);
  }
  if (k.pod.bias) {
    pack[kBias] = bias_t.data_ptr();
  }
  if (k.pod.dbias) {
    pack[kDBias] = grad_bias.data_ptr();
  }
  at::Tensor seed = philox_seed;
  at::Tensor offset = philox_offset;
  if (k.pod.dropout) {
    add_dropout(pack, seed, offset, query);
  }
  execute(k, pack, query);
  return grad_bias;
}

} // namespace pytorch_flash

#endif // AT_HIPDNN_ENABLED()
