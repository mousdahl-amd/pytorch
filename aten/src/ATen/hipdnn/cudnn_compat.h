#pragma once

// Brings hipDNN's cuDNN-compatibility shim in under the spellings that
// aten/src/ATen/native/cudnn/MHA.cpp expects, so that file's SDPA graph code is
// shared verbatim between CUDA and ROCm.
//
// This directory is deliberately outside the hipify input set
// (tools/amd_build/build_amd.py): the shim's `cudnnHandle_t` must survive as
// itself and not be rewritten to `miopenHandle_t`.

#include <ATen/cuda/CUDAConfig.h>

#if AT_HIPDNN_ENABLED()

#include <hipdnn_compatibility/cudnn/cudnn_frontend.h>
#include <c10/macros/Export.h>

// The shim lives in its own namespace and expects the consumer to establish the
// cuDNN spelling; see hipdnn_compatibility/cudnn/cudnn_frontend.h.
namespace cudnn_frontend = hipdnn_frontend::compatibility::cudnn_frontend;

// The versions the shim reports are load-bearing, not cosmetic:
//   - >= 8900 / >= 10100 selects the real (non-stub) branch of MHA.cpp;
//   - >= 90000 clears the runtime gate in sdp_utils.cpp;
//   - < 92400 / < 12500 keeps AT_CUDNN_HAS_CUMULATIVE_SEQUENCE_LENGTHS off,
//     which is required because the shim implements neither
//     SDPA_attributes::set_cu_seq_len_q/kv nor set_implementation.
// A shim version bump must therefore be reviewed against MHA.cpp, not absorbed
// silently.
static_assert(
    CUDNN_VERSION >= 90000 && CUDNN_VERSION < 92400,
    "hipDNN's cuDNN-compatibility shim reports a CUDNN_VERSION outside the "
    "range this integration was validated against; re-check MHA.cpp's version "
    "gates before widening this assert.");
static_assert(
    CUDNN_FRONTEND_VERSION >= 10100 && CUDNN_FRONTEND_VERSION < 12500,
    "hipDNN's cuDNN-compatibility shim reports a CUDNN_FRONTEND_VERSION "
    "outside the range this integration was validated against; re-check "
    "MHA.cpp's version gates before widening this assert.");

namespace at::native {

// ROCm counterpart of ATen/cudnn/Handle.h. ATen/cudnn/*.cpp is not built on
// ROCm, so MHA.cpp's handle acquisition resolves here instead.
TORCH_CUDA_CPP_API cudnnHandle_t getCudnnHandle();

} // namespace at::native

#endif // AT_HIPDNN_ENABLED()
