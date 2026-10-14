#pragma once

#include <ATen/cuda/CUDAConfig.h>

#if AT_HIPDNN_ENABLED()

#include <hipdnn_compatibility/cudnn/cudnn_frontend.h>
#include <c10/macros/Export.h>

namespace cudnn_frontend = hipdnn_frontend::compatibility::cudnn_frontend;

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

TORCH_CUDA_CPP_API cudnnHandle_t getCudnnHandle();

} // namespace at::native

#endif // AT_HIPDNN_ENABLED()
