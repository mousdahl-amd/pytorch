#pragma once

#include <ATen/HipDNNConfig.h>

#if AT_HIPDNN_ENABLED()

// hipDNN publishes HIPDNN_ENABLE_SDPA only through its CMake target, and its
// headers hide the SDPA API without it. Define it here so builds that do not
// consume that target see the same API. Only hipDNN's header-only frontend
// reads it; a hipDNN without SDPA engines declines the graphs at run time.
#ifndef HIPDNN_ENABLE_SDPA
#define HIPDNN_ENABLE_SDPA
#endif

#include <c10/macros/Export.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend.h>
#include <hipdnn_frontend/version_number.h>

namespace cudnn_frontend = hipdnn_frontend::compatibility::cudnn_frontend;

static_assert(
    HIPDNN_FRONTEND_VERSION_MAJOR * 10000 +
            HIPDNN_FRONTEND_VERSION_MINOR * 100 +
            HIPDNN_FRONTEND_VERSION_PATCH >=
        400,
    "hipDNN's frontend is older than 0.4.0, the first version this "
    "integration was validated against.");
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

namespace hipdnn {
// A distinct name from ATen/cudnn's at::native::getCudnnHandle(), which a
// hipified build may also compile.
TORCH_CUDA_CPP_API cudnnHandle_t getCudnnHandle();
} // namespace hipdnn

using hipdnn::getCudnnHandle;

} // namespace at::native

#endif // AT_HIPDNN_ENABLED()
