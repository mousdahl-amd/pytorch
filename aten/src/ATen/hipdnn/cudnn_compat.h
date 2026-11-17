#pragma once

#include <ATen/HipDNNConfig.h>

#if AT_HIPDNN_CUDNN_COMPAT_ENABLED()

// Defines HIPDNN_ENABLE_SDPA and checks the frontend version before the shim
// pulls in the frontend.
#include <ATen/hipdnn/hipdnn_frontend.h>
#include <ATen/hipdnn/Handle.h>

#include <hipdnn_compatibility/cudnn/cudnn_frontend.h>

#include <type_traits>

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
static_assert(
    std::is_same_v<cudnnHandle_t, hipdnnHandle_t>,
    "The shim's cudnnHandle_t must be hipDNN's handle for the shared pool.");

namespace at::native {

namespace hipdnn {
// A distinct name from ATen/cudnn's at::native::getCudnnHandle(), which a
// hipified build may also compile. Same pool as getHipdnnHandle().
inline cudnnHandle_t getCudnnHandle() {
  return getHipdnnHandle();
}
} // namespace hipdnn

using hipdnn::getCudnnHandle;

} // namespace at::native

#endif // AT_HIPDNN_CUDNN_COMPAT_ENABLED()
