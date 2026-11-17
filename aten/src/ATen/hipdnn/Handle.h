#pragma once

#include <ATen/HipDNNConfig.h>

#if AT_HIPDNN_ENABLED()

#include <c10/macros/Export.h>
#include <hipdnn_backend.h>

namespace at::native {

// One hipDNN handle per device and thread, bound to the current HIP stream.
// Shared by every hipDNN user in ATen; the cuDNN-compatibility layer's
// cudnnHandle_t is the same type.
TORCH_CUDA_CPP_API hipdnnHandle_t getHipdnnHandle();

} // namespace at::native

#endif // AT_HIPDNN_ENABLED()
