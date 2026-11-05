#pragma once

// AT_HIPDNN_ENABLED(): whether SDPBackend::CUDNN_ATTENTION is served by
// hipDNN's cuDNN-compatible frontend on this ROCm build.
//
// Decided from ROCM_VERSION and the presence of hipDNN's headers, not from a
// build-system-generated header, so builds that do not use CMake agree with it.
// CMake builds define AT_HIPDNN_DISABLED when USE_HIPDNN is off or hipDNN was
// not found.
#if defined(USE_ROCM) && defined(ROCM_VERSION) && ROCM_VERSION >= 100300 && \
    !defined(AT_HIPDNN_DISABLED)
#if __has_include(<hipdnn_compatibility/cudnn/cudnn_frontend.h>)
#define AT_HIPDNN_ENABLED() 1
#endif
#endif

#ifndef AT_HIPDNN_ENABLED
#define AT_HIPDNN_ENABLED() 0
#endif
