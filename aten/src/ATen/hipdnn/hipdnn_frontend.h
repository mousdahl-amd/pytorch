#pragma once

// Include hipDNN's native frontend through this header, never directly, so the
// SDPA API is visible and the version floor is checked in every translation unit.

#include <ATen/HipDNNConfig.h>

#if AT_HIPDNN_ENABLED()

// hipDNN publishes HIPDNN_ENABLE_SDPA only through its CMake target, and its
// headers hide the SDPA API without it. Define it here so builds that do not
// consume that target see the same API. Only hipDNN's header-only frontend
// reads it; a hipDNN without SDPA engines declines the graphs at run time.
#ifndef HIPDNN_ENABLE_SDPA
#define HIPDNN_ENABLE_SDPA
#endif

#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/version.h>

static_assert(
    HIPDNN_FRONTEND_VERSION_MAJOR * 10000 +
            HIPDNN_FRONTEND_VERSION_MINOR * 100 +
            HIPDNN_FRONTEND_VERSION_PATCH >=
        400,
    "hipDNN's frontend is older than 0.4.0, the first version this "
    "integration was validated against.");

#endif // AT_HIPDNN_ENABLED()
