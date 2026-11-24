"""pytest plugin: prefer one ROCm flash/mem-efficient library for the whole session.

    SDPA_LIB=hipdnn python -m pytest -p sdpa_lib_plugin test/test_transformers.py

Tests that pin a library themselves (the sdpa_backend-parametrized ones) still do, and
the common_utils decorator restores this session-wide choice after them.
"""
import os


def pytest_configure(config):
    lib = os.environ.get("SDPA_LIB")
    if not lib:
        return
    import torch

    torch.backends.cuda.preferred_rocm_fa_library(lib)
    print(f"sdpa_lib_plugin: preferred_rocm_fa_library = {torch.backends.cuda.preferred_rocm_fa_library()}")
