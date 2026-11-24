# Settings shared by build.sbatch, test.sbatch and relink.sbatch (sourced, not run).
# Each can be overridden in the submitting shell: sbatch passes its environment on.
#
# WORK must be on a filesystem the compute nodes mount, and the tree must already be
# cloned at $WORK/pytorch (see README.md). Everything a run creates lives under $WORK,
# so cleanup is `rm -rf "$WORK"`.

WORK=${WORK:-$HOME/pt-hipdnn-sdpa-validation}
PT=$WORK/pytorch
TOOLS=$PT/tools/hipdnn_sdpa_validation
VENV=$WORK/venv

# ROCm nightly Python packages; their hipDNN is the one under test.
ROCM_INDEX=${ROCM_INDEX:-https://nightly.repo.amd.com/rocm/core/whl-next/}
ROCM_VER=${ROCM_VER:-10.3.0a20261123}
ARCHS=${ARCHS:-gfx90a;gfx942;gfx950}

export AMD_COMGR_CACHE_DIR=$WORK/comgr-cache
mkdir -p "$WORK/logs" "$WORK/results" "$AMD_COMGR_CACHE_DIR"
