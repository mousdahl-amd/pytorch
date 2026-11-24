#!/usr/bin/env bash
#SBATCH --job-name=pt-hipdnn-sdpa-build
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=08:00:00
# Build this PyTorch tree for $ARCHS against the ROCm nightly Python packages (whose
# hipDNN is the one under test), on a CPU-only node. Submit as described in README.md;
# partition, image and mounts are given on the sbatch command line.
#   1. venv with rocm[libraries,devel,device-<arch>...]==$ROCM_VER;
#   2. report the nightly's hipDNN frontend version and SDPA objects per arch;
#   3. submodules, hipify, and an editable install of $WORK/pytorch into the venv;
#   4. CPU-side check: torch imports, maps hipDNN, and offers the "hipdnn" preference.
set -euo pipefail
WORK=${WORK:-$HOME/pt-hipdnn-sdpa-validation}
source "$WORK/pytorch/tools/hipdnn_sdpa_validation/env.sh"
phase() { echo; echo "=== [$(date -Is)] $1"; }
JOBS=$(python3 -c 'import os; print(len(os.sched_getaffinity(0)))')
echo "host=$(hostname) jobs=$JOBS rocm=$ROCM_VER archs=$ARCHS ref=$(git -C "$PT" rev-parse --short HEAD)"

phase venv
DEVICES=$(echo "$ARCHS" | tr ';' '\n' | sed 's/^/device-/' | paste -sd,)
python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install -q -U pip
"$VENV/bin/python" -m pip install -q --pre --index-url "$ROCM_INDEX" "rocm[libraries,devel,$DEVICES]==$ROCM_VER"
"$VENV/bin/rocm-sdk" init
source "$VENV/bin/activate"
DEVEL="$(rocm-sdk path --root)"
SITE="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "rocm-sdk $(python -m rocm_sdk version)"

phase "hipDNN in the nightly"
find "$DEVEL/include" -name version.h -path '*hipdnn_frontend*' -exec grep -h 'VERSION_STRING' {} +
for a in $(echo "$ARCHS" | tr ';' ' '); do
  n_bwd=$(find "$SITE" -path "*$a*" -iname '*sdpa_bwd*' | wc -l)
  n_fwd=$(find "$SITE" -path "*$a*" -iname '*sdpa_cdna*' | wc -l)
  echo "$a: sdpa_bwd files=$n_bwd sdpa_cdna files=$n_fwd"
done

phase pytorch
cd "$PT"
git log -1 --format='%h %s'
git submodule update -q --init --recursive --jobs 16
python -m pip install -q -r requirements.txt "scikit-build-core>=1.1.1" pytest expecttest hypothesis
python tools/amd_build/build_amd.py > "$WORK/logs/hipify.log" 2>&1
# torch/__init__.py imports this (gitignored) to preload the ROCm wheel libraries, as
# TheRock's build_prod_wheels.py generates it.
cat > torch/_rocm_init.py <<EOF
def initialize():
    import rocm_sdk
    rocm_sdk.initialize_process(
        preload_shortnames=['amd_comgr', 'amd_smi', 'amdhip64', 'hiprtc', 'hipblas', 'hipfft',
            'hiprand', 'hipsparse', 'hipsparselt', 'hipsolver', 'hipblaslt', 'miopen', 'hipdnn',
            'rocm_sysdeps_liblzma', 'rocm-openblas', 'rocprofiler-sdk', 'rocprofiler-sdk-roctx',
            'roctracer64', 'roctx64', 'rccl', 'hipfile'],
        check_version='$(python -m rocm_sdk version)')
EOF
export ROCM_PATH="$DEVEL" ROCM_HOME="$DEVEL"
export PATH="$(rocm-sdk path --bin):$PATH"
export CMAKE_PREFIX_PATH="$(rocm-sdk path --cmake)"
export PKG_CONFIG_PATH="$DEVEL/lib/rocm_sysdeps/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
# CK SDPA and MSLK are not under test and would add hours to a multi-arch build.
export USE_ROCM=1 USE_CUDA=0 PYTORCH_ROCM_ARCH="$ARCHS" BUILD_TEST=0 MAX_JOBS="$JOBS" \
  USE_ROCM_CK_SDPA=0 USE_MSLK=0
python -m pip install --no-build-isolation -v -e . > "$WORK/logs/pytorch-build.log" 2>&1 \
  || { grep -n -E 'error|Error' "$WORK/logs/pytorch-build.log" | tail -30; exit 1; }
grep -E 'Building PyTorch for GPU arch|USE_HIPDNN|hipdnn_frontend VERSION' "$WORK/logs/pytorch-build.log" | sort -u || true

phase "CPU-side check"
cd /tmp
python - <<'EOF'
import torch
print("torch", torch.__version__, "hip", torch.version.hip, "archs", torch.cuda.get_arch_list())
print("ROCm SDPA libraries", list(torch.backends.cuda._ROCmFABackends))
maps = sorted({l.split()[-1].rsplit("/", 1)[-1] for l in open("/proc/self/maps") if "hipdnn" in l})
print("hipDNN libraries mapped:", maps)
assert "hipdnn" in torch.backends.cuda._ROCmFABackends and maps, "this torch is not built with hipDNN"
EOF
echo "=== [$(date -Is)] build done"
