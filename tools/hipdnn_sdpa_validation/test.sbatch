#!/usr/bin/env bash
#SBATCH --job-name=pt-hipdnn-sdpa-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
# Test the hipDNN SDPA arms on one arch, against AOTriton as the baseline. Needs two GPUs
# of one type (give --gres on the sbatch command line, see README.md):
#   GPU 0: smoke_fwd.py, smoke_bwd.py (hipDNN log kept for the engines used), then
#          test_transformers.py and test_varlen_attention.py with "hipdnn" preferred;
#   GPU 1: the same two test files with "aotriton" preferred, concurrently.
# Results go to $WORK/results/<commit>/<arch>/; compare.py summarizes them.
set -uo pipefail
WORK=${WORK:-$HOME/pt-hipdnn-sdpa-validation}
source "$WORK/pytorch/tools/hipdnn_sdpa_validation/env.sh"
source "$VENV/bin/activate"
unset ROCM_PATH HIP_PATH ROCM_HOME
export PYTHONPATH="$TOOLS${PYTHONPATH:+:$PYTHONPATH}"
cd "$PT"
COMMIT=$(git rev-parse --short=11 HEAD)
ARCH=$(HIP_VISIBLE_DEVICES=0 python -c 'import torch; print(torch.cuda.get_device_properties(0).gcnArchName.split(":")[0])')
OUT=$WORK/results/$COMMIT/$ARCH
mkdir -p "$OUT"
{
  echo "date=$(date -Is)"
  echo "pytorch_commit=$(git rev-parse HEAD)"
  echo "rocm=$(python -m rocm_sdk version)"
  echo "hipdnn_frontend=$(find "$(rocm-sdk path --root)/include" -name version.h -path '*hipdnn_frontend*' \
    -exec grep -h 'VERSION_STRING' {} + | grep -oE '"[^"]+"' | tr -d '"')"
  echo "arch=$ARCH"
  echo "gpu=$(HIP_VISIBLE_DEVICES=0 python -c 'import torch; print(torch.cuda.get_device_name(0))')"
  echo "gpus=$(HIP_VISIBLE_DEVICES=0,1 python -c 'import torch; print(torch.cuda.device_count())')"
  echo "host=$(hostname) job=$SLURM_JOB_ID"
} | tee "$OUT/run_info.txt"

suite() {  # suite <lib> <gpu>
  local lib=$1 gpu=$2
  for t in test_transformers test_varlen_attention; do
    HIP_VISIBLE_DEVICES=$gpu SDPA_LIB=$lib timeout 4h python -m pytest "test/$t.py" -k cuda -q \
      -p no:cacheprovider -p sdpa_lib_plugin --junitxml="$OUT/$t.$lib.xml" > "$OUT/$t.$lib.out" 2>&1
    echo "[$lib] $t rc=$? $(tail -1 "$OUT/$t.$lib.out")"
  done
}

(
  HIP_VISIBLE_DEVICES=0 timeout 30m python "$TOOLS/smoke_fwd.py" > "$OUT/smoke_fwd.out" 2>&1
  echo "[hipdnn] smoke_fwd rc=$? $(grep -c '^PASS' "$OUT/smoke_fwd.out") passed, $(tail -1 "$OUT/smoke_fwd.out")"
  LOG=/tmp/hipdnn-$SLURM_JOB_ID.log
  HIP_VISIBLE_DEVICES=0 HIPDNN_LOG_LEVEL=info HIPDNN_LOG_FILE=$LOG \
    timeout 30m python "$TOOLS/smoke_bwd.py" > "$OUT/smoke_bwd.out" 2>&1
  echo "[hipdnn] smoke_bwd rc=$? $(grep -c '^PASS' "$OUT/smoke_bwd.out") passed, $(tail -1 "$OUT/smoke_bwd.out")"
  # Which hipDNN engines and kernels the backward smoke executed.
  grep -oE "engine '[a-z:_0-9]+' selected kernel|Selected engine id -?[0-9]+" "$LOG" \
    | sed -E "s/ selected kernel//" | sort | uniq -c > "$OUT/smoke_engines.txt"
  rm -f "$LOG"
  suite hipdnn 0
) &
suite aotriton 1 &
wait
echo "done $(date -Is) -> $OUT"
