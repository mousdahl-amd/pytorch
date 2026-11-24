#!/usr/bin/env bash
#SBATCH --job-name=pt-hipdnn-sdpa-relink
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=01:00:00
# Move an existing build to another commit and rebuild only torch_hip (about 12 minutes
# instead of an hour). Valid only when the commits in between change sources that hipify
# leaves alone and that only torch_hip compiles, such as
# aten/src/ATen/native/transformers/hip/flash_attn/hipdnn/; otherwise run build.sbatch
# on a fresh clone. The installed library is swapped in with a rename, so running
# processes keep the one they mapped.
#   sbatch [options] relink.sbatch <commit>     (fetched from the clone's origin)
set -euo pipefail
WORK=${WORK:-$HOME/pt-hipdnn-sdpa-validation}
source "$WORK/pytorch/tools/hipdnn_sdpa_validation/env.sh"
REF=${1:?usage: relink.sbatch <commit>}
source "$VENV/bin/activate"
DEVEL="$(rocm-sdk path --root)"
git -C "$PT" fetch -q origin
git -C "$PT" cat-file -e "$REF:tools/hipdnn_sdpa_validation/env.sh" \
  || { echo "$REF does not carry tools/hipdnn_sdpa_validation; test.sbatch needs it"; exit 1; }
git -C "$PT" checkout -q --detach "$REF"
git -C "$PT" log -1 --format='%h %s'
export ROCM_PATH="$DEVEL" ROCM_HOME="$DEVEL" PATH="$(rocm-sdk path --bin):$PATH"
cd "$PT"
cmake --build build --target torch_hip 2>&1 | tail -4
LIB=$(find build/install -name libtorch_hip.so -path '*torch/lib/*' | head -1)
cp build/lib/libtorch_hip.so "$LIB.new" && mv -f "$LIB.new" "$LIB"
echo "installed libtorch_hip.so at $(git rev-parse --short=11 HEAD)"
