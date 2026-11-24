# hipDNN SDPA validation harness

Builds this tree against a ROCm nightly whose hipDNN is under test, then runs PyTorch's
SDPA tests twice per GPU arch, once with AOTriton and once with hipDNN behind
`FLASH_ATTENTION` / `EFFICIENT_ATTENTION`, and compares the two runs test by test.
Written for a SLURM cluster with pyxis containers; the jobs only need a Linux userland with
Python 3.12, a C++ toolchain, CMake and Ninja, because ROCm itself comes from pip.

This directory is a validation aid for the hipDNN integration branch and is not meant to
be upstreamed.

| File | Purpose |
|---|---|
| `env.sh` | Settings, each overridable from the environment: `WORK`, `ROCM_VER`, `ROCM_INDEX`, `ARCHS` |
| `build.sbatch` | CPU-only job: venv with the ROCm nightly, hipify, editable PyTorch install (about 1 h for three arches) |
| `test.sbatch` | Two-GPU job per arch: smoke tests and the hipDNN suite on GPU 0, the AOTriton suite on GPU 1 |
| `relink.sbatch` | Moves an existing build to a later commit, rebuilding only `torch_hip` |
| `smoke_fwd.py`, `smoke_bwd.py` | hipDNN forward and backward arms against float64 references |
| `sdpa_lib_plugin.py` | pytest plugin: `SDPA_LIB=hipdnn|aotriton` sets the preferred ROCm library for the session |
| `compare.py` | Markdown summary: counts per arch and suite, and every test whose outcome differs |

## Run

```sh
export WORK=$HOME/pt-hipdnn-sdpa-validation      # on a filesystem the compute nodes mount
git clone https://github.com/mousdahl-amd/pytorch.git "$WORK/pytorch"
git -C "$WORK/pytorch" checkout <commit>          # a commit that carries this directory
T=$WORK/pytorch/tools/hipdnn_sdpa_validation
mkdir -p "$WORK/logs"

B=$(sbatch --parsable <cpu-node options> --output="$WORK/logs/build-%j.log" "$T/build.sbatch")
for gres in <gpu-type-a>:2 <gpu-type-b>:2; do
  sbatch --dependency=afterok:$B <gpu-node options> --gres=gpu:$gres \
    --output="$WORK/logs/test-%j.log" "$T/test.sbatch"
done
# when all test jobs finish:
source "$WORK/venv/bin/activate"
python "$T/compare.py" "$WORK/results/$(git -C "$WORK/pytorch" rev-parse --short=11 HEAD)"
```

On Alola (AUSTIN site), the options were:

| Job | Options |
|---|---|
| build | `--partition=am --constraint='CPUONLY&AUSTIN' --container-image=/cluster/images/hipdnn/hipdnn_latest_gfx950.sqsh --container-mounts="$HOME:$HOME"` |
| test | `--partition=miopen,defq --account=miopen --constraint=AUSTIN` with the same image and mounts; `--gres` of `gpu:gfx90a-mi250:2`, `gpu:gfx942-mi325x:2`, `gpu:gfx950-mi355x:2` |

Results land in `$WORK/results/<commit>/<arch>/`: `run_info.txt` (commit, ROCm and hipDNN
versions, GPU, date), JUnit XML and output of each suite and library, the smoke-test
output, and `smoke_engines.txt`.

To test a later commit of the same branch without a full rebuild, when the change is
confined to sources only `torch_hip` compiles (the hipDNN SDPA arms): `sbatch <cpu-node
options> "$T/relink.sbatch" <commit>`, then the test jobs again.
