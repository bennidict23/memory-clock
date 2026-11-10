#!/usr/bin/env bash
# Install the upstream RoboMME code at the commits used in the paper and its two Python environments (built with uv,
# following the upstream instructions), then add the NoClock memory config.
#   third_party/robomme_policy_learning                RoboMME/robomme_policy_learning @ ecf086c
#   third_party/robomme_policy_learning/third_party/robomme_benchmark   RoboMME/robomme_benchmark @ 856bc3a
#   .venv          policy server and training (JAX; upstream README: uv sync)
#   .venv-robomme  simulator and evaluation client (Python 3.11; upstream examples/robomme/readme.md)
# The upstream code is not modified; only the NoClock config file is added to its config directory. Set ROBOMME_DIR to
# install it elsewhere (and export it for the other scripts).
# Requirements: git, uv (https://docs.astral.sh/uv/), a CUDA 12 driver, and Vulkan for rendering (or Mesa's lavapipe,
# see run_eval.sh --render lavapipe). git-lfs is not required (large files are skipped with GIT_LFS_SKIP_SMUDGE=1).
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROBOMME_DIR=${ROBOMME_DIR:-$REPO/third_party/robomme_policy_learning}
POLICY_COMMIT=ecf086c
BENCH_COMMIT=856bc3a
MANISKILL="mani_skill @ git+https://github.com/YinpeiDai/ManiSkill.git@07be6fbc66350ddca200abfb0a11b692f078f7fd"
command -v uv > /dev/null || { echo "install uv first: https://docs.astral.sh/uv/getting-started/installation/"; exit 1; }

if [ ! -d "$ROBOMME_DIR/.git" ]; then
  git clone https://github.com/RoboMME/robomme_policy_learning.git "$ROBOMME_DIR"
fi
cd "$ROBOMME_DIR"
git checkout -q $POLICY_COMMIT
# the submodule is registered with an SSH URL; fetch it over HTTPS
git -c url."https://github.com/".insteadOf="git@github.com:" submodule update --init
git -C third_party/robomme_benchmark checkout -q $BENCH_COMMIT

echo "== policy environment (.venv)"
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

echo "== simulator environment (.venv-robomme)"
[ -x .venv-robomme/bin/python ] || uv venv .venv-robomme --python 3.11
# upstream requirements, with the ManiSkill fork pinned to the commit used in the paper (upstream: branch "dev")
grep -v "ManiSkill.git" examples/robomme/requirements.txt > .venv-robomme/requirements.pinned.txt
uv pip install --python .venv-robomme/bin/python -r .venv-robomme/requirements.pinned.txt "$MANISKILL"
uv pip install --python .venv-robomme/bin/python -e third_party/robomme_benchmark -e packages/openpi-client
# flash-attn (optional upstream) is not needed: the client runs the VLM sub-goal predictors with sdpa attention

echo "== NoClock memory config"
cp "$REPO/configs/perceptual-framesamp-modul-noclock.yaml" src/mme_vla_suite/models/config/robomme/

echo "== versions used in the paper: jax/jaxlib 0.5.3, flax 0.10.2, torch 2.7.1 (.venv); mani_skill 3.0.0b21, sapien 3.0.3, torch 2.9.1 (.venv-robomme)"
.venv/bin/python -c "import jax, flax, torch; print('.venv:', 'jax', jax.__version__, 'flax', flax.__version__, 'torch', torch.__version__)"
.venv-robomme/bin/python -c "import mani_skill, sapien, torch; print('.venv-robomme:', 'mani_skill', mani_skill.__version__, 'sapien', sapien.__version__, 'torch', torch.__version__)"
echo "done. Next: bash download.sh"
