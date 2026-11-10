#!/usr/bin/env bash
# Train one of the paper's policies from the pi0.5 base checkpoint with the released MME-VLA recipe: 30k steps,
# global batch 64 (memory_clock/train.py). Usage: bash train.sh NAME [extra scripts/train.py arguments]
#   clock          FrameSamp-Modul memory with time stamps (stock config)
#   clock-retrain  the same recipe again (the paper's control for pause augmentation, trained on the same machine)
#   noclock        the same without time encoding (configs/perceptual-framesamp-modul-noclock.yaml)
#   pauseaug       clock + pause augmentation of the memory timeline (p 0.8, 1-3 pauses of 20-600 frames)
#   pauseaug-weak  clock + the weaker dose (p 0.3, 1-2 pauses of 20-100 frames)
# The paper trained on 4 (Clock, NoClock) or 2 (pause augmentation) H200 GPUs with FSDP over all of them; one 80 GB+
# GPU also fits batch 64. Environment: DATA (preprocessed RoboMME data, default as in download.sh), NUM_GPUS (default:
# all visible GPUs), CKPT_OUT (default checkpoints/trained), RESUME=1 (continue instead of starting over), ROBOMME_DIR.
# The checkpoint lands in $CKPT_OUT/mme_vla_scratch/<NAME>/29999; evaluate it with run_eval.sh --ckpt <that dir>.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROBOMME_DIR=$(cd "${ROBOMME_DIR:-$REPO/third_party/robomme_policy_learning}" && pwd)
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-$REPO/checkpoints/openpi} ROBOMME_DIR
NAME=${1:?usage: bash train.sh clock|noclock|clock-retrain|pauseaug|pauseaug-weak [args]}; shift
DATA=${DATA:-$ROBOMME_DIR/data/robomme_preprocessed_data}
CKPT_OUT=$(mkdir -p "${CKPT_OUT:-$REPO/checkpoints/trained}" && cd "${CKPT_OUT:-$REPO/checkpoints/trained}" && pwd)
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L | grep -c GPU)}
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] && NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
YAML=perceptual-framesamp-modul.yaml
export PAUSE_AUG=0
case $NAME in
  clock|clock-retrain) ;;
  noclock) YAML=perceptual-framesamp-modul-noclock.yaml;;
  pauseaug) export PAUSE_AUG=1 PAUSE_AUG_P=0.8 PAUSE_AUG_MAX=3 PAUSE_AUG_LEN_MAX=600;;
  pauseaug-weak) export PAUSE_AUG=1 PAUSE_AUG_P=0.3 PAUSE_AUG_MAX=2 PAUSE_AUG_LEN_MAX=100;;
  *) echo "unknown policy $NAME"; exit 1;;
esac
MODE=--overwrite; [ -n "${RESUME:-}" ] && MODE=--resume
cd "$ROBOMME_DIR"
XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.92} .venv/bin/python "$REPO/memory_clock/train.py" \
  mme_vla_scratch --exp-name="$NAME" --batch-size=64 --num-workers=8 --dataset-path="$DATA" --num-train-steps=30000 \
  --model.use_history --model.history_config=$YAML --fsdp-devices="$NUM_GPUS" --checkpoint-base-dir="$CKPT_OUT" \
  $MODE "$@"
