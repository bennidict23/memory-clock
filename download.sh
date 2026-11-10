#!/usr/bin/env bash
# Download checkpoints (and, for training, data). Usage: bash download.sh [GROUP ...]   (default: main)
#   main      the released FrameSamp-Modul policy (Figs 2, 4, 8b) and the paper's trained policies (Figs 6, 8)
#   designs   the other released RoboMME policies of Table 2 (FrameSamp-Expert / -Context, TokenDrop-Modul, TTT,
#             pi0.5 without memory, MemER-style keyframes, text sub-goals) and the VLM sub-goal predictors
#   train     pi0.5 base weights and the preprocessed RoboMME training data (~1 TB unpacked)
# Released RoboMME checkpoints go where upstream expects them: $ROBOMME_DIR/runs/ckpts/<config>/<model>/79999.
# The paper's trained policies (huggingface.co/benni23/memory-clock) go to checkpoints/<name>/29999 (name: clock, noclock, clock-retrain,
# pauseaug, pauseaug-weak).
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROBOMME_DIR=$(cd "${ROBOMME_DIR:-$REPO/third_party/robomme_policy_learning}" && pwd)
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-$REPO/checkpoints/openpi}
HF=$ROBOMME_DIR/.venv-robomme/bin/hf
PY=$ROBOMME_DIR/.venv/bin/python
CK=$ROBOMME_DIR/runs/ckpts
cd "$ROBOMME_DIR"

fetch() {  # HF repo, path of 79999.zip in it, target model dir  ->  <dir>/79999/{params,assets} + history_config.txt
  local repo=$1 zip=$2 dst=$3
  [ -d "$dst/79999/params" ] && { echo "[skip] $dst"; return; }
  mkdir -p "$dst"
  local hc; hc=$(dirname "$zip")/history_config.txt; hc=${hc#./}
  $HF download "$repo" "$zip" --local-dir "$dst/_dl"
  $HF download "$repo" "$hc" --local-dir "$dst/_dl" 2>/dev/null && mv "$dst/_dl/$hc" "$dst/history_config.txt" || true
  mv "$dst/_dl/$zip" "$dst/79999.zip" && rm -rf "$dst/_dl"
  $PY scripts/unzip_ckpt.py "$dst"              # upstream helper: 79999.zip -> 79999/
  [ -d "$dst/79999/params" ] && rm -f "$dst/79999.zip"
}

openpi_assets() {  # the PaliGemma tokenizer (needed by every policy) and, with "base", the pi0.5 base weights
  $PY - "$@" <<'EOF'
import sys
from openpi.shared import download
print(download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"}))
if "base" in sys.argv[1:]:
    print(download.maybe_download("gs://openpi-assets/checkpoints/pi05_base"))
EOF
}

main() {
  fetch Yinpei/perceptual-framesamp-modul 79999.zip "$CK/mme_vla_suite/perceptual-framesamp-modul"
  # training norm stats = the released model's own (runs/assets/<train config>/robomme/norm_stats.json)
  mkdir -p runs/assets/mme_vla_scratch/robomme
  cp "$CK/mme_vla_suite/perceptual-framesamp-modul/79999/assets/robomme/norm_stats.json" runs/assets/mme_vla_scratch/robomme/
  openpi_assets
  # The paper's policies trained from pi0.5 base (30k steps), hosted at huggingface.co/benni23/memory-clock
  for name in clock noclock clock-retrain pauseaug pauseaug-weak; do
    dst=$REPO/checkpoints/$name
    [ -d "$dst/29999/params" ] && { echo "[skip] $dst"; continue; }
    $HF download benni23/memory-clock --include "$name/*" --local-dir "$REPO/checkpoints"
    [ -d "$dst/29999/params" ] || echo "[todo] $name is not on Hugging Face yet; train it with: bash train.sh $name"
  done
}

designs() {
  for m in perceptual-framesamp-expert perceptual-framesamp-context perceptual-tokendrop-modul recurrent-ttt-context \
           symbolic-grounded-subgoal symbolic-simple-subgoal; do
    fetch Yinpei/mme_vla_suite "$m/79999.zip" "$CK/mme_vla_suite/$m"
  done
  fetch Yinpei/pi05_baseline pi05_baseline/79999.zip "$CK/pi05_baseline/pi05_baseline"
  if [ ! -d "$CK/vlm_subgoal_predictor/memer" ]; then
    $HF download Yinpei/vlm_subgoal_predictor --local-dir "$CK/vlm_subgoal_predictor"
    find "$CK/vlm_subgoal_predictor" -name "checkpoint-*.zip" | while read -r z; do
      (cd "$(dirname "$z")" && unzip -q -o "$(basename "$z")" && rm -f "$(basename "$z")"); done
  fi
  # base model of the sub-goal predictors (run_eval.sh points the client at this local copy)
  $HF download Qwen/Qwen3-VL-4B-Instruct --local-dir "$CK/Qwen3-VL-4B-Instruct"
}

train() {
  openpi_assets base
  DATA=$ROBOMME_DIR/data/robomme_preprocessed_data
  $HF download Yinpei/robomme_preprocessed_data --repo-type dataset --local-dir "$DATA" \
    --include "data/*" "features/*" "meta/*"    # memer/ and qwenvl/ (VLM sub-goal data) are not needed
  .venv/bin/python scripts/unzip_data.py "$DATA"
}

for g in "${@:-main}"; do
  case $g in main|designs|train) $g;; *) echo "unknown group $g (main | designs | train)"; exit 1;; esac
done
echo "done"
