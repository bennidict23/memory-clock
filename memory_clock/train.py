"""Train a memory policy from the pi0.5 base checkpoint with RoboMME's released MME-VLA recipe.

Registers a TrainConfig "mme_vla_scratch" = upstream "mme_vla_suite" (model, data, optimiser, learning-rate schedule,
EMA, seed and batch size unchanged), except: weights from INIT_PARAMS (default: the pi0.5 base params under
$OPENPI_DATA_HOME, as in the upstream config), checkpoints under <checkpoint_base_dir>/mme_vla_scratch/<exp_name>,
wandb off, save cadence. The memory design is chosen by --model.history_config, e.g. perceptual-framesamp-modul.yaml
(the time-stamped memory, "Clock") or perceptual-framesamp-modul-noclock.yaml (no time encoding, "NoClock").
PAUSE_AUG=1 trains with pause augmentation of the memory timeline (pause_aug.py; dose PAUSE_AUG_P, PAUSE_AUG_MAX,
PAUSE_AUG_LEN_MAX). NOSAVE=1 never writes checkpoints (smoke tests).
Normalisation statistics must exist at runs/assets/mme_vla_scratch/robomme/norm_stats.json (download.sh copies the
released ones). Run from the robomme_policy_learning checkout; all other arguments go to its scripts/train.py, e.g.
  python train.py mme_vla_scratch --exp-name=clock --model.use_history \
      --model.history_config=perceptual-framesamp-modul.yaml --dataset-path=<data> --num-train-steps=30000
"""
import dataclasses
import os
import runpy
import sys

from mme_vla_suite.training import config as _config
import openpi.training.weight_loaders as _weight_loaders

HERE = os.path.dirname(os.path.abspath(__file__))
ROBOMME_DIR = os.environ.get("ROBOMME_DIR", os.path.join(HERE, "..", "third_party", "robomme_policy_learning"))
INIT_PARAMS = os.environ.get("INIT_PARAMS", os.path.join(os.environ.get("OPENPI_DATA_HOME", ""),
                                                         "openpi-assets/checkpoints/pi05_base/params"))

_config._CONFIGS_DICT["mme_vla_scratch"] = dataclasses.replace(
    _config._CONFIGS_DICT["mme_vla_suite"],
    name="mme_vla_scratch",
    weight_loader=_weight_loaders.CheckpointWeightLoader(INIT_PARAMS),
    checkpoint_base_dir="runs/train_ckpts",
    wandb_enabled=False,
    save_interval=5_000,
    keep_period=10_000,
)

PAUSE_AUG = os.environ.get("PAUSE_AUG", "0") == "1"
if PAUSE_AUG:
    sys.path.insert(0, HERE)      # spawn workers inherit sys.path and import pause_aug from here
    import pause_aug
    pause_aug.install()
if os.environ.get("NOSAVE") == "1":
    import openpi.training.checkpoints as _ckpt
    _ckpt.save_state = lambda *a, **k: print("[train] NOSAVE: checkpoint not saved", flush=True)
print(f"[train] init {INIT_PARAMS}, pause augmentation {'on' if PAUSE_AUG else 'off'}", flush=True)

sys.argv[0] = "train.py"
runpy.run_path(os.path.join(ROBOMME_DIR, "scripts", "train.py"), run_name="__main__")
