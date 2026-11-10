"""Optional software rendering (Mesa lavapipe, a CPU Vulkan driver) for ManiSkill, for machines where GPU Vulkan
rendering hangs or is unavailable. Enabled by run_eval.sh --render lavapipe (RENDER=lavapipe for eval_client.py).

Import before any env is created, with VK_ICD_FILENAMES pointing at the lavapipe ICD (run_eval.sh sets it), and pass
render_backend="lvp" to gym.make (eval_client.py does). ManiSkill cannot parse a device string such as
"pci:0000:00:00.0" itself (it splits on ':'), so the BackendInfo is built here and marked "sapien_cpu" so that cameras
are read back on the CPU path. LVP_DEVICE overrides the device string.
"""
import os

import sapien
import mani_skill.envs.sapien_env as _senv
import mani_skill.envs.utils.system.backend as _backend

LVP_DEVICE = os.environ.get("LVP_DEVICE", "pci:0000:00:00.0")
_orig_parse = _backend.parse_sim_and_render_backend


def _parse(sim_backend, render_backend):
    if render_backend != "lvp":
        return _orig_parse(sim_backend, render_backend)
    info = _orig_parse(sim_backend, "none")
    info.render_device = sapien.Device(LVP_DEVICE)
    info.render_backend = "sapien_cpu"
    return info


_backend.parse_sim_and_render_backend = _parse
_senv.parse_sim_and_render_backend = _parse
