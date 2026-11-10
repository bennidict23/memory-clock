"""RoboMME evaluation client with injected null events and an optional event index.

This wraps RoboMME's own evaluation script (examples/robomme/eval.py of robomme_policy_learning) at runtime; the
upstream file is not modified. EpisodeEvaluator.eval_each_episode is replaced by a copy that injects null events, and
pack_buffer is extended so that every add_buffer message tells the policy server which of its frames are null.
Normally launched by run_eval.sh, which sets the environment variables below and starts the policy server.

Null events (EVENT) are injected at the first action-chunk boundary after the policy has executed each of the policy
steps in EVENT_AT. Event steps do not count toward the policy's step budget (the env's own step limit is raised by the
total event length, so events never cause a timeout).
  pause            the robot holds its last commanded action for EVENT_LEN env steps. Nothing in the world changes.
  fidget           the wrist (joint 7) rolls back and forth around its held target (FIDGET_AMP rad, whole periods of
                   FIDGET_PERIOD steps), then holds still for the remaining steps (at least 8): a moving null event
                   with the timing and frame count of a pause.
  airgrasp         RoboMME's own failed-grasp motion (subgoal_planner_func.solve_pickup_fail, mode "z") is run with the
                   ManiSkill motion planner: the arm lifts to z >= 0.15, moves above a cube resting on the table
                   (the target cube if the task has one, otherwise the nearest), goes down to just above it, closes
                   the gripper on air, reopens and goes back up. Only injected while no cube is held (otherwise
                   retried at the next chunk boundary).
  airgrasp-return  the same, then the arm returns to its pre-event pose and gripper state.
Every fidget / air-grasp logs whether it left the world unchanged (no cube moved by more than 1 cm, none deleted, no
button pressed): "ok" or "INVALID".

EVENT_MEMORY:
  visible  event frames enter the policy's memory like any other frame; they are flagged in "null_flags" so that
           the server's memory controls (policy_server.py) can find them. The stock server ignores the flags.
  hidden   event frames are recorded to video but never sent to the policy: its memory looks as if nothing happened.

INDEX=event: a test-time event index (loop erasure). Instead of flagging injected frames, a detector decides from what
the robot can observe which execution frames are null, and the server (MEM_MODE=event_index) removes them from memory
and from the clock. A stretch of frames is erased once it closes a loop: the robot comes back to (about) the
configuration of a kept frame after having left it, the scene outside the robot's silhouette is unchanged, and no
grasp, release or load change happened in between. The loop is contracted to one frame (its apex, the kept frame
farthest from the frame returned to). Holding still is erased after a grace period of LOOP_GRACE frames. Contact
events come from what a real arm reports: the gripper width (grip / release) and the net external force on the robot
(load on / off). Robot silhouette and contact forces are read from the simulator.

Other environment variables: SPLIT (test | val), MAX_EPISODES (cap per task), SUBGOAL (sub-goal predictor of the
symbolic baselines, see run_eval.sh), VLM_DIR (local Qwen3-VL-4B-Instruct for those predictors), RENDER=lavapipe
(CPU Vulkan rendering, see render_lavapipe.py), ROBOMME_DIR (the robomme_policy_learning checkout).
Command-line arguments go to eval.py unchanged, e.g. --args.port=8000 --args.only-tasks=PickXtimes.
"""
import importlib
import importlib.util
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROBOMME_DIR = os.environ.get("ROBOMME_DIR", os.path.join(HERE, "..", "third_party", "robomme_policy_learning"))
EVAL_DIR = os.path.join(ROBOMME_DIR, "examples", "robomme")
sys.path.insert(0, HERE)
sys.path.insert(0, EVAL_DIR)

if os.environ.get("RENDER", "default") == "lavapipe":
    import render_lavapipe  # noqa: F401  (must be imported before any env is created)
import gymnasium as gym
import numpy as np

if os.environ.get("RENDER", "default") == "lavapipe":
    _orig_make = gym.make
    gym.make = lambda *a, **k: _orig_make(*a, **{"render_backend": "lvp", **k})

EVENT = os.environ.get("EVENT", "none")
assert EVENT in ("none", "pause", "fidget", "airgrasp", "airgrasp-return"), EVENT
EVENT_LEN = int(os.environ.get("EVENT_LEN", "100"))          # pause / fidget length in env steps
EVENT_AT = [int(x) for x in os.environ.get("EVENT_AT", "48,160").split(",") if x.strip()]
EVENT_MEMORY = os.environ.get("EVENT_MEMORY", "visible")
assert EVENT_MEMORY in ("visible", "hidden"), EVENT_MEMORY
VISIBLE = EVENT_MEMORY == "visible"
AIRGRASP = EVENT.startswith("airgrasp")
AIRGRASP_RETURN = EVENT == "airgrasp-return"
AIRGRASP_BUDGET = 500 if AIRGRASP_RETURN else 400            # env steps reserved per air-grasp (motion ~100-250)
FIDGET_AMP = float(os.environ.get("FIDGET_AMP", "0.3"))
FIDGET_PERIOD = int(os.environ.get("FIDGET_PERIOD", "23"))
if EVENT == "none":
    EVENT_AT = []
TOTAL_EVENT = (AIRGRASP_BUDGET if AIRGRASP else EVENT_LEN) * len(EVENT_AT)
SPLIT = os.environ.get("SPLIT", "test")
EVENT_INDEX = os.environ.get("INDEX", "time") == "event"

import env_runner as _er

_orig_init = _er.EnvRunner.__init__


def _init(self, env_id, video_save_dir, max_steps=1300):
    # the env truncates on its own step counter, which also counts event steps
    _orig_init(self, env_id, video_save_dir, max_steps=max_steps + TOTAL_EVENT)


_er.EnvRunner.__init__ = _init
if SPLIT != "test":           # the benchmark's val episodes: other env seeds, same difficulty mix as test
    _orig_builder = _er.BenchmarkEnvBuilder
    _er.BenchmarkEnvBuilder = lambda **kw: _orig_builder(**{**kw, "dataset": SPLIT})
if os.environ.get("MAX_EPISODES"):
    _cap = int(os.environ["MAX_EPISODES"])
    _er.EnvRunner.num_episodes = property(lambda self: min(_cap, self.env_builder.get_episode_num()))

# ---- event index: detector parameters (fixed across all RoboMME tasks) --------------------------------------------
DETECT_PIX = 50        # scene changed: >= this many pixels outside the dilated robot silhouettes differ ...
DETECT_DPIX = 16       # ... by more than this (max over RGB channels, 0-255) from the last frame that changed the scene
DETECT_DIL = 3         # silhouette dilation (pixels)
LOOP_TOL = 0.02        # 1 distance unit: max |d qpos| of the 7 arm joints (rad) ...
LOOP_TOL_GRIP = 0.004  # ... or of the 2 finger joints (m)
LOOP_EXIT = 2.0        # a return counts only after the robot went more than this many units away
LOOP_STILL = 0.05      # holding still: within this many units of the frame the stillness started at
LOOP_SCENE_PIX = 5     # a loop / stillness also needs < this many pixels changed outside the robot vs that frame
LOOP_GRACE = 5         # the first frames of a stillness are kept (a short natural stop is real time)
NP_W_EMPTY, NP_W_OPEN, NP_W_SETTLE, NP_S = 0.004, 0.004, 0.0005, 3   # grip from finger width (m), settled over NP_S
NP_F_ON, NP_F_OFF = 0.02, 0.01   # load on / off thresholds on |net external force| (sim units)
CONTACT_N = 3                    # debounce: a load switch needs CONTACT_N frames in a row

_le = {}   # loop-erasure state of the current episode (reset in eval_each_episode)

if EVENT_INDEX:
    import scipy.ndimage as _ndi
    _dw = importlib.import_module("robomme.env_record_wrapper.DemonstrationWrapper")   # the module, not the class
    _orig_aug = _dw.DemonstrationWrapper._augment_obs_and_info

    def _aug_detect(self, obs, info, action):
        """After every env step: did the scene outside the robot change, and did a grip or load switch happen."""
        out = _orig_aug(self, obs, info, action)
        try:
            u = self.unwrapped
            cam = obs["sensor_data"]["base_camera"]
            seg = np.asarray(cam["segmentation"][0, ..., 0].cpu().numpy())
            rgb = np.asarray(cam["rgb"][0].cpu().numpy(), dtype=np.int16)
            if getattr(u, "_robot_ids", None) is None:
                u._robot_ids = np.array(sorted({int(l.per_scene_id[0]) for l in u.agent.robot.links}))
                u._det_quiet, u._det_prev = None, None
            robot = np.isin(seg, u._robot_ids)
            if u._det_prev is not None:
                argb, arob = u._det_anchor            # the last frame that changed the scene
                m = _ndi.binary_dilation(robot | arob, iterations=DETECT_DIL)
                n_changed = int(((np.abs(rgb - argb).max(axis=-1) > DETECT_DPIX) & ~m).sum())
                u._det_quiet = n_changed < DETECT_PIX
                if not u._det_quiet:
                    u._det_anchor = (rgb, robot)
            else:
                u._det_anchor = (rgb, robot)
            u._det_prev = (rgb, robot)
            _track_contacts(u)
        except Exception as e:                    # the detector must never break an episode
            if not getattr(self, "_det_warned", False):
                print(f"[event_index] detector disabled: {type(e).__name__}: {e}")
                self._det_warned = True
        return out

    _dw.DemonstrationWrapper._augment_obs_and_info = _aug_detect


def _track_contacts(u):
    """u._det_cchange = a grip or load switch (debounced) happened with the frame just produced.
    Grip: the gripper is "holding" while its width has settled strictly between closed-on-nothing and open; a switch of
    that state is a grasp or a release (an air grasp closes to empty and never holds). Load: the net external force on
    the robot (what a wrist F/T sensor or a joint-torque estimate measures; here the sum over the simulator's contacts
    between robot links and anything else, so grip forces cancel); a switch is a lift-off, set-down, press or push."""
    q = np.asarray(u.agent.robot.get_qpos()[0].cpu().numpy(), np.float64)
    w = float(q[-2] + q[-1])
    if getattr(u, "_robot_names", None) is None:
        u._robot_names = {c.entity.name for l in u.agent.robot.links for c in l._objs}
        u._np_n, u._np_w, u._np_wmax, u._np_hold, u._np_load, u._np_run = 0, [], 0.0, False, False, 0
    F = np.zeros(3)
    for c in u.scene.get_contacts():
        r = [x.entity.name in u._robot_names for x in c.bodies]
        if r[0] != r[1]:
            F += (1.0 if r[0] else -1.0) * sum((np.asarray(p.impulse, np.float64) for p in c.points), np.zeros(3))
    u._np_w = (u._np_w + [w])[-NP_S:]
    u._np_wmax = max(u._np_wmax, w)
    ch = False
    if len(u._np_w) == NP_S and max(u._np_w) - min(u._np_w) < NP_W_SETTLE:
        hold = NP_W_EMPTY < w < u._np_wmax - NP_W_OPEN
        if hold != u._np_hold and u._np_n:
            ch = True
        u._np_hold = hold
    L = float(np.linalg.norm(F))
    want = (L < NP_F_OFF) if u._np_load else (L > NP_F_ON)
    u._np_run = u._np_run + 1 if want else 0
    if u._np_run >= CONTACT_N:
        u._np_load, u._np_run = not u._np_load, 0
        ch = ch or u._np_n > CONTACT_N
    u._det_cchange = ch
    u._np_n += 1


def _flag_frame(env_runner):
    """Event index: add the frame just produced (and just added to the policy's buffer) to the loop-erased path."""
    if not EVENT_INDEX:
        return
    u = env_runner.env.unwrapped
    cc = bool(getattr(u, "_det_cchange", False))
    _le["cc"] += cc
    quiet = bool(getattr(u, "_det_quiet", None)) and not cc     # a contact change is an event too
    cur = getattr(u, "_det_prev", None)                       # (rgb, robot mask) of the frame just produced
    _loop_erase(np.asarray(u.agent.robot.get_qpos()[0].cpu().numpy(), np.float64), quiet,
                None if cur is None else (np.asarray(cur[0], np.uint8), np.asarray(cur[1], bool)))


def _scene_px(a, b):
    """Pixels outside both (dilated) robot silhouettes that differ by > DETECT_DPIX between frames a and b."""
    m = _ndi.binary_dilation(a[1] | b[1], iterations=DETECT_DIL)
    return int(((np.abs(a[0].astype(np.int16) - b[0].astype(np.int16)).max(axis=-1) > DETECT_DPIX) & ~m).sum())


def _loop_erase(q, quiet, img):
    """Add execution frame e (robot configuration q, img = (rgb, robot mask)) to the loop-erased path _le["kept"].
    Only frames since the last scene / contact change can be returned to; erasing a loop that closes at the newest
    frame removes a suffix of what the server stores, so a pause, a fidget or an out-and-back air-grasp leaves one
    frame in memory, while robot-only motion that goes somewhere new is kept."""
    e = len(_le["q"])
    _le["q"].append(q)
    kept, imgs = _le["kept"], _le["img"]
    if not quiet:                             # the scene changed: no loop can reach back past this frame
        _le["last_change"] = e
        _le["ref"], _le["run"] = -1, 0
        kept.append(e)
        imgs.clear()
        imgs[e] = img
        return
    unit = np.full(len(q), LOOP_TOL)
    unit[-2:] = LOOP_TOL_GRIP                 # Panda: 7 arm joints, then the 2 finger joints
    since = [k for k in kept if k >= _le["last_change"]]
    if not since:
        kept.append(e)
        imgs[e] = img
        return
    same = lambda k: img is not None and imgs.get(k) is not None and _scene_px(img, imgs[k]) < LOOP_SCENE_PIX
    Q = np.stack([_le["q"][k] for k in since])
    d = (np.abs(Q - q) / unit).max(axis=1)    # distance of each kept frame since the last change to the new frame
    r = _le["ref"] if _le["ref"] in imgs else since[-1]   # the frame the current stillness started at
    if (np.abs(_le["q"][r] - q) / unit).max() < LOOP_STILL and same(r):   # holding still since r, nothing else moved
        _le["ref"] = r
        _le["run"] += 1
        if _le["run"] > LOOP_GRACE:
            _le["still"] += 1
            _le["erased"] += 1
            return
        kept.append(e)                        # within the grace period: a short stop is kept as it is
        imgs[e] = img
        return
    _le["ref"], _le["run"] = -1, 0
    for j in np.flatnonzero(d < 1.0):         # earliest kept frame the robot came back to after leaving it
        if j + 1 < len(since) and (np.abs(Q[j + 1:] - Q[j]) / unit).max() > LOOP_EXIT and same(since[j]):
            n_off = len(since) - 1 - j        # kept frames after since[j]: the loop, contracted to its apex
            off = kept[len(kept) - n_off:]
            apex = off[int(np.argmax((np.abs(Q[j + 1:] - Q[j]) / unit).max(axis=1)))]
            _le["erased"] += n_off            # the loop's frames but the apex, plus the new frame
            _le["loops"] += 1
            for k in off + [e]:
                if k != apex:
                    imgs.pop(k, None)
            del kept[len(kept) - n_off:]
            kept.append(apex)
            return
    kept.append(e)
    imgs[e] = img


def _le_pack(out, kept, n_images):
    """Fill the event-index fields of an add_buffer message for the kept path `kept`.
    Frames tracked by _loop_erase are the last ones of the buffer; the leading ones (demo video, first frame) are never
    erased. Kept frames already on the server form a common prefix of _le["sent"] and kept: the server keeps that many
    stored execution frames ("trunc_exec") and deletes the rest. Kept frames sent earlier but past that prefix (loop
    apices) are re-inserted from the server's store ("reinsert_exec"); new frames not on the kept path are flagged."""
    sent = _le["sent"]
    common = 0
    while common < min(len(kept), len(sent)) and kept[common] == sent[common]:
        common += 1
    n_exec = len(_le["q"]) - _le["buf0"]
    n_fixed = n_images - n_exec
    assert n_fixed >= 0 and (n_fixed == 0 or not sent), (n_fixed, len(sent))
    old = [e for e in kept[common:] if e < _le["buf0"]]
    new = set(kept[common + len(old):])
    assert all(e >= _le["buf0"] for e in new), (sorted(new)[:3], _le["buf0"])
    out["null_flags"] = np.array([False] * n_fixed + [e not in new for e in range(_le["buf0"], len(_le["q"]))],
                                 dtype=bool)
    out["trunc_exec"] = common
    if old:
        out["reinsert_exec"] = np.array([sent.index(e) for e in old], dtype=np.int32)
    out["n_fixed"] = n_fixed
    return out


# ---- null events ----------------------------------------------------------------------------------------------------
class _EpisodeStopped(Exception):
    pass


class _PlannerEnv:
    """The env as the motion planner sees it: every planner step goes through the evaluator's env runner."""

    def __init__(self, env_runner, on_frame):
        self._runner, self._on_frame = env_runner, on_frame
        self.unwrapped = env_runner.env.unwrapped

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        try:
            obs, stop, flag = self._runner.step(action)
        except Exception as e:             # env_runner indexes obs outside its own try block
            print(f"[airgrasp] env error during air-grasp: {type(e).__name__}: {e}")
            obs, stop, flag = (None, None, None), True, "error"
        self._on_frame(obs, action, flag)
        if stop:
            raise _EpisodeStopped(flag)
        return None, 0.0, False, False, {}


def _cube_positions(u):
    cubes = list(getattr(u, "all_cubes", []) or []) or ([u.cube] if getattr(u, "cube", None) is not None else [])
    return {i: c.pose.p[0].cpu().numpy().copy() for i, c in enumerate(cubes)}


def _button_depth(u):
    b = getattr(u, "button", None)
    if b is None:
        return 0.0
    try:
        return float(-b.get_qpos()[..., 0].reshape(-1)[0])
    except Exception:
        return 0.0


def _world_check(u, before, button_before, done):
    """Did the event leave the world unchanged? Returns (valid, description)."""
    after = _cube_positions(u)
    shifts = [float(np.linalg.norm(after[k] - before[k])) for k in before if k in after]
    deleted = sum(d > 0.5 for d in shifts)            # BinFill removes cubes that enter the bin
    moved = max([d for d in shifts if d <= 0.5], default=0.0)
    pressed = _button_depth(u) > 0.005 and button_before <= 0.005
    valid = done and moved < 0.01 and not deleted and not pressed
    return valid, (f"cubes moved max {moved * 100:.2f} cm, deleted {deleted}, button pressed {pressed} "
                   f"-> {'ok' if valid else 'INVALID'}")


def _airgrasp_target(u):
    """A cube resting on the table to fake-grasp, or (None, reason) if an air-grasp is not safe now."""
    cubes = list(getattr(u, "all_cubes", []) or [])
    if not cubes:
        return None, "task has no cubes"
    if any(bool(u.agent.is_grasping(c).any()) for c in cubes):
        return None, "holding a cube"
    target = getattr(u, "target_cube", None)
    tcp = u.agent.tcp.pose.p[0].cpu().numpy()
    board = getattr(u, "board_with_hole", None)     # BinFill: 10 cm board with the bin hole, 5 cm tall
    board_xy = board.pose.p[0].cpu().numpy()[:2] if board is not None else None
    best = None
    for c in ([target] if target is not None else cubes):
        p = c.pose.p[0].cpu().numpy()
        if not 0.0 < p[2] < 0.06:            # not resting on the table
            continue
        if board_xy is not None and np.linalg.norm(p[:2] - board_xy) < 0.09:
            continue                         # in / on the bin
        d = float(np.linalg.norm(p[:2] - tcp[:2]))
        if best is None or d < best[0]:
            best = (d, c)
    return (best[1], "ok") if best else (None, "no cube resting on the table")


def _with_retries(planner):
    """Screw moves: retry once, then fall back to RRTConnect (like RoboMME's DemonstrationWrapper)."""
    screw, rrt = planner.move_to_pose_with_screw, planner.move_to_pose_with_RRTConnect

    def move(pose, *a, **k):
        # A zero-length move makes mplib fail (screw: zero twist; RRT: TOPP on a 1-point path), e.g. the
        # "lift to 15 cm" when the TCP is already that high. Skip it: the arm is already there.
        if isinstance(pose, sapien.Pose):
            cur = planner.base_env.agent.tcp.pose.sp
            if np.linalg.norm(cur.p - pose.p) < 0.005 and abs(float(np.dot(cur.q, pose.q))) > 1 - 1e-4:
                return None
        errs = []
        for f in (screw, screw, rrt, rrt):
            try:
                r = f(pose, *a, **k)
            except _EpisodeStopped:
                raise
            except Exception as e:
                errs.append(f"{type(e).__name__}: {e}")
                continue
            if not (isinstance(r, int) and r == -1):
                return r
        raise RuntimeError("screw and RRTConnect planning failed" + (f" (last error: {errs[-1]})" if errs else ""))

    planner.move_to_pose_with_screw = move
    return planner


if AIRGRASP:
    import sapien
    from robomme.robomme_env.utils.planner_fail_safe import FailAwarePandaArmMotionPlanningSolver
    from robomme.robomme_env.utils.subgoal_planner_func import solve_pickup_fail

_spec = importlib.util.spec_from_file_location("robomme_eval", os.path.join(EVAL_DIR, "eval.py"))
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)

# Symbolic baselines (MemER-style keyframes, text sub-goals): a fine-tuned Qwen3-VL-4B predicts sub-goals in the
# client. VLM_DIR points it at a local copy of the base model (offline compute nodes); sdpa attention avoids the
# optional flash-attn build.
SUBGOAL = os.environ.get("SUBGOAL", "")
VLM_DIR = os.environ.get("VLM_DIR", "")
if VLM_DIR:
    for _m in ("subgoal_prediction.qwenvl.api", "subgoal_prediction.qwenvl.api_memer"):
        _mod = sys.modules[_m]

        def _local_engine(*a, _engine=_mod.PtEngine, **k):
            if k.get("model_id_or_path") == "Qwen/Qwen3-VL-4B-Instruct":
                k["model_id_or_path"] = VLM_DIR
            k["attn_impl"] = "sdpa"
            return _engine(*a, **k)
        _mod.PtEngine = _local_engine
# The VLM sub-goal predictor is queried once per action chunk. A visible null event runs inside one loop iteration, so
# without this the predictor would never see it: query it at the same cadence during the event, as a deployed
# predictor running on a clock would (it may advance a phantom sub-goal).
SG_EVENTS = SUBGOAL not in ("", "oracle_simple", "oracle_grounded")


def _event_subgoal(subgoal_predictor, epstate, subgoal, last_subgoal, env_runner, kind):
    new, _ = subgoal_predictor.get_subgoal(epstate.count, subgoal, last_subgoal)
    if not subgoal_predictor.video_buffer:   # QwenVL clears it and reads [-1] at the next query, which may come
        subgoal_predictor.video_buffer.append(epstate.image_buffer[-1])   # before any new frame (event end)
    if new != subgoal:
        print(f"[subgoal] ep {env_runner.episode_id}: during {kind} at policy step {epstate.count}: '{subgoal}' -> '{new}'")
    return new


_null_frames_in_buffer = [0]   # visible event frames are always the last frames of the next buffer sent


def eval_each_episode(self, env_runner, subgoal_predictor, video_save_dir):
    """Copy of EpisodeEvaluator.eval_each_episode with null-event injection (marked EVENT)."""
    client = E._websocket_client_policy.MMEVLAWebsocketClientPolicy(self.args.host, self.args.port)
    resp = client.reset()
    while not resp.get("reset_finished", False):
        time.sleep(0.1)

    epstate = E.EpisodeState()
    task_goal, recorder = self.init_episode(env_runner, epstate, video_save_dir)
    subgoal_predictor.start_episode(epstate, env_runner)

    img, wrist_img, robot_state = epstate.get_current_obs()
    prompt = task_goal
    success_flag = "unknown"
    subgoal = None
    last_subgoal = None
    last_action = None                       # EVENT
    pending = list(EVENT_AT)                 # EVENT: thresholds not yet used
    event_log = []                           # EVENT: policy steps at which events ran
    validity = []                            # EVENT: "ok" / "INVALID" per fidget / air-grasp
    last_defer_reason = None                 # EVENT (air-grasp)
    _null_frames_in_buffer[0] = 0
    _le.update(q=[], kept=[], img={}, sent=[], buf0=0, last_change=0, ref=-1, run=0, erased=0, loops=0, still=0, cc=0)

    while True:
        subgoal_predictor.step(epstate)

        # EVENT: air-grasp at a chunk boundary past the next threshold, as soon as no cube is held
        if AIRGRASP and pending and not epstate.action_plan and last_action is not None and epstate.count >= pending[0]:
            u = env_runner.env.unwrapped
            obj, why = _airgrasp_target(u)
            if obj is None:
                if why != last_defer_reason:
                    print(f"[airgrasp] ep {env_runner.episode_id}: deferred at policy step {epstate.count}: {why}")
                last_defer_reason = why
            else:
                pending.pop(0)
                last_defer_reason = None
                held = {"obs": None, "flag": success_flag, "n": 0}

                def on_frame(obs, act, flag):
                    nonlocal subgoal, last_subgoal
                    held["obs"], held["flag"] = obs, flag
                    held["n"] += 1
                    im, wim, st = obs
                    if im is None:
                        return
                    if VISIBLE:
                        epstate.add_observation(im, wim, st)
                        _null_frames_in_buffer[0] += 1
                        _flag_frame(env_runner)
                        subgoal_predictor.step(epstate)   # MemER stores every execution frame
                        if SG_EVENTS and held["n"] % self.args.obs_horizon == 0:
                            subgoal = last_subgoal = _event_subgoal(subgoal_predictor, epstate, subgoal, last_subgoal,
                                                                    env_runner, "airgrasp")
                    recorder.record(image=im.copy(), wrist_image=wim.copy(), state=st.copy(),
                                    action=act.copy(), subgoal=subgoal)

                before, button_before = _cube_positions(u), _button_depth(u)
                start_pose = u.agent.tcp.pose.sp
                start_fingers = float(u.agent.robot.get_qpos()[0, -2:].sum())   # ~0.08 open, ~0 closed
                stopped = False
                try:
                    planner = _with_retries(FailAwarePandaArmMotionPlanningSolver(
                        _PlannerEnv(env_runner, on_frame), debug=False, vis=False,
                        base_pose=u.agent.robot.pose, visualize_target_grasp_pose=False, print_env_info=False))
                    lift = start_pose.p.copy()
                    lift[2] = max(lift[2], 0.15)                       # straight up first: no obstacle model
                    planner.move_to_pose_with_screw(sapien.Pose(p=lift, q=start_pose.q))
                    solve_pickup_fail(planner.env, planner, obj=obj, z_offset=float(u.cube_half_size) * 2,
                                      xy_offset=None, mode="z")
                    planner.open_gripper()                             # 6 more steps: fully open
                    if AIRGRASP_RETURN:
                        planner.move_to_pose_with_screw(sapien.Pose(p=lift, q=start_pose.q))
                        planner.move_to_pose_with_screw(start_pose)
                        if start_fingers < 0.04:
                            planner.close_gripper(t=10)
                    status = "done"
                except _EpisodeStopped as e:
                    stopped, status = True, f"episode ended during air-grasp ({e})"
                except Exception as e:  # planning failure: the arm may have moved part of the way
                    status = f"planner failed: {type(e).__name__}: {e}"
                u._pending_waypoint = None
                valid, desc = _world_check(u, before, button_before, status == "done")
                if held["obs"] is not None and held["obs"][0] is not None:
                    img, wrist_img, robot_state = held["obs"]
                if stopped:
                    success_flag = held["flag"] if held["flag"] in ("success", "fail", "timeout", "error") else "error"
                event_log.append(epstate.count)
                validity.append("ok" if valid else "INVALID")
                print(f"[airgrasp] ep {env_runner.episode_id}: at policy step {epstate.count}: {held['n']} env steps, "
                      f"{status}; {desc}")
                if stopped:
                    break

        # EVENT: pause / fidget only at a chunk boundary, once the policy has run past the next threshold
        if (EVENT in ("pause", "fidget") and EVENT_LEN > 0 and pending and not epstate.action_plan
                and last_action is not None and epstate.count >= pending[0]):
            pending.pop(0)
            stopped = False
            n_fidget = max(EVENT_LEN - 8, 0) // FIDGET_PERIOD * FIDGET_PERIOD if EVENT == "fidget" else 0
            u = env_runner.env.unwrapped
            if n_fidget:
                before, button_before = _cube_positions(u), _button_depth(u)
            for t in range(EVENT_LEN):
                act = last_action
                if t < n_fidget:
                    act = np.array(last_action, dtype=np.float32, copy=True)
                    act[6] += FIDGET_AMP * np.sin(2 * np.pi * t / FIDGET_PERIOD)
                obs, stop_flag, success_flag = env_runner.step(act)
                img, wrist_img, robot_state = obs
                if VISIBLE:
                    epstate.add_observation(img, wrist_img, robot_state)
                    _null_frames_in_buffer[0] += 1
                    _flag_frame(env_runner)
                    subgoal_predictor.step(epstate)       # MemER stores every execution frame
                    if SG_EVENTS and (t + 1) % self.args.obs_horizon == 0:
                        subgoal = last_subgoal = _event_subgoal(subgoal_predictor, epstate, subgoal, last_subgoal,
                                                                env_runner, "pause")
                recorder.record(image=img.copy(), wrist_image=wrist_img.copy(),
                                state=robot_state.copy(), action=np.array(act).copy(), subgoal=subgoal)
                if stop_flag:
                    stopped = True
                    break
            if n_fidget:
                valid, desc = _world_check(u, before, button_before, not stopped)
                validity.append("ok" if valid else "INVALID")
                print(f"[fidget] ep {env_runner.episode_id}: at policy step {epstate.count}: {n_fidget} fidget steps, {desc}")
            event_log.append(epstate.count)
            print(f"[{EVENT}] ep {env_runner.episode_id}: held {EVENT_LEN} steps at policy step {epstate.count}")
            if stopped:
                break

        if not epstate.action_plan:
            if epstate.count % self.args.subgoal_keep_period == 0 or last_subgoal is None:
                subgoal, has_api_error = subgoal_predictor.get_subgoal(epstate.count, subgoal, last_subgoal)
            else:
                subgoal = last_subgoal
                has_api_error = False
            if has_api_error:
                break
            action_chunk = self.get_action_chunk(
                client, epstate, img, wrist_img, robot_state, prompt, subgoal,
                exec_horizon=self.args.obs_horizon,
            )
            epstate.action_plan.extend(action_chunk)
            epstate.clear_buffers()
            last_subgoal = subgoal

        action = epstate.action_plan.popleft()
        last_action = action                 # EVENT
        obs, stop_flag, success_flag = env_runner.step(action)
        epstate.count += 1

        if epstate.count > self.args.max_steps:
            success_flag = "timeout"
            break

        img, wrist_img, robot_state = obs
        epstate.add_observation(img, wrist_img, robot_state)
        _flag_frame(env_runner)
        recorder.record(image=img.copy(), wrist_image=wrist_img.copy(),
                        state=robot_state.copy(), action=action.copy(), subgoal=subgoal)
        if stop_flag:
            break

    if success_flag == "unknown":
        return "unknown"

    # One line per episode. The sub-goal the env was on when the episode ended tells a premature button press (fail on
    # a pick / place sub-goal) from an extra repetition (fail on "press the button") or a stall (timeout).
    u = env_runner.env.unwrapped
    held_wrong = any(bool(u.agent.is_grasping(c).any()) for c in (getattr(u, "non_target_cubes", None) or []))
    print(f"[episode] ep {env_runner.episode_id}: result {success_flag}, policy steps {epstate.count}, "
          f"events at {event_log}" + (f" {validity}" if validity else "")
          + f", subgoal {getattr(u, 'current_task_index', '?')}/{len(getattr(u, 'task_list', []) or [])}"
          f" '{getattr(u, 'current_task_name', '?')}', button {'pressed' if _button_depth(u) > 0.005 else 'up'}"
          f"{', holding non-target cube' if held_wrong else ''}")
    if EVENT_INDEX:
        print(f"[event_index] ep {env_runner.episode_id}: {_le['still']} still frames, {_le['loops']} loops "
              f"({_le['erased'] - _le['still']} frames); {_le['erased']} of {len(_le['q'])} execution frames erased, "
              f"{len(_le['kept'])} kept, {_le['cc']} contact changes")
    video_filename = f"{env_runner.env_id}_ep{env_runner.episode_id}_{success_flag}_{task_goal}_{env_runner.difficulty}.mp4"
    recorder.save_video(video_filename)
    subgoal_predictor.end_episode(epstate, success_flag)
    return success_flag


E.EpisodeEvaluator.eval_each_episode = eval_each_episode
_orig_pack_buffer = E.pack_buffer


def _pack_buffer(image_buffer, state_buffer, exec_start_idx=0):
    out = _orig_pack_buffer(image_buffer, state_buffer, exec_start_idx)
    flags = np.zeros(len(image_buffer), dtype=bool)
    if _null_frames_in_buffer[0]:
        flags[-_null_frames_in_buffer[0]:] = True
    out["null_flags"] = flags
    _null_frames_in_buffer[0] = 0
    if EVENT_INDEX:                          # the detector's flags replace the injection flags
        _le_pack(out, _le["kept"], len(image_buffer))
        _le["sent"] = list(_le["kept"])
        _le["buf0"] = len(_le["q"])
    return out


E.pack_buffer = _pack_buffer

if __name__ == "__main__":
    import tyro

    print(f"[eval_client] EVENT={EVENT} EVENT_LEN={EVENT_LEN} EVENT_AT={EVENT_AT} EVENT_MEMORY={EVENT_MEMORY} "
          f"INDEX={'event' if EVENT_INDEX else 'time'} SPLIT={SPLIT} SUBGOAL={SUBGOAL or '-'} "
          f"RENDER={os.environ.get('RENDER', 'default')} (env step limit +{TOTAL_EVENT})", flush=True)
    tyro.cli(E.evaluate)
