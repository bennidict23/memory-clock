"""Policy server: RoboMME's scripts/serve_policy.py plus test-time memory manipulations (MEM_MODE).

The upstream server and policy code are not modified; the memory methods of MME_VLA_Policy / MemoryBuffer are
replaced at runtime. A FrameSamp memory stores one entry per frame, keyed by its step index, and before every action
chunk samples 32 of them evenly over [0, step], each carrying the time-position embedding of its key (its "time stamp").
MEM_MODE:
  stock        unchanged upstream behaviour.
  clock_scale  every frame is stored as usual, but its time position is round(step x CLOCK_SCALE) (clamped to the
               table). No frame is dropped: tests whether the policy reads progress from the time stamps.
  nullswap     visible null events: the frames flagged by the client ("null_flags") are stored and sampled as usual,
               but once 32 frames are sampled their time positions are linspace(0, step - n_null, 32) (the clock reads
               the working time), and every null frame's image and state embeddings are replaced by those of the last
               real frame before the first null event. Keeps token count, stillness and timing; changes only which
               action phase the null frames show.
  dupfill      no event: the 32 slots are spread over the working time plus the frames a visible SIM_PAUSE-step pause
               at each of SIM_AT would have stored, and the slots that would have held pause frames are dropped; the
               remaining real frames are then each repeated in order to fill all 32 slots (duplicates keep their own
               time positions). The real frames of a visible pause, without any null frame.
  nullblock    no event: with b = number of SIM_AT points t <= step, the memory holds 32 - NB_M*b real keys evenly over
               [0, step] plus NB_M copies of key t right after the last real key <= t for each passed t (copies keep
               key t's features and position). A block of NB_M identical slots without any pause; NB_M=0 is stock.
  event_index  the event index (client INDEX=event): flagged frames are not stored and do not advance the clock, so
               every stored frame's time position counts the frames kept before it. When a loop closes at the newest
               frame the client asks to erase frames it already sent: "trunc_exec" = how many of the stored execution
               frames to keep; the rest are deleted and the clock is set back; "reinsert_exec" lists erased frames
               that stay (loop apices), moved up to the next clock ticks.
Websocket keepalive pings are disabled: ingesting a long video history (e.g. ~1000 frames through SigLIP, plus the
first-call JIT) blocks the server's event loop for longer than the default 20 s ping timeout, and the connection
would be dropped. This changes nothing else about evaluation.
Command-line arguments are passed to serve_policy.py unchanged.
"""
import os
import runpy
import sys

import numpy as np
import websockets.asyncio.server as _server

from mme_vla_suite.policies import policy as _policy
from mme_vla_suite.shared import mem_buffer as _mb
from mme_vla_suite.shared.data_utils import even_sampling_indices

HERE = os.path.dirname(os.path.abspath(__file__))
ROBOMME_DIR = os.environ.get("ROBOMME_DIR", os.path.join(HERE, "..", "third_party", "robomme_policy_learning"))
MODE = os.environ.get("MEM_MODE", "stock")
assert MODE in ("stock", "clock_scale", "nullswap", "dupfill", "nullblock", "event_index"), MODE
CLOCK_SCALE = float(os.environ.get("CLOCK_SCALE", "1.0"))
SIM_PAUSE = int(os.environ.get("SIM_PAUSE", "100"))
SIM_AT = [int(x) for x in os.environ.get("SIM_AT", "48,160").split(",") if x.strip()]
NB_M = int(os.environ.get("NB_M", "0"))
assert NB_M >= 0, NB_M
_G = {}   # per-episode bookkeeping (one server per evaluation run, so per episode after each reset)

_orig_serve = _server.serve


def _serve(*a, **k):
    k.setdefault("ping_interval", None)
    return _orig_serve(*a, **k)


_server.serve = _serve
_orig_add_buffer = _policy.MME_VLA_Policy.add_buffer


def _set_time_positions(buf, keys, pe_idx):
    v = buf.num_views
    for key, pe in zip(keys, pe_idx):
        for size in ("8x8", "4x4", "2x2"):
            buf._history_feats[key][f"pos_emb_{size}"] = buf.pos_emb_dict[size][pe * v:(pe + 1) * v]


def _with_positions(buf, feats, indices, pos):
    """Copies of the gathered features with the given time positions (the stored frames keep their own)."""
    v = buf.num_views
    return {key: dict(feats[key], **{f"pos_emb_{s}": buf.pos_emb_dict[s][p * v:(p + 1) * v]
                                     for s in ("8x8", "4x4", "2x2")})
            for key, p in zip(indices, pos)}


# ---- clock_scale ----------------------------------------------------------------------------------------------------
def add_buffer_clock_scale(self, obs: dict) -> None:
    if self.mem_buffer is None:
        return
    first = self.step_idx + 1
    _orig_add_buffer(self, obs)
    keys = list(range(first, self.step_idx + 1))
    n_max = len(self.mem_buffer.pos_emb_dict["8x8"]) // self.mem_buffer.num_views - 1
    _set_time_positions(self.mem_buffer, keys, [min(int(round(k * CLOCK_SCALE)), n_max) for k in keys])


# ---- nullswap -------------------------------------------------------------------------------------------------------
SWAP_KEYS = ("image_emb_8x8", "image_emb_4x4", "image_emb_2x2", "state_emb")


def add_buffer_track(self, obs: dict) -> None:
    """Remember the keys of null frames and the last real frame before the first null frame."""
    if self.mem_buffer is not None:
        if self.step_idx < 0:
            _G.update(n_null=0, null=set(), pre=None)
        flags = np.asarray(obs.get("null_flags", []), dtype=bool)
        _G["n_null"] += int(flags.sum())
        for i, f in enumerate(flags):            # keys as in MME_VLA_Policy.add_buffer: step_idx + 1 + i
            if f:
                _G["null"].add(self.step_idx + 1 + i)
            elif not _G["null"]:
                _G["pre"] = self.step_idx + 1 + i
    _orig_add_buffer(self, obs)


def prepare_frame_sampling_nullswap(self, step_idx, token_budget, token_per_image, history_feats_gather_fn, *a, **k):
    indices = self.get_frame_sampling_indices(step_idx, token_budget, token_per_image)
    feats = history_feats_gather_fn(indices, *a, **k)
    max_size = token_budget // (token_per_image * self.num_views)
    if _G["n_null"] > 0 and len(indices) == max_size:   # without null frames this is the stock pattern
        tpos = np.linspace(0, step_idx - _G["n_null"], max_size, dtype=np.int32).tolist()
        feats = _with_positions(self, feats, indices, tpos)
    if any(key in _G["null"] for key in indices) and _G["pre"] is not None:
        src = history_feats_gather_fn([_G["pre"]], *a, **k)[_G["pre"]]
        feats = {key: dict(f, **{n: src[n] for n in SWAP_KEYS}) if key in _G["null"] else f
                 for key, f in feats.items()}
    return self._prepare_frame_sampling(feats, indices, token_budget, token_per_image)


# ---- dupfill --------------------------------------------------------------------------------------------------------
def starve_indices(step_idx, max_size):
    """Real keys left in the 32 slots if a visible SIM_PAUSE-step pause had fired at each SIM_AT <= step_idx (pause i
    stores all-time keys t+i*P+1 .. t+(i+1)*P, as with visible pauses); returns (keys, n_virtual_pauses)."""
    nulls = [(t + i * SIM_PAUSE + 1, t + (i + 1) * SIM_PAUSE) for i, t in enumerate(sorted(SIM_AT)) if step_idx >= t]
    keys = []
    for key in even_sampling_indices(step_idx + SIM_PAUSE * len(nulls), max_size):
        off = 0
        for lo, hi in nulls:
            if key < lo:
                break
            if key <= hi:
                off = None
                break
            off += SIM_PAUSE
        if off is not None:
            keys.append(key - off)
    return keys, len(nulls)


def prepare_frame_sampling_dupfill(self, step_idx, token_budget, token_per_image, history_feats_gather_fn, *a, **k):
    max_size = token_budget // (token_per_image * self.num_views)
    indices, n_virtual = starve_indices(step_idx, max_size)
    if n_virtual and len(indices) < max_size:
        indices = [indices[i * len(indices) // max_size] for i in range(max_size)]   # each kept frame ~1.5x
    feats = history_feats_gather_fn(indices, *a, **k)
    return self._prepare_frame_sampling(feats, indices, token_budget, token_per_image)


# ---- nullblock ------------------------------------------------------------------------------------------------------
def nullblock_indices(step_idx, max_size):
    """max_size - NB_M*b real keys evenly over [0, step_idx], then NB_M copies of key t right after the last real
    key <= t for each of the b SIM_AT points t <= step_idx (chronological)."""
    passed = sorted(t for t in SIM_AT if step_idx >= t)
    assert max_size - NB_M * len(passed) > 0, (max_size, NB_M, passed)
    real = even_sampling_indices(step_idx, max_size - NB_M * len(passed))
    keys, todo = [], list(passed)
    for key in real + [None]:                     # None flushes the points >= the last real key
        while todo and (key is None or key > todo[0]):
            keys += [todo.pop(0)] * NB_M
        if key is not None:
            keys.append(key)
    return keys


def prepare_frame_sampling_nullblock(self, step_idx, token_budget, token_per_image, history_feats_gather_fn, *a, **k):
    indices = nullblock_indices(step_idx, token_budget // (token_per_image * self.num_views))
    feats = history_feats_gather_fn(indices, *a, **k)
    return self._prepare_frame_sampling(feats, indices, token_budget, token_per_image)


# ---- event_index ----------------------------------------------------------------------------------------------------
def _rekey(buf, ent, k):
    """A stored frame moved to time position k: same image/state features, the position embedding of k."""
    ent, v = dict(ent), buf.num_views
    for s in ("8x8", "4x4", "2x2"):
        if f"pos_emb_{s}" in ent:
            ent[f"pos_emb_{s}"] = buf.pos_emb_dict[s][k * v:(k + 1) * v]
    return ent


def add_buffer_event_index(self, obs: dict) -> None:
    """Drop flagged frames (no key, no clock tick); apply the client's loop erasures to the stored frames."""
    if self.mem_buffer is None:
        return
    images, states = obs["images"], obs["state"]
    flags = np.asarray(obs.get("null_flags", np.zeros(len(images), dtype=bool)), dtype=bool)
    if obs.get("exec_start_idx", 0) > 0:
        self.exec_start_idx = obs["exec_start_idx"]
    if self.step_idx < 0:
        _G.update(base=0)
    if "trunc_exec" in obs:                      # keep the first trunc_exec execution frames stored
        cut = _G["base"] + int(obs["trunc_exec"])
        hist = self.mem_buffer._history_feats
        re = [hist[_G["base"] + int(i)] for i in obs.get("reinsert_exec", [])]
        for k in [k for k in hist if k >= cut]:
            del hist[k]
        self.step_idx = cut - 1
        for ent in re:
            self.step_idx += 1
            hist[self.step_idx] = _rekey(self.mem_buffer, ent, self.step_idx)
    keep = [i for i in range(len(images)) if not flags[i]]
    if keep:
        self.mem_buffer.add_buffer(images[keep], states[keep], list(range(self.step_idx + 1, self.step_idx + 1 + len(keep))))
        self.step_idx += len(keep)
    _G["base"] += int(obs.get("n_fixed", 0))    # leading frames the client never erases (demo video, first frame)


if MODE == "clock_scale":
    _policy.MME_VLA_Policy.add_buffer = add_buffer_clock_scale
elif MODE == "nullswap":
    _policy.MME_VLA_Policy.add_buffer = add_buffer_track
    _mb.MemoryBuffer.prepare_frame_sampling = prepare_frame_sampling_nullswap
elif MODE == "dupfill":
    _mb.MemoryBuffer.prepare_frame_sampling = prepare_frame_sampling_dupfill
elif MODE == "nullblock":
    _mb.MemoryBuffer.prepare_frame_sampling = prepare_frame_sampling_nullblock
elif MODE == "event_index":
    _policy.MME_VLA_Policy.add_buffer = add_buffer_event_index
print(f"[policy_server] MEM_MODE={MODE}" + {
    "clock_scale": f" CLOCK_SCALE={CLOCK_SCALE}", "dupfill": f" SIM_PAUSE={SIM_PAUSE} SIM_AT={SIM_AT}",
    "nullblock": f" NB_M={NB_M} SIM_AT={SIM_AT}"}.get(MODE, ""), flush=True)

sys.argv[0] = "serve_policy.py"
runpy.run_path(os.path.join(ROBOMME_DIR, "scripts", "serve_policy.py"), run_name="__main__")
