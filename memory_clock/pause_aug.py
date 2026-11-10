"""Pause augmentation of the memory timeline of training samples (train.py with PAUSE_AUG=1).

Only the memory of a sample is edited; the current observation and the action labels are untouched. The timeline of a
sample at step s is the list of stored frames 0..s of its episode (demo video first). With probability p_pause,
1..max_pauses pauses are inserted, each a run of L copies (L log-uniform in [20, pause_len_max]) of a random frame
that is not the current one. The 32 memory slots are then spread evenly over the augmented timeline, and every slot
gets the time position of its place in the timeline, so time keeps advancing over a pause as it does at test time.

The dataset class lives in this module (not in train.py) because openpi's data-loader workers are started with
multiprocessing "spawn": each worker re-imports modules and unpickles the dataset, so only a module-level subclass that
the workers can import carries the augmentation into them. Everything it needs is computed in __init__ in the main
process and travels with the pickled dataset.
"""
import math
import os

import numpy as np

from mme_vla_suite.training import dataset as _ds

N_SLOTS = 32


def pause_timeline(step_idx, exec_start, rng, p_pause=0.8, max_pauses=3, pause_len=(20, 600)):
    """Frame index per timeline entry; the last entry is always the current frame step_idx. Demo-video frames
    (before exec_start) are never edited."""
    exec_start = int(min(max(exec_start, 0), step_idx))
    real = list(range(exec_start, step_idx + 1))
    # Two draws that decided time-warp and replay augmentations in the paper's implementation (both disabled for
    # the pause arm); they are kept so that the random stream, and hence every sample, is the same.
    rng.random()
    rng.random()
    if rng.random() < p_pause and len(real) >= 2:         # needs a frame before the current one
        for _ in range(int(rng.integers(1, max_pauses + 1))):
            L = int(round(math.exp(rng.uniform(math.log(pause_len[0]), math.log(pause_len[1])))))
            at = int(rng.integers(0, len(real) - 1))       # never after the current frame
            real[at + 1:at + 1] = [real[at]] * L
    return np.array(list(range(exec_start)) + real, dtype=np.int64)


def select_slots(timeline, n_slots=N_SLOTS):
    """Memory slots spread evenly over the timeline: (frame index per slot, time position per slot)."""
    pos = np.arange(len(timeline)) if len(timeline) <= n_slots else np.linspace(0, len(timeline) - 1, n_slots).astype(int)
    return timeline[pos], pos


def pe_table_4x4(max_len=4096, dim=768):
    """Time-position embeddings for 4x4 frames, (max_len, 16, dim), as stored in the feature files."""
    import jax.numpy as jnp
    from mme_vla_suite.shared.posemb_3d import PosEmb3D
    pe = np.asarray(PosEmb3D(dim=dim)(jnp.arange(max_len), 4), dtype=np.float32)
    return pe.reshape(max_len, -1, dim)


class PauseAugSampleDataset(_ds.SampleDataset):
    def __getitem__(self, idx):       # remember the demo-video length of the sample being built (one per worker)
        data = super().__getitem__(idx)
        self._last_exec_start = int(data["exec_start_idx"].item())
        return data


class PauseAugDataset(_ds.RoboMMEDataset):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.dataset.__class__ = PauseAugSampleDataset
        self._pe = pe_table_4x4(4096)   # built here (JAX); workers only read it
        self._dose = dict(p_pause=float(os.environ.get("PAUSE_AUG_P", "0.8")),
                          max_pauses=int(os.environ.get("PAUSE_AUG_MAX", "3")),
                          pause_len=(20, int(os.environ.get("PAUSE_AUG_LEN_MAX", "600"))))
        print(f"[train] pause augmentation: {self._dose}", flush=True)

    def prepare_frame_sampling(self, epis_idx, step_idx):
        rng = np.random.default_rng()
        timeline = pause_timeline(step_idx, self.dataset._last_exec_start, rng, **self._dose)
        slots, clock = select_slots(timeline)
        feats = self._gather_history_feat(sorted(set(slots.tolist())), epis_idx)
        per_slot = {j: {"image_emb_4x4": feats[i]["image_emb_4x4"],
                        "pos_emb_4x4": self._pe[min(int(c), len(self._pe) - 1)][None],
                        "state_emb": feats[i]["state_emb"]}
                    for j, (i, c) in enumerate(zip(slots, clock))}
        return self.mem_buffer._prepare_frame_sampling(per_slot, list(range(len(slots))),
                                                       self.history_config.budget, self.history_config.token_per_image)


def install():
    """Make the training data loader build PauseAugDataset (mme_vla_suite.training.dataloader binds the class name at
    import time, so it is rebound there)."""
    from mme_vla_suite.training import dataloader as _dl
    _dl.RoboMMEDataset = PauseAugDataset
