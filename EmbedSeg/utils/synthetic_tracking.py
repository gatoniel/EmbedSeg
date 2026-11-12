"""
Tiny synthetic 2D+t / 3D+t data (moving discs/balls, some of which divide) for
tests and quick sanity experiments with the tracking code.
"""
import os

import numpy as np
import tifffile

from EmbedSeg.utils.tracking import build_parent_map


def _grid(shape):
    return np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"), 0)


def make_sequence(
    ndim=2,
    shape=None,
    n_frames=6,
    n_objects=3,
    radius=4.0,
    speed=1.5,
    divide=((3, 0),),
    seed=0,
):
    """Simulate objects moving linearly. ``divide`` = ((frame, object_index), ...):
    at ``frame`` the object is replaced by two daughters with *new* ids.

    Returns ``dict(images, instances, centers, lineage)``; ``instances`` carry
    track-consistent ids, ``lineage`` maps daughter id -> parent id.
    """
    rng = np.random.default_rng(seed)
    shape = tuple(shape or ((48,) * ndim))
    grid = _grid(shape)
    # well separated start positions on a coarse lattice
    cells = int(np.ceil(n_objects ** (1.0 / ndim))) + 1
    lattice = np.stack(
        np.meshgrid(*[np.arange(1, cells)] * ndim, indexing="ij"), -1
    ).reshape(-1, ndim)
    rng.shuffle(lattice)
    pos = {}
    vel = {}
    rad = {}
    next_id = 1
    for k in range(n_objects):
        pos[next_id] = lattice[k] * (np.array(shape) / cells) + rng.uniform(-2, 2, ndim)
        v = rng.normal(size=ndim)
        vel[next_id] = speed * v / (np.linalg.norm(v) + 1e-9)
        rad[next_id] = radius
        next_id += 1
    order = list(pos)  # object index -> current id
    lineage = {}
    images, instances, centers = [], [], []
    for t in range(n_frames):
        for (tf, oi) in divide:
            if tf == t and oi < len(order):
                pid = order[oi]
                d = rng.normal(size=ndim)
                d = d / (np.linalg.norm(d) + 1e-9) * radius * 0.7
                new = []
                for sgn in (+1, -1):
                    pos[next_id] = pos[pid] + sgn * d
                    vel[next_id] = vel[pid] + sgn * d / (radius * 0.7) * 0.5
                    rad[next_id] = radius * 0.8
                    lineage[next_id] = pid
                    new.append(next_id)
                    next_id += 1
                del pos[pid]
                order[oi] = new[0]
                order.append(new[1])
        inst = np.zeros(shape, dtype=np.int16)
        cen = np.zeros(shape, dtype=bool)
        for i, c in pos.items():
            d2 = ((grid - np.reshape(c, (ndim,) + (1,) * ndim)) ** 2).sum(0)
            inst[d2 <= rad[i] ** 2] = i
            cen[tuple(np.clip(np.round(c).astype(int), 0, np.array(shape) - 1))] = True
        img = (inst > 0).astype(np.float32) * 200 + rng.normal(0, 5, shape) + 20
        images.append(img.astype(np.float32))
        instances.append(inst)
        centers.append(cen)
        for i in pos:
            pos[i] = pos[i] + vel[i]
    return dict(
        images=np.stack(images),
        instances=np.stack(instances),
        centers=np.stack(centers),
        lineage=lineage,
    )


def write_tracking_samples(seq, data_dir, split="train", n_frames=2, center="center-medoid", prefix="seq0"):
    """Write one sample per reference frame ``t >= n_frames-1`` in the on-disk
    format read by :class:`EmbedSeg.datasets.TrackingDataset`."""
    paths = {k: os.path.join(data_dir, split, k) for k in ("images", "masks", center)}
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    T = len(seq["images"])
    names = []
    for t in range(n_frames - 1, T):
        sl = slice(t - n_frames + 1, t + 1)
        inst = seq["instances"][sl]
        parent = build_parent_map(inst[-1], inst[-2], seq["lineage"])
        masks = np.concatenate([inst, parent[None]], 0).astype(np.int16)
        name = "{}_{:04d}.tif".format(prefix, t)
        tifffile.imwrite(os.path.join(paths["images"], name), seq["images"][sl])
        tifffile.imwrite(os.path.join(paths["masks"], name), masks)
        tifffile.imwrite(
            os.path.join(paths[center], name), seq["centers"][sl].astype(np.uint8)
        )
        names.append(name)
    return names
