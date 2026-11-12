"""
Inference for time-lapse tracking: association of segmented objects to the
previous frame and a driver that runs a trained tracking model over a sequence.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt, find_objects

from EmbedSeg.utils.tracking import prev_seg_channels


# ---------------------------------------------------------------- geometry
def grid_steps(grid):
    """Coordinate step per pixel for each coordinate channel of ``grid``
    (``(nd, *spatial)``, channel order x, y[, z])."""
    nd = grid.shape[0]
    steps = []
    for a in range(nd):
        ax = nd - 1 - a  # spatial array axis of coordinate a
        i0 = (a,) + (0,) * nd
        i1 = (a,) + tuple(1 if k == ax else 0 for k in range(nd))
        steps.append(float(grid[i1] - grid[i0]))
    return np.array(steps)


def embedding_to_index(emb, steps):
    """Convert predicted coordinates ``(nd, *sp)`` (x, y[, z]) to float pixel
    indices in array-axis order (``[z,] y, x``)."""
    nd = emb.shape[0]
    return np.stack([emb[nd - 1 - k] / steps[nd - 1 - k] for k in range(nd)])


def object_centers(labels):
    """Centre pixel (the object pixel closest to its centroid) of every object.
    Cheap stand-in for the medoid used for the training centre images."""
    centers = np.zeros(labels.shape, dtype=bool)
    for lab, sl in enumerate(find_objects(labels.astype(np.int32)), start=1):
        if sl is None:
            continue
        coords = np.argwhere(labels[sl] == lab)
        if len(coords) == 0:
            continue
        c = coords[np.argmin(((coords - coords.mean(0)) ** 2).sum(1))]
        centers[tuple(c + [s.start for s in sl])] = True
    return centers


# ------------------------------------------------------------- association
def associate_tracks(
    instance_t,
    track_idx,
    prev_labels,
    division_prob=None,
    next_id=None,
    min_vote=0.3,
    max_dist=2.0,
    div_threshold=0.5,
):
    """Link the objects of frame t to the objects of frame t-1.

    Every pixel of an object votes for the previous-frame object found at its
    predicted position (``track_idx``, float pixel indices in array-axis order; a
    previous object within ``max_dist`` pixels counts). The majority label is the
    parent if it collects at least ``min_vote`` of all pixels. An object that is
    the only child of its parent inherits the parent's id; if two or more objects
    share a parent, that is a division and each daughter gets a new id; objects
    without parent start a new track.

    ``division_prob`` (per-pixel, optional) is only used as a consistency check.

    Returns dict(labels, lineage {new id: parent id}, next_id, parents {object: parent},
    divisions [(parent, [new ids])], conflicts [(kind, id)]).
    """
    prev_labels = np.asarray(prev_labels)
    if next_id is None:
        next_id = int(prev_labels.max()) + 1
    if max_dist > 0 and (prev_labels > 0).any():
        dist, inds = distance_transform_edt(prev_labels == 0, return_indices=True)
        lookup = np.where(dist <= max_dist, prev_labels[tuple(inds)], 0)
    else:
        lookup = prev_labels

    ids = [int(i) for i in np.unique(instance_t[instance_t > 0])]
    hi = np.array(prev_labels.shape) - 1
    parents = {}
    for i in ids:
        m = instance_t == i
        pos = np.rint(track_idx[:, m]).astype(int)
        pos = np.clip(pos, 0, hi.reshape(-1, 1))
        votes = lookup[tuple(pos)]
        nz = votes[votes > 0]
        parents[i] = 0
        if nz.size:
            vals, counts = np.unique(nz, return_counts=True)
            k = counts.argmax()
            if counts[k] / votes.size >= min_vote:
                parents[i] = int(vals[k])

    children = {}
    for i, p in parents.items():
        if p:
            children.setdefault(p, []).append(i)

    labels = np.zeros(instance_t.shape, dtype=np.int32)
    lineage, divisions, new_id_of = {}, [], {}
    for i in ids:
        p = parents[i]
        if p and len(children[p]) == 1:
            new_id_of[i] = p
        else:
            new_id_of[i] = next_id
            next_id += 1
            if p:
                lineage[new_id_of[i]] = p
        labels[instance_t == i] = new_id_of[i]
    for p, kids in children.items():
        if len(kids) > 1:
            divisions.append((p, [new_id_of[k] for k in kids]))

    conflicts = []
    if division_prob is not None:
        prob = {i: float(division_prob[instance_t == i].mean()) for i in ids}
        for p, kids in children.items():
            if len(kids) > 1 and min(prob[k] for k in kids) < div_threshold:
                conflicts.append(("division_without_flag", p))
            if len(kids) == 1 and prob[kids[0]] >= div_threshold:
                conflicts.append(("flag_without_division", p))
        for i in ids:
            if parents[i] == 0 and prob[i] >= div_threshold:
                conflicts.append(("flag_without_parent", new_id_of[i]))
    return dict(
        labels=labels,
        lineage=lineage,
        next_id=next_id,
        parents=parents,
        divisions=divisions,
        conflicts=conflicts,
    )


# ------------------------------------------------------------------ driver
def _normalize_frame(frame, norm):
    frame = frame.astype(np.float32)
    if norm == "min-max-percentile":
        mi, ma = np.percentile(frame, 1), np.percentile(frame, 99.8)
        return (frame - mi) / (ma - mi + 1e-20)
    if norm == "mean-std":
        return (frame - frame.mean()) / (frame.std() + 1e-20)
    return frame


def _pad_to_multiple(x, m=8):
    import torch.nn.functional as F

    pads = []
    for s in reversed(x.shape[2:]):
        pads += [0, (-s) % m]
    return F.pad(x, pads, mode="reflect") if any(pads) else x


def track_sequence(
    model,
    frames,
    cluster,
    n_frames=2,
    prev_seg=("mask", "blob"),
    blob_sigma=2.0,
    n_sigma=None,
    cluster_kwargs=None,
    norm="min-max-percentile",
    device="cpu",
    min_vote=0.3,
    max_dist=2.0,
    div_threshold=0.5,
):
    """Segment and track a whole sequence ``frames`` (``(T, [Z,] Y, X)``).

    * ``t = 0``: only the segmentation heads run (``heads="seg"``) with an empty
      context (all earlier-frame channels zero).
    * ``t >= 1``: the full network runs on the window of the last ``n_frames``
      frames plus the segmentation (mask / centre blob) of the earlier frames,
      objects are segmented with ``cluster`` and linked to frame ``t-1``.

    ``cluster`` is an ``EmbedSeg.utils.utils.Cluster`` / ``Cluster_3d`` object.
    Returns ``dict(labels (T, ...), lineage {child id: parent id}, info [...])``.
    """
    import torch

    frames = np.asarray(frames)
    T, nd = frames.shape[0], frames.ndim - 1
    n_sigma = nd if n_sigma is None else n_sigma
    cluster_kwargs = dict(cluster_kwargs or {})
    grid = (cluster.xym if nd == 2 else cluster.xyzm).cpu().numpy()
    steps = grid_steps(grid)
    n_seg = nd + n_sigma + 1
    spatial = frames.shape[1:]
    grid = grid[(slice(None),) + tuple(slice(0, s) for s in spatial)]

    norm_frames = np.stack([_normalize_frame(f, norm) for f in frames])
    labels = np.zeros(frames.shape, dtype=np.int32)
    centers = np.zeros(frames.shape, dtype=bool)
    lineage, info, next_id = {}, [], 1
    model = model.to(device).eval()

    for t in range(T):
        chans, seg_chans = [], []
        for j in range(t - n_frames + 1, t + 1):
            chans.append(norm_frames[j] if j >= 0 else np.zeros(spatial, np.float32))
        for j in range(t - n_frames + 1, t):
            if j >= 0 and prev_seg:
                seg_chans.append(prev_seg_channels(labels[j], centers[j], prev_seg, blob_sigma))
            else:
                seg_chans.append(np.zeros((len(prev_seg),) + spatial, np.float32))
        x = np.concatenate([np.stack(chans)] + seg_chans, 0).astype(np.float32)
        x = torch.from_numpy(x)[None].to(device)

        with torch.no_grad():
            out = model(_pad_to_multiple(x), heads="seg" if t == 0 else "all")
        out = out[(slice(None), slice(None)) + tuple(slice(0, s) for s in spatial)][0]
        instance = cluster.cluster(out, n_sigma=n_sigma, **cluster_kwargs)
        instance = instance.cpu().numpy().astype(np.int32) if hasattr(instance, "cpu") else np.asarray(instance)

        if t == 0:
            lab = np.zeros_like(instance)
            for i in [int(v) for v in np.unique(instance[instance > 0])]:
                lab[instance == i] = next_id
                next_id += 1
            info.append(dict(t=t))
        else:
            o = n_seg
            o_np = out.cpu().numpy()
            emb = np.tanh(o_np[o : o + nd]) + grid
            div_prob = 1.0 / (1.0 + np.exp(-o_np[o + nd + n_sigma]))
            res = associate_tracks(
                instance,
                embedding_to_index(emb, steps),
                labels[t - 1],
                division_prob=div_prob,
                next_id=next_id,
                min_vote=min_vote,
                max_dist=max_dist,
                div_threshold=div_threshold,
            )
            lab, next_id = res["labels"], res["next_id"]
            lineage.update(res["lineage"])
            info.append(dict(t=t, divisions=res["divisions"], conflicts=res["conflicts"]))
        labels[t] = lab
        centers[t] = object_centers(lab)
    return dict(labels=labels, lineage=lineage, info=info)
