"""
Helpers for time-lapse tracking with EmbedSeg (works for 2D+t and 3D+t).

Conventions
-----------
A training sample covers a window of ``K`` frames ``[t-K+1, ..., t]``; ``t`` is the
reference frame that is segmented. All frames of a sample are cropped with the
same window. On disk one sample consists of

* ``images/<name>.tif``:  ``K``   x ([Z,] Y, X)   intensities
* ``masks/<name>.tif``:   ``K+1`` x ([Z,] Y, X)   track-consistent instance ids for
  the K frames (-1 = ignore) followed by a ``parent`` channel: for each pixel of
  frame ``t`` the id (in frame ``t-1``) of the parent object, 0 if none
* ``center-<kind>/<name>.tif``: ``K`` x ([Z,] Y, X)   boolean centre images

The network input stacks ``K`` intensity channels followed by ``K-1`` blocks of
"previous segmentation" channels (one channel per entry of ``prev_seg``).
"""
import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter

PREV_SEG_KINDS = ("mask", "blob")


def n_input_channels(n_frames, prev_seg=()):
    """Number of network input channels for a window of ``n_frames`` frames."""
    return n_frames + (n_frames - 1) * len(prev_seg)


def blob_from_centers(center_image, sigma=2.0):
    """Gaussian blob (peak value 1) around every True pixel of ``center_image``."""
    center_image = np.asarray(center_image, dtype=bool)
    if not center_image.any():
        return np.zeros(center_image.shape, dtype=np.float32)
    blob = gaussian_filter(center_image.astype(np.float32), sigma, mode="constant")
    peak = (1.0 / (np.sqrt(2 * np.pi) * sigma)) ** center_image.ndim
    return np.clip(blob / peak, 0, 1).astype(np.float32)


def prev_seg_channels(instance, center_image, prev_seg=("mask", "blob"), blob_sigma=2.0):
    """Channels that describe an already segmented frame.

    ``instance`` holds ids (>0 objects). Returns an array ``(len(prev_seg), ...)``.
    """
    channels = []
    for kind in prev_seg:
        if kind == "mask":
            channels.append((instance > 0).astype(np.float32))
        elif kind == "blob":
            channels.append(blob_from_centers(center_image, blob_sigma))
        else:
            raise ValueError("unknown prev_seg kind {!r}, use {}".format(kind, PREV_SEG_KINDS))
    if not channels:
        return np.zeros((0,) + instance.shape, dtype=np.float32)
    return np.stack(channels)


def perturb_prev_instance(instance, center_image, rng, p_drop=0.05, p_erode=0.1):
    """Imitate errors of a predicted previous segmentation (exposure bias):
    randomly drop objects and erode others. Centre pixels of dropped objects are
    removed as well. Returns new ``(instance, center_image)``."""
    instance = instance.copy()
    center_image = center_image.copy()
    for i in np.unique(instance[instance > 0]):
        r = rng.random()
        if r < p_drop:
            m = instance == i
            instance[m] = 0
            center_image[m] = False
        elif r < p_drop + p_erode:
            m = instance == i
            eroded = binary_erosion(m)
            if eroded.any():
                instance[m & ~eroded] = 0
    return instance, center_image


def build_parent_map(instance_t, instance_prev, lineage=None):
    """For every pixel of frame t the id of the parent object in frame t-1.

    An object whose id also exists in frame t-1 continues that track (parent = id).
    Otherwise ``lineage[id]`` (child id -> parent id, e.g. from a Cell Tracking
    Challenge ``man_track.txt``) is used if that parent is present in frame t-1.
    Objects without a resolvable parent get 0.
    """
    lineage = lineage or {}
    prev_ids = set(np.unique(instance_prev[instance_prev > 0]).tolist())
    parent = np.zeros(instance_t.shape, dtype=np.int16)
    for i in np.unique(instance_t[instance_t > 0]):
        i = int(i)
        if i in prev_ids:
            p = i
        else:
            p = int(lineage.get(i, 0))
            if p not in prev_ids:
                p = 0
        if p:
            parent[instance_t == i] = p
    return parent


def division_map(instance_t, parent):
    """Pixels of objects in t that share their parent with another object in t."""
    out = np.zeros(instance_t.shape, dtype=np.uint8)
    children = {}
    for i in np.unique(instance_t[instance_t > 0]):
        m = instance_t == i
        p = int(parent[m].max())
        if p > 0:
            children.setdefault(p, []).append(m)
    for masks in children.values():
        if len(masks) >= 2:
            for m in masks:
                out[m] = 1
    return out


def sample_context_dropout(n_frames, rng, p_nocontext=0.2, p_leading=0.2):
    """How many of the *oldest* frames to zero out for one training sample.

    * with prob ``p_nocontext`` all ``n_frames - 1`` previous frames (sequence start,
      segmentation only, no tracking loss);
    * else with prob ``p_leading`` (only if ``n_frames > 2``) between 1 and
      ``n_frames - 2`` leading frames (t-1 stays available);
    * else 0.
    """
    r = rng.random()
    if r < p_nocontext:
        return n_frames - 1
    if n_frames > 2 and r < p_nocontext + p_leading:
        return int(rng.integers(1, n_frames - 1))
    return 0
