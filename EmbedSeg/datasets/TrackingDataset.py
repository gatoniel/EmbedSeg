import glob
import os
import random

import numpy as np
import tifffile
from torch.utils.data import Dataset

from EmbedSeg.utils.tracking import (
    build_parent_map,
    division_map,
    n_input_channels,
    perturb_prev_instance,
    prev_seg_channels,
    sample_context_dropout,
)


def _normalize(x, norm, axis, data_type="8-bit"):
    """Per-channel normalisation (``axis`` = spatial axes)."""
    x = x.astype(np.float32)
    if norm == "min-max-percentile":
        mi = np.percentile(x, 1, axis=axis, keepdims=True)
        ma = np.percentile(x, 99.8, axis=axis, keepdims=True)
        return (x - mi) / (ma - mi + 1e-20)
    if norm == "mean-std":
        return (x - x.mean(axis=axis, keepdims=True)) / (
            x.std(axis=axis, keepdims=True) + 1e-20
        )
    if norm == "absolute":
        return x / (255.0 if data_type == "8-bit" else 65535.0)
    return x


class TrackingDataset(Dataset):
    """
    Dataset for 2D+t / 3D+t tracking. One sample is a window of ``n_frames``
    frames (see :mod:`EmbedSeg.utils.tracking` for the on-disk layout).

    Returned keys (all channel-first, spatial dims are ``YX`` or ``ZYX``)

    * ``image``: ``(C, ...)``, ``C = n_frames + (n_frames-1)*len(prev_seg)``: the
      ``n_frames`` intensity frames (oldest first), then for each previous frame
      (oldest first) one channel per entry of ``prev_seg``
    * ``instance``, ``label``, ``center_image``: reference frame ``t``, ``(1, ...)``
    * ``instance_prev``, ``center_image_prev``: frame ``t-1``
    * ``parent``: id in frame ``t-1`` of the parent of each pixel of ``t`` (0: none)
    * ``division``: 1 on pixels of objects that share their parent with another one
    * ``has_context``: False if frame ``t-1`` was blanked (tracking loss is off)

    During training, *context dropout* (``p_nocontext``/``p_leading``) blanks the
    oldest frames so that the network also learns to segment without context
    (start of a sequence), and ``perturb_prev`` imitates errors of a predicted
    previous segmentation.
    """

    def __init__(
        self,
        data_dir="./",
        center="center-medoid",
        type="train",
        ndim=2,
        n_frames=2,
        prev_seg=("mask", "blob"),
        blob_sigma=2.0,
        bg_id=0,
        size=None,
        transform=None,
        norm="min-max-percentile",
        data_type="8-bit",
        normalization=False,
        p_nocontext=0.0,
        p_leading=0.0,
        perturb_prev=False,
        **kwargs,
    ):
        assert ndim in (2, 3)
        assert n_frames >= 2
        print(
            "{}-D tracking `{}` dataloader created! Accessing data from {}/{}/".format(
                ndim, type, data_dir, type
            )
        )
        root = os.path.join(data_dir, type)
        self.image_list = sorted(glob.glob(os.path.join(root, "images", "*.tif")))
        self.instance_list = sorted(glob.glob(os.path.join(root, "masks", "*.tif")))
        self.center_image_list = sorted(glob.glob(os.path.join(root, center, "*.tif")))
        print("Number of samples in `{}` directory is {}".format(type, len(self.image_list)))
        assert len(self.instance_list) in (0, len(self.image_list))
        assert len(self.center_image_list) in (0, len(self.image_list))

        self.ndim = ndim
        self.n_frames = n_frames
        self.prev_seg = tuple(prev_seg)
        self.blob_sigma = blob_sigma
        self.bg_id = bg_id
        self.size = size
        self.real_size = len(self.image_list)
        self.transform = transform
        self.norm = norm
        self.data_type = data_type
        self.normalization = normalization
        self.p_nocontext = p_nocontext
        self.p_leading = p_leading
        self.perturb_prev = perturb_prev

    @property
    def input_channels(self):
        return n_input_channels(self.n_frames, self.prev_seg)

    def __len__(self):
        return self.real_size if self.size is None else self.size

    def __getitem__(self, index):
        index = index if self.size is None else random.randint(0, self.real_size - 1)
        K = self.n_frames
        sp_axes = tuple(range(1, self.ndim + 1))
        rng = np.random.default_rng()
        sample = {"im_name": self.image_list[index]}

        image = tifffile.imread(self.image_list[index])
        assert image.shape[0] == K and image.ndim == self.ndim + 1, (
            "expected {} frames of a {}-D image, got shape {}".format(K, self.ndim, image.shape)
        )
        if self.normalization:
            image = _normalize(image, self.norm, sp_axes, self.data_type)
        image = image.astype(np.float32)

        has_labels = len(self.instance_list) != 0
        if has_labels:
            masks = tifffile.imread(self.instance_list[index]).astype(np.int32)
            assert masks.shape[0] == K + 1, "masks need {} channels, got {}".format(
                K + 1, masks.shape
            )
            instances, parent = masks[:K], masks[K]
            if masks.max() > np.iinfo(np.int16).max:
                raise ValueError("instance ids above 32767 are not supported (int16)")
            centers = tifffile.imread(self.center_image_list[index]).astype(bool)
            assert centers.shape[0] == K

            inst_t = instances[-1].copy()
            inst_prev = instances[-2].copy()
            # parent: >0 id in frame t-1, 0 no parent, -1 parent unknown (see
            # utils.tracking.build_parent_map); ids missing from the previous
            # (unperturbed) frame cannot be used and become unknown as well
            valid = np.isin(parent, np.unique(inst_prev[inst_prev > 0]))
            parent = np.where(parent > 0, np.where(valid, parent, -1), np.minimum(parent, 0))
            parent = parent.astype(np.int16)
            label = (inst_t > self.bg_id).astype(np.uint8)
            inst_t_out = inst_t.astype(np.int16)

        # context dropout: blank the oldest frames
        n_blank = sample_context_dropout(K, rng, self.p_nocontext, self.p_leading)

        # previous-segmentation channels for frames t-K+1 .. t-1
        seg_blocks = []
        if has_labels and self.prev_seg:
            for j in range(K - 1):
                inst_j, cen_j = instances[j], centers[j]
                if self.perturb_prev:
                    inst_j, cen_j = perturb_prev_instance(inst_j, cen_j, rng)
                seg_blocks.append(
                    prev_seg_channels(inst_j, cen_j, self.prev_seg, self.blob_sigma)
                )
        elif self.prev_seg:  # no labels available: caller must provide them (inference)
            seg_blocks = [
                np.zeros((len(self.prev_seg),) + image.shape[1:], dtype=np.float32)
                for _ in range(K - 1)
            ]
        for j in range(n_blank):
            image[j] = 0
            if seg_blocks:
                seg_blocks[j] = np.zeros_like(seg_blocks[j])
        sample["image"] = np.concatenate([image] + seg_blocks, axis=0)
        sample["has_context"] = np.array(n_blank < K - 1)

        if has_labels:
            if n_blank >= K - 1:  # no previous frame -> no tracking targets
                parent = np.zeros_like(parent)
                inst_prev_out = np.zeros_like(inst_prev)
                cen_prev_out = np.zeros_like(centers[-2])
            else:
                inst_prev_out, cen_prev_out = inst_prev, centers[-2]
            sample["instance"] = inst_t_out[np.newaxis]
            sample["label"] = label[np.newaxis]
            sample["center_image"] = centers[-1][np.newaxis]
            sample["instance_prev"] = inst_prev_out.astype(np.int16)[np.newaxis]
            sample["center_image_prev"] = cen_prev_out[np.newaxis]
            sample["parent"] = parent[np.newaxis]
            sample["division"] = division_map(inst_t, parent)[np.newaxis]

        if self.transform is not None:
            return self.transform(sample)
        return sample
