import os

import numpy as np
import pytest

pytest.importorskip("torch")
tifffile = pytest.importorskip("tifffile")

from EmbedSeg.datasets.TrackingDataset import TrackingDataset
from EmbedSeg.utils.generate_crops import process_tracking, read_lineage
from EmbedSeg.utils.synthetic_tracking import make_sequence


def _write_sequence(root, seq):
    ims, insts = [], []
    os.makedirs(os.path.join(root, "raw"), exist_ok=True)
    for t, (im, inst) in enumerate(zip(seq["images"], seq["instances"])):
        ims.append(os.path.join(root, "raw", "im_{:03d}.tif".format(t)))
        insts.append(os.path.join(root, "raw", "mask_{:03d}.tif".format(t)))
        tifffile.imwrite(ims[-1], im)
        tifffile.imwrite(insts[-1], inst.astype(np.int32))
    return ims, insts


def test_read_lineage(tmp_path):
    f = tmp_path / "man_track.txt"
    f.write_text("1 0 5 0\n2 0 9 0\n3 4 9 1\n4 4 9 1\n")
    assert read_lineage(str(f)) == {3: 1, 4: 1}


@pytest.mark.parametrize("ndim,shape,crop", [(2, (40, 40), (24, 24)), (3, (16, 24, 24), (16, 16, 16))])
def test_process_tracking_roundtrip(tmp_path, ndim, shape, crop):
    seq = make_sequence(ndim=ndim, shape=shape, n_frames=5, radius=3.0, speed=1.0, divide=((3, 0),), seed=2)
    seq["instances"][:, ..., 0:2] = -1  # unlabelled border strip
    ims, insts = _write_sequence(str(tmp_path), seq)
    track_file = tmp_path / "man_track.txt"
    track_file.write_text("".join("{} 0 9 {}\n".format(c, p) for c, p in seq["lineage"].items()))
    kw = dict(crop_size_x=crop[-1], crop_size_y=crop[-2])
    if ndim == 3:
        kw["crop_size_z"] = crop[0]
    names = process_tracking(
        ims, insts, str(tmp_path / "crops"), "train",
        center="approximate-medoid", lineage=str(track_file), n_frames=2, name="seq", **kw
    )
    assert names
    ds = TrackingDataset(
        data_dir=str(tmp_path / "crops"), center="center-approximate-medoid",
        type="train", ndim=ndim, n_frames=2,
    )
    assert len(ds) == len(names)
    saw_division = False
    for i in range(len(ds)):
        s = ds[i]
        assert s["image"].shape[1:] == crop
        assert s["instance"].shape == (1,) + crop
        saw_division |= bool(s["division"].any())
        # every labelled object of frame t has exactly one centre pixel
        for g in np.unique(s["instance"][s["instance"] > 0]):
            assert (s["center_image"] & (s["instance"] == g)).sum() == 1
        # parents exist in the previous frame
        par = np.unique(s["parent"][s["parent"] > 0])
        assert set(par.tolist()) <= set(np.unique(s["instance_prev"]).tolist())
    assert saw_division
    # the ignore label survives cropping where the strip is inside a window
    masks = [tifffile.imread(str(tmp_path / "crops" / "train" / "masks" / n)) for n in names]
    assert any((m == -1).any() for m in masks)


def test_process_tracking_rejects_large_ids(tmp_path):
    seq = make_sequence(ndim=2, shape=(32, 32), n_frames=3, divide=())
    seq["instances"] = seq["instances"].astype(np.int32)
    seq["instances"][2][seq["instances"][2] > 0] += 40000
    ims, insts = _write_sequence(str(tmp_path), seq)
    with pytest.raises(ValueError):
        process_tracking(ims, insts, str(tmp_path / "c"), "train", 16, 16, "approximate-medoid")
