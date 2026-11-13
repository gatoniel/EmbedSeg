import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("tifffile")

from EmbedSeg.datasets.TrackingDataset import TrackingDataset
from EmbedSeg.utils.synthetic_tracking import make_sequence, write_tracking_samples
from EmbedSeg.utils.tracking import (
    blob_from_centers,
    build_parent_map,
    division_map,
    n_input_channels,
)


def test_parent_map_and_division_map():
    prev = np.zeros((10, 10), dtype=np.int16)
    prev[1:4, 1:4] = 1
    prev[6:9, 6:9] = 2
    cur = np.zeros_like(prev)
    cur[1:3, 1:4] = 5  # daughters of 1 (new ids)
    cur[3:4, 1:4] = 6
    cur[6:9, 6:9] = 2  # continues
    cur[0, 9] = 7  # new object without parent
    parent = build_parent_map(cur, prev, {5: 1, 6: 1})
    assert set(np.unique(parent[cur == 5])) == {1}
    assert set(np.unique(parent[cur == 6])) == {1}
    assert set(np.unique(parent[cur == 2])) == {2}
    assert set(np.unique(parent[cur == 7])) == {-1}  # no parent found: unknown
    assert set(np.unique(build_parent_map(cur, prev, {5: 1, 6: 1}, new_objects=True)[cur == 7])) == {0}
    assert not parent[cur == 0].any()  # background stays 0
    div = division_map(cur, parent)
    assert div[cur == 5].all() and div[cur == 6].all()
    assert not div[cur == 2].any() and not div[cur == 7].any()


def test_blob_peak_is_one_at_centre():
    c = np.zeros((21, 21), dtype=bool)
    c[10, 10] = True
    b = blob_from_centers(c, 2.0)
    assert b[10, 10] == pytest.approx(1.0, abs=1e-4)
    assert b[0, 0] < 1e-3


@pytest.mark.parametrize("ndim,K", [(2, 2), (2, 3), (3, 2)])
def test_dataset_shapes_and_alignment(tmp_path, ndim, K):
    shape = (32,) * ndim if ndim == 2 else (20,) * 3
    seq = make_sequence(ndim=ndim, shape=shape, n_frames=6, radius=3.0, divide=((3, 0),))
    write_tracking_samples(seq, str(tmp_path), "train", n_frames=K)
    ds = TrackingDataset(data_dir=str(tmp_path), type="train", ndim=ndim, n_frames=K)
    assert ds.input_channels == n_input_channels(K, ("mask", "blob"))
    n = len(ds)
    assert n == 6 - K + 1
    # the sample whose reference frame is the division frame (t=3)
    s = ds[3 - (K - 1)]
    assert s["image"].shape == (ds.input_channels,) + shape
    for k in ("instance", "label", "center_image", "instance_prev", "parent", "division"):
        assert s[k].shape == (1,) + shape, k
    assert bool(s["has_context"])
    assert s["division"].any(), "division frame must have daughter pixels"
    # parents point to ids that exist in frame t-1
    ids = np.unique(s["parent"][s["parent"] > 0])
    assert set(ids.tolist()) <= set(np.unique(s["instance_prev"]).tolist())
    # frame 1 (no division): every parent equals own id, no division flag
    s1 = ds[0]
    assert not s1["division"].any()
    m = s1["instance"] > 0
    assert (s1["parent"][m] == s1["instance"][m]).all()


def test_context_dropout_blanks_previous_frames(tmp_path):
    seq = make_sequence(ndim=2, shape=(32, 32), n_frames=5, divide=())
    write_tracking_samples(seq, str(tmp_path), "train", n_frames=2)
    ds = TrackingDataset(data_dir=str(tmp_path), type="train", ndim=2, n_frames=2, p_nocontext=1.0)
    s = ds[1]
    assert not bool(s["has_context"])
    # previous frame + its seg channels blanked, reference frame untouched
    assert np.all(s["image"][0] == 0) and np.all(s["image"][2:] == 0)
    assert s["image"][1].std() > 0
    assert not s["parent"].any() and not s["instance_prev"].any()
    assert s["instance"].any()  # segmentation target remains


def test_ignore_label_kept_in_reference_instance(tmp_path):
    seq = make_sequence(ndim=2, shape=(32, 32), n_frames=3, divide=())
    seq["instances"][2][:, 14:18] = -1
    write_tracking_samples(seq, str(tmp_path), "train", n_frames=2)
    ds = TrackingDataset(data_dir=str(tmp_path), type="train", ndim=2, n_frames=2)
    s = ds[1]
    assert (s["instance"][0][:, 14:18] == -1).all()
    assert not s["division"][0][:, 14:18].any()


def test_unknown_parent_is_marked_minus_one():
    prev = np.zeros((10, 10), dtype=np.int16)
    prev[1:4, 1:4] = 1
    cur = np.zeros_like(prev)
    cur[1:3, 1:4] = 5  # daughter of 1, visible parent
    cur[6:9, 6:9] = 8  # daughter of 99, which is not in this crop
    cur[0, 9] = 7  # genuinely new object
    parent = build_parent_map(cur, prev, {5: 1, 8: 99})
    assert set(np.unique(parent[cur == 5])) == {1}
    assert set(np.unique(parent[cur == 8])) == {-1}
    assert set(np.unique(parent[cur == 7])) == {-1}
    assert not division_map(cur, parent)[cur == 8].any()


def test_parent_hidden_by_ignore_label_is_unknown_in_dataset(tmp_path):
    seq = make_sequence(ndim=2, shape=(32, 32), n_frames=5, radius=3.0, divide=((3, 0),))
    pid = list(seq["lineage"].values())[0]
    seq["instances"][2][seq["instances"][2] == pid] = -1  # parent unlabelled in t-1
    write_tracking_samples(seq, str(tmp_path), "train", n_frames=2)
    ds = TrackingDataset(data_dir=str(tmp_path), type="train", ndim=2, n_frames=2)
    s = ds[2]  # reference frame t = 3 (division)
    daughters = [d for d, p in seq["lineage"].items() if p == pid]
    for d in daughters:
        assert (s["parent"][s["instance"] == d] == -1).all()
    assert not s["division"].any()  # no division label from unknown parents
    assert (s["instance_prev"] == -1).any()
