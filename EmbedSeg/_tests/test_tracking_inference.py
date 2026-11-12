import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tifffile")

from EmbedSeg.utils.synthetic_tracking import make_sequence
from EmbedSeg.utils.tracking import build_parent_map, division_map
from EmbedSeg.utils.tracking_inference import (
    associate_tracks,
    embedding_to_index,
    grid_steps,
    object_centers,
    track_sequence,
)


def _idx(inst, target):
    """per-pixel predicted index map pointing to ``target`` (y, x) for every pixel"""
    out = np.zeros((2,) + inst.shape, dtype=float)
    out[0], out[1] = target
    return out


def test_associate_continue_division_new_and_ids():
    prev = np.zeros((40, 40), dtype=np.int32)
    prev[5:12, 5:12] = 3  # will divide
    prev[25:32, 25:32] = 4  # continues
    cur = np.zeros_like(prev)
    cur[4:8, 5:12] = 1  # daughter a
    cur[9:13, 5:12] = 2  # daughter b
    cur[26:33, 26:33] = 5  # continues
    cur[35:38, 2:5] = 6  # new object
    idx = np.zeros((2,) + cur.shape)
    idx[:, cur == 1] = np.array([8, 8]).reshape(2, 1)
    idx[:, cur == 2] = np.array([8, 8]).reshape(2, 1)
    idx[:, cur == 5] = np.array([28, 28]).reshape(2, 1)
    idx[:, cur == 6] = np.array([20, 20]).reshape(2, 1)  # points to background
    prob = np.zeros(cur.shape)
    prob[(cur == 1) | (cur == 2)] = 0.9
    res = associate_tracks(cur, idx, prev, division_prob=prob, next_id=10)
    lab = res["labels"]
    assert (lab[cur == 5] == 4).all()  # inherits
    assert lab[cur == 1][0] != lab[cur == 2][0]
    assert {lab[cur == 1][0], lab[cur == 2][0]} == {10, 11}
    assert res["lineage"] == {10: 3, 11: 3}
    assert res["divisions"] == [(3, [10, 11])]
    assert lab[cur == 6][0] == 12  # new track
    assert res["next_id"] == 13
    assert res["conflicts"] == []


def test_associate_reports_flag_conflicts_and_tolerates_small_offsets():
    prev = np.zeros((30, 30), dtype=np.int32)
    prev[10:20, 10:20] = 7
    cur = np.zeros_like(prev)
    cur[11:19, 11:19] = 1
    idx = np.zeros((2,) + cur.shape)
    idx[:, cur == 1] = np.array([21, 15]).reshape(2, 1)  # 2 px outside parent
    prob = np.where(cur == 1, 0.95, 0.0)
    res = associate_tracks(cur, idx, prev, division_prob=prob, max_dist=2.0)
    assert (res["labels"][cur == 1] == 7).all()  # linked within max_dist
    assert res["conflicts"] == [("flag_without_division", 7)]
    res0 = associate_tracks(cur, idx, prev, max_dist=0)
    assert (res0["labels"][cur == 1] == 8).all()  # not linked -> new id


def test_object_centers_one_pixel_per_object_inside():
    lab = np.zeros((20, 20), dtype=np.int32)
    lab[2:9, 2:9] = 1
    lab[12:18, 4:10] = 2
    c = object_centers(lab)
    assert c.sum() == 2
    assert (lab[c] > 0).all() and set(lab[c]) == {1, 2}


class OracleModel(torch.nn.Module):
    """Returns the ideal prediction for a ground-truth sequence, one frame per call."""

    def __init__(self, seq, grid):
        super().__init__()
        self.seq, self.grid, self.t = seq, grid, 0
        self.heads_called = []

    def forward(self, x, heads="all"):
        self.heads_called.append(heads)
        t = self.t
        self.t += 1
        inst = self.seq["instances"][t]
        H, W = inst.shape
        grid = self.grid[:, :H, :W]
        cen = object_centers(inst)
        out = torch.zeros(1, 10 if heads == "all" else 5, H, W)
        out[0, 4] = -10.0
        out[0, 2:4] = 1.0
        for g in np.unique(inst[inst > 0]):
            m = torch.from_numpy(inst == g)
            ys, xs = np.nonzero(cen & (inst == g))
            c = grid[:, ys[0], xs[0]]
            out[0, 0:2][:, m] = torch.atanh((c.view(2, 1) - grid[:, m]).clamp(-0.99, 0.99))
            out[0, 4][m] = 10.0
        if heads == "all":
            prev = self.seq["instances"][t - 1]
            parent = build_parent_map(inst, prev, self.seq["lineage"])
            div = division_map(inst, parent)
            cen_prev = object_centers(prev)
            out[0, 7:9] = 1.0
            out[0, 9] = -10.0
            for p in np.unique(parent[parent > 0]):
                m = torch.from_numpy(parent == p)
                ys, xs = np.nonzero(cen_prev & (prev == p))
                c = grid[:, ys[0], xs[0]]
                out[0, 5:7][:, m] = torch.atanh((c.view(2, 1) - grid[:, m]).clamp(-0.99, 0.99))
            out[0, 9][torch.from_numpy(div > 0)] = 10.0
        return out


def test_grid_steps_and_embedding_roundtrip():
    from EmbedSeg.utils.utils import Cluster_3d

    cl = Cluster_3d(8, 16, 24, 0.5, 1.0, 1.0, "cpu")
    steps = grid_steps(cl.xyzm.numpy())
    assert np.allclose(steps, [1.0 / 23, 1.0 / 15, 0.5 / 7])
    g = cl.xyzm.numpy()
    idx = embedding_to_index(g, steps)  # (z, y, x) index of every voxel
    zz, yy, xx = np.meshgrid(np.arange(8), np.arange(16), np.arange(24), indexing="ij")
    assert np.allclose(idx, np.stack([zz, yy, xx]))


def test_track_sequence_end_to_end_with_oracle():
    from EmbedSeg.utils.utils import Cluster

    seq = make_sequence(
        ndim=2, shape=(48, 48), n_frames=6, n_objects=3, radius=5.0, speed=1.5, divide=((3, 0),), seed=1
    )
    cluster = Cluster(48, 48, 1.0, 1.0, "cpu")
    model = OracleModel(seq, cluster.xym)
    res = track_sequence(
        model,
        seq["images"],
        cluster,
        n_frames=2,
        n_sigma=2,
        cluster_kwargs=dict(min_object_size=10),
    )
    # t = 0 used only the segmentation heads, all later frames the full network
    assert model.heads_called == ["seg"] + ["all"] * 5
    pred, gt = res["labels"], seq["instances"]
    for t in range(6):
        assert len(np.unique(pred[t][pred[t] > 0])) == len(np.unique(gt[t][gt[t] > 0]))
    # map gt track id -> predicted id per frame; continuing tracks keep their id
    def mapping(t):
        return {int(g): int(np.bincount(pred[t][gt[t] == g]).argmax()) for g in np.unique(gt[t][gt[t] > 0])}

    maps = [mapping(t) for t in range(6)]
    for t in range(1, 6):
        for g, p in maps[t].items():
            if g in maps[t - 1]:
                assert maps[t - 1][g] == p, "track id changed without division"
    # the division at t = 3: two daughters with new ids and the same predicted parent
    gt_daughters = [d for d, par in seq["lineage"].items()]
    assert len(gt_daughters) == 2
    parent_gt = seq["lineage"][gt_daughters[0]]
    assert res["lineage"] == {maps[3][d]: maps[2][parent_gt] for d in gt_daughters}
    assert len({maps[3][d] for d in gt_daughters}) == 2
    assert [i["divisions"] for i in res["info"][1:] if i["t"] != 3 and i["divisions"]] == []
    assert all(i["conflicts"] == [] for i in res["info"][1:])
