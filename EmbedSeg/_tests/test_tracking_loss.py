import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tifffile")

from EmbedSeg.criterions import get_loss
from EmbedSeg.criterions.my_loss import SpatialEmbLoss
from EmbedSeg.criterions.my_loss_3d import SpatialEmbLoss_3d
from EmbedSeg.criterions.my_loss_tracking import SpatialEmbTrackingLoss
from EmbedSeg.datasets.TrackingDataset import TrackingDataset
from EmbedSeg.utils.synthetic_tracking import make_sequence, write_tracking_samples


def _batch(tmp_path, ndim, frame_idx, shape, n_frames=2, ignore_stripe=False, hide_parent=False, **ds_kw):
    seq = make_sequence(ndim=ndim, shape=shape, n_frames=5, radius=3.0, divide=((3, 0),))
    if hide_parent:  # the dividing parent is unlabelled (-1) in frame t-1
        pid = list(seq['lineage'].values())[0]
        seq['instances'][2][seq['instances'][2] == pid] = -1
    if ignore_stripe:
        seq["instances"][:, ..., 14:16] = -1
    write_tracking_samples(seq, str(tmp_path), "train", n_frames=n_frames)
    ds = TrackingDataset(data_dir=str(tmp_path), type="train", ndim=ndim, n_frames=n_frames, **ds_kw)
    s = ds[frame_idx]
    t = lambda k: torch.from_numpy(np.asarray(s[k]))[None]  # noqa: E731
    return dict(
        instances=t("instance")[:, 0],
        labels=t("label")[:, 0],
        center_images=t("center_image")[:, 0],
        instances_prev=t("instance_prev")[:, 0],
        center_images_prev=t("center_image_prev")[:, 0],
        parents=t("parent")[:, 0],
        divisions=t("division")[:, 0],
        has_context=t("has_context"),
    )


def _loss(ndim, shape):
    kw = dict(grid_y=shape[-2], grid_x=shape[-1], pixel_y=1, pixel_x=1)
    if ndim == 3:
        kw.update(grid_z=shape[0], pixel_z=1)
    return SpatialEmbTrackingLoss(**kw)


@pytest.mark.parametrize("ndim,shape", [(2, (32, 32)), (3, (16, 16, 16))])
def test_loss_finite_and_gradients_reach_tracking_head(tmp_path, ndim, shape):
    b = _batch(tmp_path, ndim, 2, shape)  # reference frame t=3: division
    crit = _loss(ndim, shape)
    C = 2 * (2 * ndim + 1) if ndim == 3 else 2 * (2 * 2 + 1)
    torch.manual_seed(0)
    pred = (0.1 * torch.randn(1, C, *shape)).requires_grad_()
    loss = crit(pred, **b)
    assert torch.isfinite(loss)
    loss.backward()
    o = crit.n_seg
    assert pred.grad[:, :o].abs().sum() > 0
    assert pred.grad[:, o:].abs().sum() > 0
    for k in ("track", "var", "reg", "div"):
        assert k in crit.last_terms


def test_perfect_tracking_prediction_has_lower_loss(tmp_path):
    shape = (32, 32)
    b = _batch(tmp_path, 2, 2, shape)  # division frame
    crit = _loss(2, shape)
    nd, ns, o = 2, 2, crit.n_seg
    grid = crit.grid[:, :32, :32]
    parent = b["parents"][0]
    cen_prev = b["center_images_prev"][0]
    inst_prev = b["instances_prev"][0]

    def predict(perfect):
        pred = torch.zeros(1, 2 * o, *shape)
        pred[:, o + nd : o + nd + ns] = 1.0  # sigma
        pred[:, o + nd + ns] = -10.0
        if perfect:
            for p in parent.unique():
                if p == 0:
                    continue
                grp = parent == p
                m = (inst_prev == p) & cen_prev
                ys, xs = torch.nonzero(m, as_tuple=True)
                center = grid[:, ys[0], xs[0]]  # x, y
                off = (center.view(2, 1) - grid[:, grp]).clamp(-0.99, 0.99)
                pred[0, o : o + nd][:, grp] = torch.atanh(off)
            pred[0, o + nd + ns][b["divisions"][0] > 0] = 10.0
        return pred

    crit(predict(True), **b)
    good = dict(crit.last_terms)
    crit(predict(False), **b)
    bad = dict(crit.last_terms)
    assert good["reg"] < 1e-4 < bad["reg"]
    assert good["track"] < bad["track"]
    assert good["div"] < bad["div"]


def test_no_context_samples_get_no_tracking_gradient(tmp_path):
    shape = (32, 32)
    b = _batch(tmp_path, 2, 2, shape, p_nocontext=1.0)
    assert not bool(b["has_context"][0])
    crit = _loss(2, shape)
    pred = (0.1 * torch.randn(1, 2 * crit.n_seg, *shape)).requires_grad_()
    crit(pred, **b).backward()
    assert pred.grad[:, crit.n_seg:].abs().sum() == 0
    assert pred.grad[:, :crit.n_seg].abs().sum() > 0


def test_ignored_pixels_get_no_tracking_gradient(tmp_path):
    shape = (32, 32)
    b = _batch(tmp_path, 2, 1, shape, ignore_stripe=True)
    assert (b["instances"][0][:, 14:16] == -1).all()
    crit = _loss(2, shape)
    pred = (0.1 * torch.randn(1, 2 * crit.n_seg, *shape)).requires_grad_()
    crit(pred, **b).backward()
    g = pred.grad[0, crit.n_seg:, :, 14:16]
    assert g.abs().sum() == 0


@pytest.mark.parametrize("ndim,shape", [(2, (32, 32)), (3, (16, 16, 16))])
def test_without_tracking_head_equals_plain_loss(tmp_path, ndim, shape):
    b = _batch(tmp_path, ndim, 1, shape)
    crit = _loss(ndim, shape)
    base_cls = SpatialEmbLoss if ndim == 2 else SpatialEmbLoss_3d
    kw = dict(grid_y=shape[-2], grid_x=shape[-1], pixel_y=1, pixel_x=1)
    if ndim == 3:
        kw.update(grid_z=shape[0], pixel_z=1)
    base = base_cls(n_sigma=ndim, **kw)
    pred = 0.1 * torch.randn(1, crit.n_seg, *shape)
    a = crit(pred, b["instances"], b["labels"], b["center_images"])
    ref = base(pred, b["instances"], b["labels"], b["center_images"])
    assert torch.allclose(a, ref)


def test_get_loss_is_opt_in():
    plain = get_loss(None, 16, 16, None, 1, 1, False, {"n_sigma": 2, "foreground_weight": 1})
    assert isinstance(plain, SpatialEmbLoss)
    trk = get_loss(
        None, 16, 16, None, 1, 1, False,
        {"n_sigma": 2, "foreground_weight": 1, "tracking": True},
    )
    assert isinstance(trk, SpatialEmbTrackingLoss)


def test_unknown_parent_pixels_get_no_division_gradient_and_no_metrics(tmp_path):
    shape = (32, 32)
    b = _batch(tmp_path, 2, 2, shape, hide_parent=True)  # division frame, parent hidden
    unknown = (b["parents"][0] < 0) & (b["instances"][0] > 0)
    known = (b["parents"][0] > 0) & (b["instances"][0] > 0)
    assert unknown.any() and known.any()
    crit = _loss(2, shape)
    nd, ns = 2, 2
    pred = (0.1 * torch.randn(1, 2 * crit.n_seg, *shape)).requires_grad_()
    from EmbedSeg.utils.utils import AverageMeter

    crit(pred, **b, iou=True, iou_meter=AverageMeter()).backward()
    div_grad = pred.grad[0, crit.n_seg + nd + ns]
    assert div_grad[unknown].abs().sum() == 0
    assert div_grad[known].abs().sum() > 0
    m = crit.report_tracking_metrics()
    n_known_objects = len(np.unique(b["instances"][0][known].numpy()))
    assert m["n_objects"] == n_known_objects  # objects with unknown parent are not scored
