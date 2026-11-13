import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tifffile")

from EmbedSeg.criterions import get_loss
from EmbedSeg.datasets import get_dataset
from EmbedSeg.models import get_model
from EmbedSeg.train import _tracking_kwargs
from EmbedSeg.utils.create_dicts import (
    create_loss_dict,
    create_model_dict,
    create_tracking_dataset_dict,
    create_tracking_model_dict,
)
from EmbedSeg.utils.synthetic_tracking import make_sequence, write_tracking_samples


def test_default_dicts_unchanged_for_segmentation():
    ld = create_loss_dict(n_sigma=3)
    assert set(ld["lossOpts"]) == {"n_sigma", "foreground_weight"}
    assert set(ld["lossW"]) == {"w_inst", "w_var", "w_seed"}
    assert create_model_dict(1, [6, 1], name="3d")["name"] == "branched_erfnet_3d"
    assert create_model_dict(1, [4, 1], name="2d")["name"] == "branched_erfnet"


def test_tracking_dicts():
    md = create_tracking_model_dict(n_frames=3, prev_seg=("mask", "blob"), name="3d_tracking")
    assert md["name"] == "branched_erfnet_3d"
    assert md["kwargs"] == {"num_classes": [6, 1, 7], "input_channels": 3 + 2 * 2}
    md2 = create_tracking_model_dict(n_frames=2, prev_seg=(), name="2d_tracking")
    assert md2["name"] == "branched_erfnet"
    assert md2["kwargs"] == {"num_classes": [4, 1, 5], "input_channels": 2}
    ld = create_loss_dict(n_sigma=2, tracking=True, w_div=2)
    assert ld["lossOpts"]["tracking"] and ld["lossW"]["w_div"] == 2
    assert {"w_track", "w_track_reg", "w_div"} <= set(ld["lossW"])


def test_dataloader_to_loss_wiring_2d(tmp_path):
    shape = (32, 32)
    seq = make_sequence(ndim=2, shape=shape, n_frames=6, radius=3.0, divide=((3, 0),))
    for split in ("train", "val"):
        write_tracking_samples(seq, str(tmp_path / "proj"), split, n_frames=2)
    dd = create_tracking_dataset_dict(
        str(tmp_path), "proj", size=None, center="medoid", type="train", batch_size=3, workers=0
    )
    vd = create_tracking_dataset_dict(
        str(tmp_path), "proj", size=None, center="medoid", type="val", batch_size=3, workers=0
    )
    assert vd["kwargs"]["p_nocontext"] == 0 and not vd["kwargs"]["perturb_prev"]
    assert dd["kwargs"]["p_nocontext"] > 0

    md = create_tracking_model_dict(name="2d_tracking")
    ld = create_loss_dict(n_sigma=2, tracking=True)
    model = get_model(md["name"], md["kwargs"])
    model.init_output(2)
    crit = get_loss(None, 32, 32, None, 1, 1, False, ld["lossOpts"])

    # training batch through the real DataLoader + transforms
    ds = get_dataset(dd["name"], dd["kwargs"])
    batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=3, shuffle=False, num_workers=0)))
    kw = _tracking_kwargs(batch, "cpu")
    assert kw["has_context"].shape == (3,) and kw["has_context"].dtype == torch.bool
    out = model(batch["image"])
    assert out.shape[1] == 10
    loss = crit(
        out,
        batch["instance"].squeeze(1),
        batch["label"].squeeze(1),
        batch["center_image"].squeeze(1),
        **ld["lossW"],
        **kw,
    )
    assert torch.isfinite(loss)
    loss.backward()  # a single backward pass, no optimisation
    tracking_decoder = model.decoders[2]
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in tracking_decoder.parameters())

    # validation: metrics are collected when iou=True
    vds = get_dataset(vd["name"], vd["kwargs"])
    vb = next(iter(torch.utils.data.DataLoader(vds, batch_size=3, shuffle=False, num_workers=0)))
    from EmbedSeg.utils.utils import AverageMeter

    with torch.no_grad():
        crit(
            model(vb["image"]),
            vb["instance"].squeeze(1),
            vb["label"].squeeze(1),
            vb["center_image"].squeeze(1),
            **ld["lossW"],
            **_tracking_kwargs(vb, "cpu"),
            iou=True,
            iou_meter=AverageMeter(),
        )
    m = crit.report_tracking_metrics(reset=True)
    assert m["n_objects"] > 0 and 0.0 <= m["assoc_acc"] <= 1.0 and 0.0 <= m["div_f1"] <= 1.0
    assert crit.report_tracking_metrics()["n_objects"] == 0  # reset


def test_segmentation_batch_has_no_tracking_kwargs():
    assert _tracking_kwargs({"image": 1, "instance": 2}, "cpu") == {}
