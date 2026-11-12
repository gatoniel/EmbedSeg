import numpy as np
import pytest

torch = pytest.importorskip("torch")

from EmbedSeg.criterions.my_loss import SpatialEmbLoss
from EmbedSeg.datasets.TwoDimensionalDataset import TwoDimensionalDataset
from EmbedSeg.utils.generate_crops import fill_label_holes


def _sample(size=32):
    instance = np.zeros((size, size), dtype=np.int16)
    instance[4:12, 4:12] = 1
    instance[20:28, 20:28] = 2
    ignore = np.zeros_like(instance, dtype=bool)
    ignore[:, 14:18] = True  # unlabelled stripe, no object inside
    instance[ignore] = -1
    center = np.zeros((size, size), dtype=bool)
    center[8, 8] = True
    center[24, 24] = True
    return instance, center, ignore


def test_decode_instance_keeps_ignore():
    pic = np.zeros((16, 16), dtype=np.int32)
    pic[2:6, 2:6] = 7
    pic[8:, :] = -1
    instance, label = TwoDimensionalDataset.decode_instance(pic, False, bg_id=0)
    assert (instance[8:] == -1).all()
    assert (instance[2:6, 2:6] == 1).all()
    assert (label[8:] == 0).all() and (label[2:6, 2:6] == 1).all()


def test_fill_label_holes_keeps_ignore():
    lbl = np.zeros((20, 20), dtype=np.int32)
    lbl[2:12, 2:12] = 1
    lbl[5, 5] = 0  # hole to fill
    lbl[15:, :] = -1
    filled = fill_label_holes(lbl)
    assert filled[5, 5] == 1
    assert (filled[15:] == -1).all()
    assert (filled >= -1).all() and filled.max() == 1


def test_loss_ignores_unlabelled_pixels():
    instance, center, ignore = _sample()
    label = (instance > 0).astype(np.uint8)
    loss_fn = SpatialEmbLoss(grid_y=32, grid_x=32, pixel_y=1, pixel_x=1)

    prediction = torch.randn(1, 5, 32, 32, requires_grad=True)
    loss = loss_fn(
        prediction,
        torch.from_numpy(instance)[None],
        torch.from_numpy(label)[None],
        torch.from_numpy(center)[None],
    )
    loss.backward()
    assert torch.isfinite(loss)
    # the sigma / seed outputs at ignored pixels get no gradient from the
    # seed loss and are not used as instance members
    seed_grad = prediction.grad[0, 4][torch.from_numpy(ignore)]
    assert torch.all(seed_grad == 0)


def test_loss_without_ignore_unchanged_by_label_change():
    # same image, but stripe labelled as background instead of ignored, must differ
    instance, center, ignore = _sample()
    inst_bg = instance.copy()
    inst_bg[ignore] = 0
    label = (instance > 0).astype(np.uint8)
    loss_fn = SpatialEmbLoss(grid_y=32, grid_x=32, pixel_y=1, pixel_x=1)
    prediction = torch.randn(1, 5, 32, 32)
    args = (torch.from_numpy(label)[None], torch.from_numpy(center)[None])
    l_ignore = loss_fn(prediction, torch.from_numpy(instance)[None], *args)
    l_bg = loss_fn(prediction, torch.from_numpy(inst_bg)[None], *args)
    assert not torch.isclose(l_ignore, l_bg)
