import pytest

torch = pytest.importorskip("torch")

from EmbedSeg.criterions.my_loss_3d import SpatialEmbLoss_3d  # noqa: E402


def _data(batch=2, shape=(12, 16, 16), n_obj=6, extra_channels=0, seed=0):
    g = torch.Generator().manual_seed(seed)
    z, y, x = shape
    instances = torch.zeros(batch, z, y, x, dtype=torch.long)
    centers = torch.zeros(batch, z, y, x, dtype=torch.bool)
    for b in range(batch):
        for k in range(1, n_obj + 1):
            cz, cy, cx = (int(torch.randint(2, d - 2, (1,), generator=g)) for d in shape)
            instances[b, cz - 1 : cz + 2, cy - 1 : cy + 2, cx - 1 : cx + 2] = k
            if k % 2:  # odd: marked single center voxel; even: centroid fallback
                centers[b, cz, cy, cx] = True
    instances[0, :, :, :2] = -1  # ignore stripe
    labels = (instances > 0).long()
    C = 3 + 3 + 1 + extra_channels
    pred = 0.3 * torch.randn(batch, C, z, y, x, generator=g)
    return pred, instances, labels, centers


@pytest.mark.parametrize("extra_channels", [0, 2])
def test_matches_dense_value_and_gradient(extra_channels):
    shape = (12, 16, 16)
    pred, inst, lab, cen = _data(shape=shape, extra_channels=extra_channels)
    kw = dict(grid_z=shape[0], grid_y=shape[1], grid_x=shape[2])
    dense = SpatialEmbLoss_3d(memory_efficient=False, **kw)
    eff = SpatialEmbLoss_3d(memory_efficient=True, **kw)

    p1 = pred.clone().requires_grad_()
    l1 = dense(p1, inst, lab, cen)
    l1.backward()
    p2 = pred.clone().requires_grad_()
    l2 = eff(p2, inst, lab, cen)
    l2.backward()

    torch.testing.assert_close(l2.detach(), l1.detach(), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(p2.grad, p1.grad, rtol=1e-4, atol=1e-6)


def test_no_grad_uses_dense_path_and_same_value():
    shape = (12, 16, 16)
    pred, inst, lab, cen = _data(shape=shape)
    crit = SpatialEmbLoss_3d(grid_z=shape[0], grid_y=shape[1], grid_x=shape[2])
    with torch.no_grad():
        v = crit(pred, inst, lab, cen)
    p = pred.clone().requires_grad_()
    torch.testing.assert_close(v, crit(p, inst, lab, cen).detach(), rtol=1e-5, atol=1e-6)


def test_no_instances_still_differentiable():
    shape = (8, 8, 8)
    pred, inst, lab, cen = _data(shape=shape, n_obj=0)
    crit = SpatialEmbLoss_3d(grid_z=shape[0], grid_y=shape[1], grid_x=shape[2])
    p = pred.clone().requires_grad_()
    crit(p, inst, lab, cen).backward()
    assert p.grad is not None


def _cuda_usable():
    if not torch.cuda.is_available():
        return False
    try:
        torch.zeros(1, device="cuda") + 1
        return True
    except Exception:  # e.g. "no kernel image" on GPUs the build doesn't support
        return False


@pytest.mark.skipif(not _cuda_usable(), reason="needs a usable CUDA device")
def test_peak_memory_does_not_scale_with_instances():
    shape = (32, 64, 64)
    peaks = {}
    for mode in (False, True):
        crit = SpatialEmbLoss_3d(
            grid_z=shape[0], grid_y=shape[1], grid_x=shape[2], memory_efficient=mode
        ).cuda()
        pred, inst, lab, cen = (t.cuda() for t in _data(batch=1, shape=shape, n_obj=60))
        p = pred.requires_grad_()
        torch.cuda.reset_peak_memory_stats()
        crit(p, inst, lab, cen).backward()
        peaks[mode] = torch.cuda.max_memory_allocated()
    assert peaks[True] < 0.5 * peaks[False]
