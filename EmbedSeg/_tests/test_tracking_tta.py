import numpy as np
import pytest

torch = pytest.importorskip("torch")

from EmbedSeg.utils import test_time_augmentation as tta


class _NoCuda:
    """Run the TTA code on CPU (it calls `.cuda()` unconditionally)."""

    def __enter__(self):
        self.orig = torch.Tensor.cuda
        torch.Tensor.cuda = lambda self, *a, **k: self
        return self

    def __exit__(self, *exc):
        torch.Tensor.cuda = self.orig


class _Concat(torch.nn.Module):
    def __init__(self, *models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)

    def forward(self, x):
        return torch.cat([m(x) for m in self.models], 1)


def _conv_models(ndim, c_out, n):
    conv = torch.nn.Conv2d if ndim == 2 else torch.nn.Conv3d
    models = []
    for k in range(n):
        torch.manual_seed(k)
        models.append(conv(2, c_out, 3, padding=1).eval())
    return models


def test_blockwise_tta_2d_matches_single_block_tta():
    a, b = _conv_models(2, 5, 2)
    x = torch.randn(1, 2, 12, 12)
    with _NoCuda(), torch.no_grad():
        ref_a = tta.apply_tta_2d(x, a)
        ref_b = tta.apply_tta_2d(x, b)
        both = tta.apply_tta_2d(x, _Concat(a, b))
    assert both.shape == (1, 10, 12, 12)
    assert torch.allclose(both[:, :5], ref_a, atol=1e-5)
    assert torch.allclose(both[:, 5:], ref_b, atol=1e-5)


@pytest.mark.parametrize("index", list(range(16)))
def test_blockwise_tta_3d_matches_single_block_tta(index):
    a, b = _conv_models(3, 7, 2)
    x = torch.randn(1, 2, 8, 8, 8)
    with _NoCuda(), torch.no_grad():
        ref_a = tta.apply_tta_3d(x, a, index)
        ref_b = tta.apply_tta_3d(x, b, index)
        both = tta.apply_tta_3d(x, _Concat(a, b), index)
    assert both.shape == (1, 14, 8, 8, 8)
    assert np.allclose(both[:, :7], ref_a, atol=1e-5)
    assert np.allclose(both[:, 7:], ref_b, atol=1e-5)


def test_plain_models_behave_as_before():
    # a model whose channel count is not a multiple of the block size is passed through
    (m,) = _conv_models(2, 5, 1)
    (m3,) = _conv_models(3, 7, 1)
    x2, x3 = torch.randn(1, 2, 12, 12), torch.randn(1, 2, 8, 8, 8)
    with _NoCuda(), torch.no_grad():
        assert tta.apply_tta_2d(x2, m).shape == (1, 5, 12, 12)
        assert tta.apply_tta_3d(x3, m3, 5).shape == (1, 7, 8, 8, 8)
        (odd,) = _conv_models(2, 6, 1)  # 6 channels: not a multiple of 5
        assert tta.apply_tta_2d(x2, odd).shape == (1, 6, 12, 12)
