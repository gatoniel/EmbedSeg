import pytest

torch = pytest.importorskip("torch")

from EmbedSeg.models import get_model


def _model(name, num_classes, input_channels):
    return get_model(name, dict(num_classes=num_classes, input_channels=input_channels))


def test_tracking_model_2d_channels_and_seg_heads():
    model = _model("branched_erfnet", [4, 1, 5], input_channels=3).eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        full = model(x)
        seg = model(x, heads="seg")
    assert full.shape == (1, 10, 32, 32)
    assert seg.shape == (1, 5, 32, 32)
    # seg-only pass equals the first block of the full pass
    assert torch.allclose(full[:, :5], seg, atol=1e-5)


def test_tracking_model_3d_channels_and_seg_heads():
    model = _model("branched_erfnet_3d", [6, 1, 7], input_channels=3).eval()
    x = torch.randn(1, 3, 16, 16, 16)
    with torch.no_grad():
        full = model(x)
        seg = model(x, heads="seg")
    assert full.shape == (1, 14, 16, 16, 16)
    assert seg.shape == (1, 7, 16, 16, 16)
    assert torch.allclose(full[:, :7], seg, atol=1e-5)


def test_init_output_covers_tracking_head():
    model = _model("branched_erfnet_3d", [6, 1, 7], input_channels=2)
    model.init_output(3)
    for idx in (0, 2):
        bias = model.decoders[idx].output_conv.bias
        assert torch.all(bias[0:3] == 0)
        assert torch.all(bias[3:6] == 1)


def test_old_config_unchanged():
    model = _model("branched_erfnet", [4, 1], input_channels=1).eval()
    x = torch.randn(1, 1, 32, 32)
    with torch.no_grad():
        assert model(x).shape == (1, 5, 32, 32)
        assert model(x, heads="seg").shape == (1, 5, 32, 32)
    with pytest.raises(ValueError):
        model(x, heads="bogus")
