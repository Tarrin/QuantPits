"""Focused tests for the Qlib-compatible TSMixer backbone."""

import pytest


torch = pytest.importorskip("torch", reason="torch not installed")

from quantpits.utils.model_wrappers.custom.pytorch_tsmixer import TSMixer


def test_forward_shape_preserves_batch_dimension():
    model = TSMixer(
        d_feat=5,
        seq_len=7,
        hidden_time=4,
        hidden_feature=8,
        n_blocks=2,
        dropout=0.0,
    )

    output = model(torch.randn(4, 7, 5))

    assert output.shape == (4, 1)


def test_forward_shape_for_single_sample_batch():
    model = TSMixer(
        d_feat=5,
        seq_len=7,
        hidden_time=4,
        hidden_feature=8,
        n_blocks=1,
        dropout=0.0,
    )

    output = model(torch.randn(1, 7, 5))

    assert output.shape == (1, 1)


def test_forward_and_backward_gradients_are_finite():
    model = TSMixer(
        d_feat=5,
        seq_len=7,
        hidden_time=4,
        hidden_feature=8,
        n_blocks=2,
        dropout=0.1,
    )
    features = torch.randn(3, 7, 5, requires_grad=True)

    loss = model(features).square().mean()
    loss.backward()

    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    parameter_grads = [parameter.grad for parameter in model.parameters()]
    assert parameter_grads
    assert all(gradient is not None for gradient in parameter_grads)
    assert all(torch.isfinite(gradient).all() for gradient in parameter_grads)


def test_eval_mode_is_deterministic_and_keeps_output_shape():
    model = TSMixer(
        d_feat=5,
        seq_len=7,
        hidden_time=4,
        hidden_feature=8,
        n_blocks=2,
        dropout=0.5,
    )
    model.eval()
    features = torch.randn(1, 7, 5)

    with torch.no_grad():
        first = model(features)
        second = model(features)

    assert first.shape == (1, 1)
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("d_feat", 0),
        ("seq_len", 0),
        ("hidden_time", 0),
        ("hidden_feature", 0),
        ("n_blocks", 0),
        ("dropout", -0.1),
        ("dropout", 1.0),
    ],
)
def test_invalid_constructor_values_raise_value_error(argument, value):
    kwargs = {argument: value}

    with pytest.raises(ValueError, match=argument):
        TSMixer(**kwargs)


@pytest.mark.parametrize(
    "features",
    [
        torch.randn(7, 5),
        torch.randn(2, 6, 5),
        torch.randn(2, 7, 4),
        torch.randn(0, 7, 5),
    ],
)
def test_invalid_input_dimensions_raise_value_error(features):
    model = TSMixer(
        d_feat=5,
        seq_len=7,
        hidden_time=4,
        hidden_feature=8,
        n_blocks=1,
        dropout=0.0,
    )

    with pytest.raises(ValueError, match="shape|seq_len|d_feat|batch"):
        model(features)

