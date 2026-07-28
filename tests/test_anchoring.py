import torch

from aap.anchoring import asymmetric_anchor_objective, cosine_error


def test_cosine_error_is_zero_for_identical_features():
    features = torch.randn(2, 4, 8)
    errors = cosine_error(features, features)
    torch.testing.assert_close(errors, torch.zeros_like(errors), atol=1e-5, rtol=0)


def test_loss_uses_only_real_samples_and_maps_only_tampered_samples():
    anchor = torch.randn(3, 4, 8)
    projected = anchor[:, None].repeat(1, 2, 1, 1)
    projected[1] = -projected[1]
    labels = torch.tensor([0, 1, 2])

    loss, error_maps = asymmetric_anchor_objective(projected, anchor, labels)

    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-5, rtol=0)
    assert error_maps[0] is None
    assert error_maps[1] is None
    torch.testing.assert_close(
        error_maps[2],
        torch.zeros_like(error_maps[2]),
        atol=1e-5,
        rtol=0,
    )


def test_non_real_batch_keeps_zero_loss_in_graph():
    projected = torch.randn(2, 1, 4, 8, requires_grad=True)
    anchor = torch.randn(2, 4, 8)
    labels = torch.tensor([1, 2])

    loss, _ = asymmetric_anchor_objective(projected, anchor, labels)
    loss.backward()

    assert projected.grad is not None
    torch.testing.assert_close(projected.grad, torch.zeros_like(projected.grad))
