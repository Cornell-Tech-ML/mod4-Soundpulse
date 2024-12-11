import pytest
from hypothesis import given

import minitorch
from minitorch import Tensor

from .strategies import assert_close
from .tensor_strategies import tensors


@pytest.mark.task4_3
@given(tensors(shape=(1, 1, 4, 4)))
def test_avg(t: Tensor) -> None:
    out = minitorch.avgpool2d(t, (2, 2))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(2) for j in range(2)]) / 4.0
    )

    out = minitorch.avgpool2d(t, (2, 1))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(2) for j in range(1)]) / 2.0
    )

    out = minitorch.avgpool2d(t, (1, 2))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(1) for j in range(2)]) / 2.0
    )
    minitorch.grad_check(lambda t: minitorch.avgpool2d(t, (2, 2)), t)


@pytest.mark.task4_4
@given(tensors(shape=(2, 3, 4)))
def test_max(t: Tensor) -> None:
    dim = 2
    q = minitorch.max(t, dim)
    assert q.shape == (2, 3, 1)
    for i in range(2):
        for j in range(3):
            expected = max([t[i, j, k] for k in range(4)])
            assert_close(q[i, j, 0], expected)

    dim = 1
    q = minitorch.max(t, dim)
    assert q.shape == (2, 1, 4)
    for i in range(2):
        for j in range(4):
            expected = max([t[i, k, j] for k in range(3)])
            assert_close(q[i, 0, j], expected)

    dim = 0
    q = minitorch.max(t, dim)
    assert q.shape == (1, 3, 4)
    for i in range(3):
        for j in range(4):
            expected = max([t[k, i, j] for k in range(2)])
            assert_close(q[0, i, j], expected)

    # Test max over all elements (dim=None)
    q = minitorch.max(t, dim=None)
    expected = max([t[i] for i in t._tensor.indices()])
    assert_close(q[0], expected)


@pytest.mark.task4_4
def test_max_backward() -> None:
    # Test dim=None backprop manually.
    # Define a 2D tensor
    backend = minitorch.TensorBackend(minitorch.FastOps)
    t = Tensor.make(
        [1.0, 9.0, 0.0, 5.0, 3.0, 8.0, 6.0, -1.0, 9.0], (3, 3), (3, 1), backend=backend
    )
    t.requires_grad_(True)

    # Run max and backward with grad_output
    out = minitorch.max(t, dim=1)
    grad_output = minitorch.tensor([1.0, 1.0, 1.0])
    out.backward(grad_output)

    # Check against expected gradients
    expected_grad = [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]

    assert t.grad is not None
    for i in range(3):
        for j in range(3):
            assert_close(t.grad[i, j], expected_grad[i][j])


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_max_pool(t: Tensor) -> None:
    out = minitorch.maxpool2d(t, (2, 2))
    print(out)
    print(t)
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(2) for j in range(2)])
    )

    out = minitorch.maxpool2d(t, (2, 1))
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(2) for j in range(1)])
    )

    out = minitorch.maxpool2d(t, (1, 2))
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(1) for j in range(2)])
    )


@pytest.mark.task4_4
@given(tensors())
def test_drop(t: Tensor) -> None:
    q = minitorch.dropout(t, 0.0)
    idx = q._tensor.sample()
    assert q[idx] == t[idx]
    q = minitorch.dropout(t, 1.0)
    assert q[q._tensor.sample()] == 0.0
    q = minitorch.dropout(t, 1.0, ignore=True)
    idx = q._tensor.sample()
    assert q[idx] == t[idx]


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_softmax(t: Tensor) -> None:
    q = minitorch.softmax(t, 3)
    x = q.sum(dim=3)
    assert_close(x[0, 0, 0, 0], 1.0)

    q = minitorch.softmax(t, 1)
    x = q.sum(dim=1)
    assert_close(x[0, 0, 0, 0], 1.0)

    minitorch.grad_check(lambda a: minitorch.softmax(a, dim=2), t)


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_log_softmax(t: Tensor) -> None:
    q = minitorch.softmax(t, 3)
    q2 = minitorch.logsoftmax(t, 3).exp()
    for i in q._tensor.indices():
        assert_close(q[i], q2[i])

    minitorch.grad_check(lambda a: minitorch.logsoftmax(a, dim=2), t)
