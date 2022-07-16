"""Tests for gpu_name function."""
import pytest

from gpucompare.example import gpu_name


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Tesla T4", "GPU: Tesla T4!"),
        ("RTX 2060 Super", "GPU: RTX 2060 Super!"),
    ],
)
def test_hello(name, expected):
    """Example test with parametrization."""
    assert gpu_name(name) == expected
