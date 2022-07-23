"""Unit tests for gpucompare"""
import pytest

from gpucompare.compare import compare_gpus


@pytest.mark.parametrize(
    ("gpu_spec_list", "expected"),
    [
        (
            [
                {
                    "gpu_name": "A2",
                    "architecture": "ampere",
                    "int8_perf": "36",
                    "mem_bandwidth": "200",
                },
                {
                    "gpu_name": "A10",
                    "architecture": "ampere",
                    "int8_perf": "250",
                    "mem_bandwidth": "600",
                },
                {
                    "gpu_name": "A30",
                    "architecture": "ampere",
                    "int8_perf": "330",
                    "mem_bandwidth": "933",
                },
            ],
            (
                [
                    {
                        "gpu_name": "A2",
                        "architecture": "ampere",
                        "int8_perf": "36",
                        "mem_bandwidth": "200",
                        "performance": "1x",
                    },
                    {
                        "gpu_name": "A10",
                        "architecture": "ampere",
                        "int8_perf": "250",
                        "mem_bandwidth": "600",
                        "performance": "3.0x",
                    },
                    {
                        "gpu_name": "A30",
                        "architecture": "ampere",
                        "int8_perf": "330",
                        "mem_bandwidth": "933",
                        "performance": "4.67x",
                    },
                ],
                {"A10/A2": "3.0x", "A30/A2": "4.67x"},
            ),
        ),
    ],
)
def test_compare_gpus(gpu_spec_list, expected):
    """test with parametrization."""
    assert compare_gpus(gpu_spec_list) == expected
