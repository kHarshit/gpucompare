"""Parse input data"""

import csv

from .compare import compare_gpus


def parse_csv(csv_path: str) -> tuple[list[dict[str, str]], dict[str, str]]:
    """Print GPU name.

    Args:
        csv_path (str): path to CSV file containing row-wise GPU data

    Returns:
        out_detailed_dict: detailed dictionary as string containing performance comparison of GPUs
        out_concise_dict: concise dictionary as string containing performance comparison of GPUs

    Examples:
        .. code:: python

            >>> parse_csv("/path/to/gpu_data.csv")
            ([{'gpu_name': 'A2', 'architecture': 'ampere', 'int8_perf': '36', 'mem_bandwidth': '200', 'performance': '1x'}, {'gpu_name': 'A10', 'architecture': 'ampere', 'int8_perf': '250', 'mem_bandwidth': '600', 'performance': '3.0x'}, {'gpu_name': 'A30', 'architecture': 'ampere', 'int8_perf': '330', 'mem_bandwidth': '933', 'performance': '4.67x'}], {'A10/A2': '3.0x', 'A30/A2': '4.67x'})
    """
    # list containing dicts having gpus specs
    gpu_spec_list = []
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        csv_header = []
        for i, row in enumerate(csv_reader):
            # set first row as header
            if i == 0:
                csv_header = row
            # append rest rows' data as dictionary in a list
            else:
                gpu_spec = {}
                for spec, col_val in zip(csv_header, row):
                    gpu_spec[spec] = col_val
                gpu_spec_list.append(gpu_spec)
    # compare gpus
    return compare_gpus(gpu_spec_list)
