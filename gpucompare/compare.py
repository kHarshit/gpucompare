"""Contains logic to compare gpus"""


def compare_gpus(
    gpu_spec_list: list[dict[str, str]]
) -> tuple[list[dict[str, str]], dict[str, str]]:
    """Compare GPU specs

    Args:
        gpu_spec_list (list[dict[str, str]]): a list of dictionaries having gpu specs

    Returns:
        out_detailed_dict: detailed dictionary as string containing performance comparison of GPUs
        out_concise_dict: concise dictionary as string containing performance comparison of GPUs

    Examples:
        .. code:: python

            >>> compare_gpus([{'gpu_name': 'A2', 'architecture': 'ampere', 'int8_perf': '36', 'mem_bandwidth': '200'}, {'gpu_name': 'A10', 'architecture': 'ampere', 'int8_perf': '250', 'mem_bandwidth': '600'}, {'gpu_name': 'A30', 'architecture': 'ampere', 'int8_perf': '330', 'mem_bandwidth': '933'}])
            ([{'gpu_name': 'A2', 'architecture': 'ampere', 'int8_perf': '36', 'mem_bandwidth': '200', 'performance': '1x'}, {'gpu_name': 'A10', 'architecture': 'ampere', 'int8_perf': '250', 'mem_bandwidth': '600', 'performance': '3.0x'}, {'gpu_name': 'A30', 'architecture': 'ampere', 'int8_perf': '330', 'mem_bandwidth': '933', 'performance': '4.67x'}], {'A10/A2': '3.0x', 'A30/A2': '4.67x'})
    """
    # params to compare
    spec_params = [
        "architecture",
        "mem_bandwidth",
        "cuda_cores",
        "fp32_perf",
        "int8_perf",
        "fp16_perf",
    ]
    # architecture speedup in increasing order
    arch_list = ["volta", "turing", "ampere", "hopper"]
    # ampere_w.r.t._volta will have speedup of 0.05+0.05
    # volta_w.r.t._ampere will have speedup of -0.05-0.05
    arch_speedup = [0.05, 0.05, 0.05]

    # create copy of gpu spec list of dicts to output
    # add one more key "performance" in output
    out_detailed_dict = gpu_spec_list
    # store spec comparison of gpus in concise format
    out_concise_dict = {}
    # take first gpu as base gpu to compare with
    base_gpu_spec = gpu_spec_list[0]
    # add perf as 1x to first gpus spec
    out_detailed_dict[0]["performance"] = "1x"
    # compare rest gpus with base gpu
    for i, gpu_spec in enumerate(gpu_spec_list[1:]):
        # difference in gpu performance
        gpu_perf_diff = float("inf")
        arch_diff = 0.0
        # iterate the specs which are to be compared
        for spec in spec_params:
            # check if spec data exists
            if spec in gpu_spec:
                if spec == "architecture":
                    if gpu_spec[spec] in arch_list and base_gpu_spec[spec] in arch_list:
                        current_gpu_arch_index = arch_list.index(gpu_spec[spec])
                        base_gpu_arch_index = arch_list.index(base_gpu_spec[spec])
                        if current_gpu_arch_index >= base_gpu_arch_index:
                            for j in range(base_gpu_arch_index, current_gpu_arch_index):
                                arch_diff += arch_speedup[j]
                        else:
                            for j in range(current_gpu_arch_index, base_gpu_arch_index):
                                arch_diff -= arch_speedup[j]
                else:
                    perf_ratio = float(gpu_spec[spec]) / float(base_gpu_spec[spec])
                    # take min criterion
                    if perf_ratio < gpu_perf_diff:
                        gpu_perf_diff = perf_ratio
        gpu_perf_diff += arch_diff
        gpu_perf_diff = round(gpu_perf_diff, 2)
        out_detailed_dict[i + 1]["performance"] = str(gpu_perf_diff) + "x"
        out_concise_dict[gpu_spec["gpu_name"] + "/" + base_gpu_spec["gpu_name"]] = (
            str(gpu_perf_diff) + "x"
        )

    return out_detailed_dict, out_concise_dict
