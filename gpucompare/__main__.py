from typing import Optional

from enum import Enum
from random import choice

import typer
from rich.console import Console

from gpucompare import version
from gpucompare.parse import parse_csv


class Color(str, Enum):
    white = "white"
    red = "red"
    cyan = "cyan"
    magenta = "magenta"
    yellow = "yellow"
    green = "green"


app = typer.Typer(
    name="gpucompare",
    help="Compare GPUs",
    add_completion=False,
)
console = Console()


def version_callback(print_version: bool) -> None:
    """Print the version of the package."""
    if print_version:
        console.print(f"[yellow]gpucompare[/] version: [bold blue]{version}[/]")
        raise typer.Exit()


def print_table(my_dict, col_list=None):
    """Pretty print a list of dictionaries (my_dict) as a dynamically sized table.
    If column names (col_list) aren't specified, they will show in random order.
    """
    if not col_list:
        col_list = list(my_dict[0].keys() if my_dict else [])
    my_list = [col_list]  # 1st row = header
    for item in my_dict:
        my_list.append(
            [str(item[col] if item[col] is not None else "") for col in col_list]
        )
    col_size = [max(map(len, col)) for col in zip(*my_list)]
    format_str = " | ".join([f"{{:<{i}}}" for i in col_size])
    my_list.insert(1, ["-" * i for i in col_size])  # Seperating line
    return my_list, format_str


@app.command(name="")
def main(
    csv_data: str = typer.Option(
        ...,
        help="""CSV file containing row-wise GPU data

            \b 
            Possible columns:
            gpu_name (str): name of gpu  [required]
            architecture (str): GPU architecture
            cuda_cores (int): number of cuda cores
            fp32_perf (float): fp32 performance in TFLOPS
            fp16_perf (float): fp16 performance in TFLOPS
            int8_perf (float): int8 performance in TOPS
            mem (float): gpu memory in GiB
            mem_bandwidth (float): memory bandwidth in GB/s
            """,
    ),
    output: str = typer.Option(
        "concise",
        "--output",
        case_sensitive=False,
        help="Output in 'concise' or 'detailed' format.",
    ),
    # color: Optional[Color] = typer.Option(
    #     None,
    #     "-c",
    #     "--color",
    #     "--colour",
    #     case_sensitive=False,
    #     help="Color for print. If not specified then choice will be random.",
    # ),
    print_version: bool = typer.Option(
        None,
        "-v",
        "--version",
        callback=version_callback,
        is_eager=True,
        help="Prints the version of the gpucompare package.",
    ),
) -> None:
    """Compare GPUs"""
    # if color is None:
    color = choice(list(Color))

    detailed_dict, concise_dict = parse_csv(csv_data)
    if output == "concise":
        console.print(f"[bold {color}]{concise_dict}[/]")
    elif output == "detailed":
        out_list, format_str = print_table(detailed_dict)
        for item in out_list:
            console.print(f"[bold {color}]{format_str.format(*item)}[/]")


if __name__ == "__main__":
    app()
