from dacapo.store.create_store import create_config_store
from dacapo.experiments import Run

import daisy
from funlib.persistence import open_ds, prepare_ds

import click
import numpy as np

import subprocess
import logging

import os

logger = logging.getLogger(__file__)
@click.group()
@click.option(
    "--log-level",
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
    ),
    default="INFO",
)
def cli(log_level):
    logging.basicConfig(level=getattr(logging, log_level.upper()))


def spawn_worker(
    name,
    criterion,
    channels,
    out_container,
    out_dataset,
    in_container,
    in_dataset,
    billing,
    local=True,
    min_raw=0,
    max_raw=255,
    mask_containers=list(),
    mask_datasets=list(),
    instance: bool = False,
):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    predict_script = os.path.join(current_dir, 'predict_worker.py')
    def run_worker():
        mask_args = []
        for mask_container, mask_dataset in zip(mask_containers, mask_datasets):
            mask_args.extend(["-mc", mask_container, "-md", mask_dataset])
        if local:
            subprocess.run(
                [
                    "python",
                    predict_script,
                    "start-worker",
                    "-n",
                    f"{name}",
                    "-c",
                    f"{criterion}",
                    "-cs",
                    f"{channels}",
                    "-oc",
                    f"{out_container}",
                    "-od",
                    f"{out_dataset}",
                    "-ic",
                    f"{in_container}",
                    "-id",
                    f"{in_dataset}",
                    "--min-raw",
                    f"{min_raw}",
                    "--max-raw",
                    f"{max_raw}",
                    "--instance",
                    f"{instance}",
                ]
                + mask_args
            )
        else:
            subprocess.run(
                [
                    "bsub",
                    "-P",
                    billing,
                    "-J",
                    "pred",
                    "-q",
                    "gpu_tesla",
                    "-n",
                    "10",
                    "-gpu",
                    "num=1",
                    "-o",
                    f"prediction_logs/{name}.out",
                    "-e",
                    f"prediction_logs/{name}.err",
                    "python",
                    predict_script,
                    "start-worker",
                    "-n",
                    f"{name}",
                    "-c",
                    f"{criterion}",
                    "-cs",
                    f"{channels}",
                    "-oc",
                    f"{out_container}",
                    "-od",
                    f"{out_dataset}",
                    "-ic",
                    f"{in_container}",
                    "-id",
                    f"{in_dataset}",
                    "--min-raw",
                    f"{min_raw}",
                    "--max-raw",
                    f"{max_raw}",
                    "--instance",
                    f"{instance}",
                ]
                + mask_args
            )

    return run_worker


@cli.command()
@click.option("-n", "--name", type=str)
@click.option("-c", "--criterion", type=str)
@click.option("-cs", "--channels", type=str)
@click.option("-oc", "--out_container", type=click.Path(file_okay=False))
@click.option("-od", "--out_dataset", type=str)
@click.option("-ic", "--in_container", type=click.Path(exists=True, file_okay=False))
@click.option("-id", "--in_dataset", type=str)
@click.option("-w", "--workers", type=int, default=1)
@click.option(
    "-roi",
    "--roi",
    type=str,
    required=False,
    help="The roi to predict on. Passed in as [lower:upper, lower:upper, ... ]",
)
@click.option("--local/--bsub", default=True)
@click.option("--billing", default=None)
@click.option("--min-raw", type=float, default=0)
@click.option("--max-raw", type=float, default=255)
@click.option(
    "-mc",
    "--mask-container",
    type=click.Path(file_okay=False),
    multiple=True,
    default=list(),
)
@click.option(
    "-md",
    "--mask-dataset",
    type=click.Path(file_okay=False),
    multiple=True,
    default=list(),
)
@click.option("--instance", type=bool, default=False)
def predict(
    name,
    criterion,
    channels,
    out_container,
    out_dataset,
    in_container,
    in_dataset,
    workers,
    roi,
    local,
    billing,
    min_raw,
    max_raw,
    mask_container,
    mask_dataset,
    instance,
):
    if not local:
        assert billing is not None
    parsed_channels = [channel.split(":") for channel in channels.split(",")]

    raw = open_ds(in_container, in_dataset)

    config_store = create_config_store()
    run_config = config_store.retrieve_run_config(name)
    run = Run(run_config,load_starter_model = False)

    model = run.model

    # import torch
    # device = torch.device("cuda")
    # model = model.to(device)
    if roi is not None:
        parsed_start, parsed_end = zip(
            *[
                tuple(int(coord) for coord in axis.split(":"))
                for axis in roi.strip("[]").split(",")
            ]
        )
        parsed_roi = daisy.Roi(
            daisy.Coordinate(parsed_start),
            daisy.Coordinate(parsed_end) - daisy.Coordinate(parsed_start),
        )
    else:
        parsed_roi = raw.roi

    total_write_roi = raw.roi
    output_voxel_size = model.scale(raw.voxel_size)
    print("input_voxel_size", raw.voxel_size)
    print("output_voxel_size", output_voxel_size)

    eval_input_shape = model.eval_input_shape

    # from funlib.geometry import Coordinate
    # eval_input_shape = Coordinate((288*2, 288, 288))

    read_shape = eval_input_shape * raw.voxel_size
    print(f"read_shape: {eval_input_shape} * {raw.voxel_size} -> {read_shape}")
    write_shape = (
        model.compute_output_shape(eval_input_shape)[1] * output_voxel_size
    )
    print(f"write_shape: {write_shape}")
    # return
    context = (read_shape - write_shape) / 2
    read_roi = daisy.Roi((0,) * read_shape.dims, read_shape)
    write_roi = read_roi.grow(-context, -context)

    total_write_roi = parsed_roi.snap_to_grid(raw.voxel_size)
    total_read_roi = total_write_roi.grow(context, context)

    if not instance:
        for indexes, channel in parsed_channels:
            num_channels = None if "-" not in indexes else len(indexes.split("-"))
            prepare_ds(
                out_container,
                f"{out_dataset}/{channel}",
                total_roi=total_write_roi,
                voxel_size=output_voxel_size,
                write_size=write_roi.shape,
                dtype=np.uint8,
                num_channels=num_channels,
            )
    else:
        num_channels = model.num_out_channels
        assert len(parsed_channels) == 1
        indexes, channel = parsed_channels[0]
        for i in range(0, num_channels, 3):
            prepare_ds(
                out_container,
                f"{out_dataset}/{channel}__{i}",
                total_roi=total_write_roi,
                voxel_size=output_voxel_size,
                write_size=write_roi.shape,
                dtype=np.float32,
                num_channels=min(3, num_channels - i),
            )


    task = daisy.Task(
        f"predict_{name}",
        total_roi=total_read_roi,
        read_roi=read_roi,
        write_roi=write_roi,
        process_function=spawn_worker(
            name,
            criterion,
            channels,
            out_container,
            out_dataset,
            in_container,
            in_dataset,
            billing,
            local,
            min_raw,
            max_raw,
            mask_container,
            mask_dataset,
            instance,
        ),
        check_function=None,
        read_write_conflict=False,
        fit="overhang",
        num_workers=workers,
        max_retries=0,
        timeout=None,
    )

    daisy.run_blockwise([task])


if __name__ == "__main__":
    cli()