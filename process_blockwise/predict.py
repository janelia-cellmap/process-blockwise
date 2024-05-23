import click
import logging
from pathlib import Path
import yaml
import subprocess
import os


def is_string_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False
    

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


@cli.command()
@click.option("-p", "--prediction", type=click.Path(exists=True, dir_okay=False))
def predict(prediction,):
    predictions = yaml.safe_load(Path(prediction).open("r").read())
    runs_path = predictions["runs_path"]
    output_folder = predictions["output_folder"]
    input_pred = predictions["input"]
    workers = predictions["workers"]
    in_dataset = predictions["in_dataset"]
    if "roi" in predictions:
        roi = predictions["roi"]
    else:
        roi = None


    for run, params in predictions["runs"].items():
        print(run)
        print(params)

        iteration = params["checkpoint"]
        output = params["output"]
        dataset = params["dataset"]
        print("dataset:",dataset)
        if "output_file" in params:
            output_file = params["output_file"]
        else:
            output_file = output_folder

        if iteration == "latest":
            iteration_path = os.path.join(runs_path,run,"checkpoints/iterations/")
            iterations = os.listdir(iteration_path)
            iteration = str(max([int(i) for i in iterations if is_string_int(i)]))
        os.makedirs(f"prediction_logs/{run}",exist_ok=True)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        predict_script = os.path.join(current_dir, 'predict_daisy.py')

        command = ["bsub",
        "-P",
        "cellmap",
        "-J",
        "pred_maaaaster",
        "-o",
        f"prediction_logs/{run}/prediction_master.out",
        "-e",
        f"prediction_logs/{run}/prediction_master.err",
        "-n"
        "12",
        "python",
        predict_script,
        "predict",
        "-n",
        run,
        "-c",
        iteration,
        "-cs",
        output,
        "-oc",
        output_file,
        "-od",
        dataset,
        "-ic",
        input_pred,
        "-id",
        in_dataset,
        "-w",
        str(workers),
        "--bsub",
        "--billing",
        "cellmap",
        ] + (["--roi", roi] if roi is not None else [])
        print("command: ",command)
        subprocess.run(command)
        print("finished!")

if __name__ == "__main__":
    cli()
