import argparse
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
    

def predict(prediction,):
    predictions = yaml.safe_load(Path(prediction).open("r").read())
    
    output_folder = predictions["output_folder"]
    input_pred = predictions["input"]
    workers = predictions["workers"]
    billing = predictions.get("billing", None)
    instances = predictions.get("instances", None)
    script_path  = predictions.get("script", None)
    
    if billing is None:
        raise ValueError("Billing must be set in the prediction yaml file")
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
        dataset = run
        print("dataset:",dataset)
        if "output_file" in params:
            output_file = params["output_file"]
        else:
            output_file = output_folder

        if iteration == "latest":
            runs_path = predictions["runs_path"]
            iteration_path = os.path.join(runs_path,run,"checkpoints/iterations/")
            iterations = os.listdir(iteration_path)
            iteration = str(max([int(i) for i in iterations if is_string_int(i)]))
        os.makedirs(f"prediction_logs/{run}",exist_ok=True)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        predict_script = os.path.join(current_dir, 'predict_daisy.py')

        command = ["bsub",
        "-P",
        billing,
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
        billing,
        ] + (["--roi", roi] if roi is not None else []) + (["--instance",instances] if instances is not None else []) + (["--script", script_path] if script_path is not None else [])
        print("command: ",command)
        subprocess.run(command)
        # sleep 1 min
        # import time
        # time.sleep(60)
        print("finished!")


def main():
    parser = argparse.ArgumentParser(description="Process blockwise with a given path.")
    parser.add_argument('path', type=str, help='The path to process blockwise')
    args = parser.parse_args()
    
    config_yaml = args.path
    predict(config_yaml)



if __name__ == "__main__":
    main()
