import os
import subprocess
import time
import sys
import daisy
import numpy as np
import tempfile
from funlib.persistence import prepare_ds
from funlib.persistence import Array, open_ds
from config import *
from scipy.cluster.hierarchy import DisjointSet
from tqdm import tqdm
import pickle
from glob import glob

from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

def load_and_deserialize(npy_file):
    loaded_serialized_dict = np.load(npy_file, allow_pickle=True)
    return pickle.loads(loaded_serialized_dict)

def merge_sizes(folder_path):
    npy_files = glob(os.path.join(folder_path, "sizes_*.npy"))
    sizes = Counter()
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(load_and_deserialize, npy_file) for npy_file in npy_files]
        
        for future in tqdm(as_completed(futures), total=len(npy_files)):
            current = future.result()
            sizes.update(current)
    
    return dict(sizes)


def load_npz(file):
    data = np.load(file)
    return data["nodes"], data["edges"]

def read_cross_block_merges(tmpdir):
    print("Reading cross block merges...")
    block_files = glob(os.path.join(tmpdir, "block_*.npz"))

    nodes_list = []
    edges_list = []

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(load_npz, block_file) for block_file in block_files]

        for future in tqdm(futures, total=len(futures)):
            nodes, edges = future.result()
            nodes_list.append(nodes)
            edges_list.append(edges)

    return np.concatenate(nodes_list), np.concatenate(edges_list)


import networkx as nx
def find_components(nodes, edges):
    print("Finding components...")
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    
    components = list(nx.connected_components(G))
    component_map = {node: idx for idx, comp in enumerate(components) for node in comp}
    
    return [component_map[node] for node in nodes]


def save_elements(tmpdir, nodes, edges, components):
    print("Saving elements...")
    np.savez(os.path.join(tmpdir, "elements.npz"), nodes=nodes, edges=edges, components=components)


def format_roi(string_roi):
    parsed_start, parsed_end = zip(
                *[
                    tuple(int(coord) for coord in axis.split(":"))
                    for axis in string_roi.strip("[]").split(",")
                ]
            )
    total_roi = daisy.Roi(
                daisy.Coordinate(parsed_start),
                daisy.Coordinate(parsed_end) - daisy.Coordinate(parsed_start),
            )
    return total_roi

def segment_blockwise(config_yaml):
    config = Config(config_yaml)
    process_config = config.get_process_config()
    
    input_file = config.data.input_container
    dataset = config.data.in_dataset
    output_file = config.data.output_container
    context = config.data.context

    array_in = open_ds(input_file, dataset)

    roi = config.data.roi
    if roi:
        total_roi = format_roi(roi)
    else:
        total_roi = array_in.roi

    voxel_size = array_in.voxel_size

    block_size = np.array(array_in.data.chunks) * np.array(voxel_size)
    write_size = daisy.Coordinate(block_size)

    write_roi = daisy.Roi((0,) * len(write_size), write_size)

    read_roi = write_roi.grow(context, context)
    
    num_voxels_in_block = (read_roi / array_in.voxel_size).size

    task = config.task

    task_name = task.task_name
    tmpdir = task.tmpdir

    num_cpus = task.num_cpus
    num_workers = task.num_workers


    for current_step, step_dict in process_config.items():
        print(f"Running step: {current_step}")

        out_dataset = os.path.join(config.data.output_group, current_step)


        array_out = prepare_ds(
            output_file,
            out_dataset,
            total_roi,
            voxel_size=voxel_size,
            write_size=write_size,
            dtype=np.uint64,
        )

        print("Starting segmentation...")
        print(f"total_roi: {total_roi.grow(context, context)}:")
        print(f"read_roi: {read_roi}:")
        print(f"write_roi: {write_roi}:")

        def start_worker():
            worker_id = daisy.Context.from_env()["worker_id"]
            task_id = daisy.Context.from_env()["task_id"]

            print(f"worker {worker_id} started for task {task_id}...")
            log_file_path = f"./daisy_logs/{task_id}/worker_{worker_id}"
            current_dir = os.path.dirname(os.path.abspath(__file__))
            segment_worker_script = os.path.join(current_dir, 'segment_worker.py')


            subprocess.run(
                [
                    "bsub",
                    "-K",
                    "-P",
                    "cellmap",
                    "-J",
                    f"segment_worker_{task_id}_{worker_id}",
                    "-n",
                    str(num_cpus),
                    "-e",
                    f"{log_file_path}.err",
                    "-o",
                    f"{log_file_path}.out",
                    "python",
                    segment_worker_script,
                    tmpdir,
                    config_yaml,
                    current_step,
                ]
            )

        task = daisy.Task(
            f"segment_{task_name}",
            total_roi.grow(context, context),
            read_roi,
            write_roi,
            process_function=start_worker,
            num_workers=num_workers,
            fit="shrink",
            read_write_conflict=False,
            timeout=10,
        )
        daisy.run_blockwise([task])


        print("Finished segmentation. Relabeling...")

        save_edges =  step_dict.get('params', {}).get('save_edges', False)
        if save_edges:

            nodes, edges = read_cross_block_merges(os.path.join(tmpdir, "blocks"))

            components = find_components(nodes, edges)

            save_elements(tmpdir, nodes, edges, components)
    
    # wandb.finish()


if __name__ == "__main__":
    args = sys.argv[1:]
    config_yaml = args[1]
    segment_blockwise(config_yaml)
