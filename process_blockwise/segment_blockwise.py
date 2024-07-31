import os
import subprocess
import time
import sys
import daisy
import numpy as np
import tempfile
from funlib.persistence import prepare_ds
from funlib.persistence import Array, open_ds
from .config import *
from scipy.cluster.hierarchy import DisjointSet
from tqdm import tqdm
import pickle
from glob import glob
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

import logging

logger = logging.getLogger(__file__)

def get_minimal_dtype(value):
    dtypes = [
        np.uint8, np.uint16, np.uint32, np.uint64,
    ]
    for dtype in dtypes:
        if np.can_cast(value, dtype):
            return dtype
    
    raise ValueError("Value is too large to fit in available dtypes.")

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


def load_ids(tmpdir):
        a = np.load(os.path.join(tmpdir, "ids.npz"))
        return a["old_ids"], a["new_ids"]

def load_edges_npz(file):
    data = np.load(file)
    return data["nodes"], data["edges"]

def load_ids_npz(file):
    data = np.load(file)
    return data["ids"]

def read_merge_ids(tmpdir):
    print("Reading cross block merges...")
    block_files = glob(os.path.join(tmpdir, "ids_*.npz"))

    result = set()

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(load_ids_npz, block_file) for block_file in block_files]

        for future in tqdm(futures, total=len(futures)):
            ids = future.result()
            if len(ids) > 0:
                result.update(ids)

    return np.array(list(result))

def read_cross_block_merges(tmpdir):
    print("Reading cross block merges...")
    block_files = glob(os.path.join(tmpdir, "block_*.npz"))

    nodes_list = []
    edges_list = []

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(load_edges_npz, block_file) for block_file in block_files]

        for future in tqdm(futures, total=len(futures)):
            nodes, edges = future.result()
            if len(nodes) > 0:
                nodes_list.append(nodes)
            if len(edges) > 0:
                edges_list.append(edges)

    return np.concatenate(nodes_list), np.concatenate(edges_list)
    # return nodes_list, edges_list


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

    write_roi = daisy.Roi((0,) * len(write_size), write_size*2)

    read_roi = write_roi.grow(context, context)
    
    num_voxels_in_block = (read_roi / array_in.voxel_size).size

    task = config.task

    task_name = task.task_name
    tmpdir = os.path.join(task.tmpdir, task_name)

    if task.empty_tmpdir:
        if os.path.exists(tmpdir):
            logger.error(f"Removing tmpdir: {tmpdir}")
            import shutil
            shutil.rmtree(tmpdir)
    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)

    num_cpus = task.num_cpus
    num_workers = task.num_workers


    for current_step, step_dict in process_config.items():
        print(f"Running step: {current_step}")

        out_dataset = os.path.join(config.data.output_group, current_step)

        overwrite = step_dict.get('override', False)
        save_result = step_dict.get('save_result', True)

        if overwrite:
            logger.error(f"Overwriting dataset: {out_dataset}")
            import shutil
            shutil.rmtree(os.path.join(output_file, out_dataset))

        use_ids = step_dict.get('params', {}).get('use_ids', False)
        minimal_dtype = step_dict.get('params', {}).get('minimal_dtype', False)

        if minimal_dtype:
            if not use_ids:
                raise ValueError("Cannot use minimal dtype without using ids.")
            _, new_ids = load_ids(tmpdir)
            max_value = len(new_ids) +1
            print(f"Max value: {max_value}")
            current_dtype = get_minimal_dtype(max_value)
            print(f"Using dtype: {current_dtype}")
        else:
            current_dtype = np.uint64

        if save_result:
            array_out = prepare_ds(
                output_file,
                out_dataset,
                total_roi,
                voxel_size=voxel_size,
                write_size=write_size,
                dtype=current_dtype,
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
            f"segment_{current_step}_{task_name}",
            total_roi.grow(context, context),
            read_roi,
            write_roi,
            process_function=start_worker,
            num_workers=num_workers,
            fit="shrink",
            read_write_conflict=False,
            timeout=10,
        )
        
        skip = step_dict.get('skip', False)
        print(f"Skip: {skip}")
        print(f"step_dict: {step_dict}")
        
        if not skip:
            daisy.run_blockwise([task])


        print("Finished segmentation. Relabeling...")

        save_edges =  step_dict.get('params', {}).get('save_edges', False)
        save_ids =  step_dict.get('params', {}).get('save_ids', False)
        if save_edges:

            nodes, edges = read_cross_block_merges(os.path.join(tmpdir, "blocks"))

            components = find_components(nodes, edges)

            save_elements(tmpdir, nodes, edges, components)
        if save_ids:
            old_ids = read_merge_ids(os.path.join(tmpdir, "ids"))
            new_ids = np.arange(1, len(old_ids)+1)
            np.savez_compressed(os.path.join(tmpdir, "ids.npz"), old_ids=old_ids, new_ids=new_ids)


    
    # wandb.finish()

def main():
    parser = argparse.ArgumentParser(description="Process blockwise with a given path.")
    parser.add_argument('path', type=str, help='The path to process blockwise')
    args = parser.parse_args()
    
    config_yaml = args.path
    segment_blockwise(config_yaml)



if __name__ == "__main__":
    main()
