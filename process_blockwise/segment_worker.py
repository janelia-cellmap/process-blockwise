import os
import sys
from process_blockwise.config import *
import daisy
from funlib.persistence import Array, open_ds
from funlib.segment.arrays.replace_values import replace_values
import logging

logger = logging.getLogger(__file__)


import numpy as np

def load_elements(tmpdir):
        a = np.load(os.path.join(tmpdir, "elements.npz"))
        return a["nodes"], a["edges"], a["components"]

def relabel_in_block(array_in, array_out, old_values, new_values, block):
    a = array_in.to_ndarray(block.write_roi)
    new_values = new_values.astype(a.dtype)
   
    replace_values(a, old_values, new_values, inplace=True)
    array_out[block.write_roi] = a

def mask_value(array_out, value, block):
    a = array_out.to_ndarray(block.write_roi)
    a[a == value] = 0
    array_out[block.write_roi] = a

def segment_function(array_in, roi, steps):
    data = array_in.to_ndarray(roi, fill_value=0)
    for step_name, step_args in steps.items():
        func = process_functions.get(step_name)

        if func:
            data = func(data, **step_args)
        else:
            raise ConfigError(f"Unknown process step: {step_name}")
    return data



def segment_worker(tmpdir,config_yaml,current_step):
    client = daisy.Client()
    config = Config(config_yaml)
    process_config = config.get_process_config()
    step_dict = process_config.get(current_step, {})
    input_file, dataset = config.get_input_for_step(current_step)

    # input_file = config.data.input_container
    # dataset = config.data.in_dataset
    output_file = config.data.output_container
    out_dataset = os.path.join(config.data.output_group, current_step)

    print(f"input_file: {input_file}, dataset: {dataset}, output_file: {output_file}, out_dataset: {out_dataset}")
    # out_dataset = config.data.output_group
    context = config.data.context

    array_in = open_ds(input_file, dataset)
    array_out = open_ds(output_file, out_dataset, mode="a")
    # array_out = open_ds(output_file, config.data.output_group, mode="a")

    voxel_size = array_in.voxel_size

    block_size = np.array(array_in.data.chunks) * np.array(voxel_size)
    write_size = daisy.Coordinate(block_size)

    write_roi = daisy.Roi((0,) * len(write_size), write_size)

    read_roi = write_roi.grow(context, context)
    
    num_voxels_in_block = (read_roi / array_in.voxel_size).size


    save_edges = step_dict.get('params', {}).get('save_edges', False)
    
    task_type = step_dict.get('params', {}).get('type', None)

    if task_type is None:
        raise ValueError(f"Task type is not defined in the config file. params: {step_dict}, all configs: {process_config}")


    folder = os.path.join(tmpdir, "blocks")
    if not os.path.exists(folder):
        os.makedirs(folder)

    

    nodes_list = []
    edges_list = []

    if task_type == "segmentation":

        while True:
            with client.acquire_block() as block:
                if block is None:
                    break


                segmentation = segment_function(array_in, block.read_roi, step_dict['steps'])


                if save_edges:

                    id_bump = block.block_id[1] * num_voxels_in_block
                    segmentation += id_bump
                    segmentation[segmentation == id_bump] = 0


                # wrap segmentation into daisy array
                segmentation = Array(
                    segmentation, roi=block.read_roi, voxel_size=array_in.voxel_size
                )

                

                # store segmentation in out array
                array_out[block.write_roi] = segmentation[block.write_roi]

                if save_edges:
                    neighbor_roi = block.write_roi.grow(
                        array_in.voxel_size, array_in.voxel_size
                    )

                    # clip segmentation to 1-voxel context
                    segmentation = segmentation.to_ndarray(roi=neighbor_roi, fill_value=0)
                    neighbors = array_out.to_ndarray(roi=neighbor_roi, fill_value=0)

                    unique_pairs = []

                    for d in range(3):
                        slices_neg = tuple(
                            slice(None) if dd != d else slice(0, 1) for dd in range(3)
                        )
                        slices_pos = tuple(
                            slice(None) if dd != d else slice(-1, None) for dd in range(3)
                        )

                        pairs_neg = np.array(
                            [
                                segmentation[slices_neg].flatten(),
                                neighbors[slices_neg].flatten(),
                            ]
                        )
                        pairs_neg = pairs_neg.transpose()

                        pairs_pos = np.array(
                            [
                                segmentation[slices_pos].flatten(),
                                neighbors[slices_pos].flatten(),
                            ]
                        )
                        pairs_pos = pairs_pos.transpose()

                        unique_pairs.append(
                            np.unique(np.concatenate([pairs_neg, pairs_pos]), axis=0)
                        )

                if save_edges:
                    unique_pairs = np.concatenate(unique_pairs).astype(np.uint64)
                    zero_u = unique_pairs[:, 0] == 0
                    zero_v = unique_pairs[:, 1] == 0
                    non_zero_filter = np.logical_not(np.logical_or(zero_u, zero_v))


                    edges = unique_pairs[non_zero_filter]
                    nodes = np.unique(edges)


                    nodes_list.append(nodes)
                    edges_list.append(edges)
        
        if save_edges:
            nodes = np.concatenate(nodes_list)
            edges = np.concatenate(edges_list)

            np.savez_compressed(
                os.path.join(folder, "block_%d.npz" % client.worker_id),
                nodes=nodes,
                edges=edges,
            )

    elif task_type == "relabel":
        nodes, edges, components = load_elements(tmpdir)
        while True:
            with client.acquire_block() as block:
                if block is None:
                    break
                print(f"Segmenting in block {block}")
                # mask_value(array_out, 2, block)
                relabel_in_block(array_in, array_out, nodes, components, block)

    print("worker finished.")


if __name__ == "__main__":
    # get tmpdir from command line arguments
    args = sys.argv[1:]
    tmp_dir = args[0]
    config_yaml = args[1]
    current_step = args[2]
    segment_worker(tmp_dir, config_yaml,current_step)
