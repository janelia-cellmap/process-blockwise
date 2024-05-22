from process_blockwise.segment_worker import segment_worker
import os




if __name__ == "__main__":
    # get tmpdir from command line arguments
    config_path = "/groups/cellmap/cellmap/zouinkhim/process-blockwise/examples/configs/peroxisome_process_config.yaml"
    tmp_dir = "test"
    config_yaml = config_path
    current_step = "step_1"
    segment_worker(tmp_dir, config_yaml,current_step)
