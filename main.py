from process_blockwise.config import Config
import os

if __name__ == "__main__":

    config_path = "/groups/cellmap/cellmap/zouinkhim/process-blockwise/examples/configs/peroxisome_process_config.yaml"

    config = Config(config_path)
    config.show_config()
    steps = config.get_process_steps()
    print(steps)