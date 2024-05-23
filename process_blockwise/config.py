import yaml
import os
import tempfile
from datetime import datetime
import uuid
from .process_functions import process_functions

class ConfigError(Exception):
    pass

class Data:
    def __init__(self, data_dict):
        self.input_container = data_dict.get('input_container')
        self.in_dataset = data_dict.get('in_dataset')
        self.output_container = data_dict.get('output_container')
        self.output_group = data_dict.get('output_group')
        self.roi = data_dict.get('roi', None)
        self.context = data_dict.get('context', 0)

        self.validate_mandatory_parts()

    def validate_mandatory_parts(self):
        mandatory_fields = ['input_container', 'in_dataset', 'output_container', 'output_group']
        for field in mandatory_fields:
            if getattr(self, field) is None:
                raise ConfigError(f"Mandatory field '{field}' missing in 'data' section.")

    def __repr__(self):
        return (f"Data(input_container={self.input_container}, in_dataset={self.in_dataset}, "
                f"output_container={self.output_container}, output_group={self.output_group}, "
                f"roi={self.roi}, context={self.context})")

class Task:
    def __init__(self, task_dict):
        self.task_name = task_dict.get('task_name', None)
        if self.task_name is None:
            self.task_name = f"{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4()}"
        self.tmpdir = task_dict.get('tmpdir', None)
        if self.tmpdir is None:
            self.tmpdir = tempfile.mkdtemp()
        else:
            if not os.path.exists(self.tmpdir):
                os.makedirs(self.tmpdir)
        self.num_cpus = task_dict.get('num_cpus', 1)
        self.num_workers = task_dict.get('num_workers', 20)

    def __repr__(self):
        return (f"Task(task_name={self.task_name}, tmpdir={self.tmpdir}, "
                f"num_cpus={self.num_cpus}, num_workers={self.num_workers})")
    

class Mask:
    def __init__(self, mask_dict):
        self.container = mask_dict.get('container')
        self.dataset = mask_dict.get('dataset')
        self.threshold = mask_dict.get('threshold', None)
        self.dilate = mask_dict.get('dilate', None)
        self.erode = mask_dict.get('erode', None)
        self.resize = mask_dict.get('resize', None)

    def __repr__(self):
        return (f"Mask(container={self.container}, dataset={self.dataset}, threshold={self.threshold}, "
                f"dilate={self.dilate}, erode={self.erode}, resize={self.resize})")
    

class ProcessStep:
    def __init__(self, step_dict):
        self.params = step_dict.get('params', {})
        self.steps = step_dict.get('steps', {})

    def run(self):
        print(f"Running step with params: {self.params}")
        for step_name, step_args in self.steps.items():
            func = process_functions.get(step_name)
            if func:
                func(**step_args)
            else:
                raise ConfigError(f"Unknown process step: {step_name}")


class Config:
    def __init__(self, config_file):
        self.config_file = config_file
        self.config_data = self.load_config()
        self.task = Task(self.get_task_config())
        self.data = Data(self.get_data_config())
        self.masks = {name: Mask(mask) for name, mask in self.get_masks_config().items()}

    def validate_mandatory_parts(self):
        data_config = self.config_data.get('data', {})
        mandatory_fields = ['input_container', 'in_dataset', 'output_container', 'output_group']

        missing_fields = [field for field in mandatory_fields if field not in data_config]
        if missing_fields:
            raise ConfigError(f"Mandatory fields missing in 'data' section: {', '.join(missing_fields)}")



    def load_config(self):
        with open(self.config_file, 'r') as file:
            config_data = yaml.safe_load(file)
        return config_data

    def get_task_config(self):
        return self.config_data.get('task', {})

    def get_data_config(self):
        return self.config_data.get('data', {})

    def get_masks_config(self):
        return self.config_data.get('masks', {})

    def get_process_config(self):
        return self.config_data.get('process', {})

    def show_config(self):
        print("Task Configurations:")
        print(self.get_task_config())
        print("\nData Configurations:")
        print(self.get_data_config())
        print("\nMasks Configurations:")
        print(self.get_masks_config())
        print("\nProcess Configurations:")
        print(self.get_process_config())

    def get_process_steps(self):
        steps = []
        process_config = self.get_process_config()
        print("process_config: ",process_config)
        for step, elms in process_config.items():
            print("step: ",step)
            print("elms: ",elms)
            for step_name, step_args in elms.items():
                func = process_functions.get(step_name)
                if func:
                    steps.append((func, step_args))
                else:
                    raise ConfigError(f"Unknown process step: {step_name}")
        return steps
