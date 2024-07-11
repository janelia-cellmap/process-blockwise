# Blockwise postprocessing code using DaCapo blockwise tasks
LSF blockwise postprocessing library for processing predictions. No script are needed. you just need to provide a yaml file with the configuration of the postprocessing task.

Example postprocessing yaml file 
```yaml
task:
  task_name: "20240710_c-elegans-op50_ld"
  tmpdir: "/nrs/cellmap/zouinkhim/tmp_daisy_2/"
  num_cpus: 1
  num_workers: 200
  billing: "cellmap"
  # empty_tmpdir: True

data:
  input_container: '/nrs/cellmap/zouinkhim/predictions/c-elegen/op50/c_elegen_bw_op50_ld_scratch_0_300000.zarr'
  in_dataset: 'ld/ld'
  output_container: '/nrs/cellmap/zouinkhim/predictions/c-elegen/op50/jrc_c-elegans-bw-1_postprocessed.zarr'
  output_group: 'ld'
  # roi: "[320000:330000,100000:110000,10000:20000]"
  context: 8

process:
  step_1:
    # override: True
    skip: True
    params:
      type: segmentation
      save_edges: True
    steps:
      instances:
        gaussian_kernel: 4
        threshold: 0.7
  step_2:
    params:
      type: relabel
```

### Submitting the postprocessing job
```bash
$ process_blockwise config.yaml
```


---
Ps. The postprocessing code is still under development. And made for personal usage. If you have any issues or questions, please contact us