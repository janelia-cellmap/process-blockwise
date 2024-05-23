rm -rf daisy_logs
rm -rf task.out
rm -rf task.err

bsub -P cellmap -J post_process_proxisome_master -n 14 -o task.out -e task.err python ../../process_blockwise/segment_blockwise.py /groups/cellmap/cellmap/zouinkhim/process-blockwise/examples/configs/peroxisome_process_config.yaml
