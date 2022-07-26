#!/bin/bash
bsub -n 4 -W 48:00 -o execution_output -R "rusage[mem=8192, ngpus_excl_p=4]" python HF_transformer.py --cfg configs/default_gpu.yaml
