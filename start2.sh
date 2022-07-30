#!/bin/bash
bsub -n 4 -W 10:00 -o execution_output -R "rusage[mem=8192, ngpus_excl_p=1]" python SCAL_transformer.py --cfg configs/default_gpu.yaml