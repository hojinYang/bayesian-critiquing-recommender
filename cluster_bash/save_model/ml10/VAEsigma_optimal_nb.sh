#!/usr/bin/env bash
module load python/3.7
source ~/ENV/bin/activate

cd /home/hojin/projects/def-ssanner/hojin/vae-pe
#cd ~/code/vae-pe
python model_save.py --model_name VAEsigma --data_name ml10 --data_dir fold0_valid/fold0 --log_dir VAEsigma_optimal_nb_fold0valid --conf VAEsigma_optimal_nb.config
