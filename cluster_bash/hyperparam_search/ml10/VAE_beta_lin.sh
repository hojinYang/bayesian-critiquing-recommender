#!/usr/bin/env bash
module load python/3.7
source ~/ENV/bin/activate

cd /home/hojin/projects/def-ssanner/hojin/vae-pe
python hp_search.py --model_name VAE --data_name ml10 --data_dir fold0_valid/fold0 --log_dir VAE_beta_lin --conf VAE_beta_lin.config