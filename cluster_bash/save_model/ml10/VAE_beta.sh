#!/usr/bin/env bash
module load python/3.7
source ~/ENV/bin/activate

cd /home/hojin/projects/def-ssanner/hojin/vae-pe
#cd ~/code/vae-pe
python model_save.py --model_name VAE --data_name ml10 --data_dir fold0_valid/fold0 --log_dir VAE_beta_fold0valid --conf VAE_beta.config
