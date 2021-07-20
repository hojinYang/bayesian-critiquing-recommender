#!/usr/bin/env bash
module load python/3.7
source ~/ENV/bin/activate

cd /home/hojin/projects/def-ssanner/hojin/vae-pe
python model_evaluate.py --model_name VAEmultilayer --data_name yelp --log_dir VAE_beta_multilayer --conf VAE_beta_multilayer.config
