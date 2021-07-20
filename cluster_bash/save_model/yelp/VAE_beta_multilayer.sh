#!/usr/bin/env bash
module load python/3.7
source ~/ENV/bin/activate

cd /home/hojin/projects/def-ssanner/hojin/vae-pe
#cd ~/code/vae-pe
python model_save.py --model_name VAEmultilayer --data_name yelp --data_dir fold0 --log_dir VAE_beta_multilayer --conf VAE_beta_multilayer.config
python model_save.py --model_name VAEmultilayer --data_name yelp --data_dir fold1 --log_dir VAE_beta_multilayer --conf VAE_beta_multilayer.config
python model_save.py --model_name VAEmultilayer --data_name yelp --data_dir fold2 --log_dir VAE_beta_multilayer --conf VAE_beta_multilayer.config
python model_save.py --model_name VAEmultilayer --data_name yelp --data_dir fold3 --log_dir VAE_beta_multilayer --conf VAE_beta_multilayer.config
python model_save.py --model_name VAEmultilayer --data_name yelp --data_dir fold4 --log_dir VAE_beta_multilayer --conf VAE_beta_multilayer.config
