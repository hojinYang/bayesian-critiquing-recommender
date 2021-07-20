from comet_ml import Experiment

import os
import numpy as np
import argparse
import torch
from pathlib import Path
import json
import models
import glob
import pandas as pd

from utils.Dataset import Dataset
from utils.KAVgenerator import KAVgenerator
from utils.Simulator import Simulator

DATA_PATH = Path("./data") 
MODEL_PATH = Path("./saves") 
CONFIG_PATH = Path("./conf_simulate")

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='yelp')
    parser.add_argument('--data_dir', type=str, default='fold0')
    parser.add_argument('--saved_model', type=str, default='VAE_beta_multilayer.pt')
    parser.add_argument('--conf', type=str, default='sim_abs_diff_neg1_noise0.config')
    parser.add_argument('--seed', type=int, default=201231)
    p = parser.parse_args()

    np.random.seed(p.seed)
    torch.random.manual_seed(p.seed)

    data_dir = DATA_PATH / p.data_name / p.data_dir
    model_dir = MODEL_PATH / p.data_name / (p.data_dir + '-'  + p.saved_model) 
    config_dir = CONFIG_PATH / p.data_name / p.conf
    print(config_dir)

    
    with open(config_dir) as f:
        conf = json.load(f)
    project_name = p.data_name + '-' + 'pe-cold'

    # load model
    model = torch.load(model_dir, map_location=torch.device('cpu'))
    item_embeddings = model.decoder.weight.detach().numpy()
    
    # load data
    dataset = Dataset(data_dir=data_dir, load_keyphrases=True)

    # generate keyphrase embedding
    k = KAVgenerator(dataset.train_item_keyphrase_matrix, item_embeddings, 20)
    keyphrase_embeddings = k.get_all_mean_kav(20, 20)

    #experiment = Experiment(project_name=project_name)
    #experiment.log_parameters(conf)
    dataset.train_item_keyphrase_matrix[dataset.train_item_keyphrase_matrix<20] = 0
    s = Simulator(dataset, model, keyphrase_embeddings, dataset.train_item_keyphrase_matrix, conf, alpha)
    r = s.simulate_hr()


    save_dir = MODEL_PATH / p.data_name /  (p.data_dir + '-' + p.conf.split(".")[0] + '.csv')
    pd.DataFrame(r).to_csv(str(save_dir), index=False)

    #experiment.end()