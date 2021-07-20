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
from utils.Params import Params
from utils.Dataset import Dataset
from utils.Logger import Logger
from utils.Evaluator import Evaluator
from utils.Trainer import Trainer
from utils.HPShelper import conf_dict_generator


DATA_PATH = Path("./data") 
LOG_PATH = Path("./saves") 
CONFIG_PATH = Path("./conf") 

def fit(experiment, model_name, dataset, log_dir, device, skip_eval):
    d = conf_dict_generator[model_name](experiment)
    d['skip_eval'] = skip_eval
    conf = Params()
    conf.update_dict(d)

    model_base = getattr(models, model_name)
    model = model_base(conf, dataset.num_users, dataset.num_items, device)
    
    evaluator = Evaluator()
    logger = Logger(log_dir)
    logger.info(conf)
    logger.info(dataset)

    trainer = Trainer(
        dataset=dataset,
        model=model,
        evaluator=evaluator,
        logger=logger,
        conf=conf
    )

    trainer.train()
    return (trainer.best_score['RMSE'], trainer.best_epoch, model)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='VAEsigma')
    parser.add_argument('--data_name', type=str, default='yelp')
    parser.add_argument('--data_dir', type=str, default='fold0')
    parser.add_argument('--log_dir', type=str, default='VAEsigma')
    parser.add_argument('--conf', type=str, default='VAE_beta_multilayer.config')
    parser.add_argument('--seed', type=int, default=201231)
    p = parser.parse_args()

    np.random.seed(p.seed)
    torch.random.manual_seed(p.seed)

    data_dir = DATA_PATH / p.data_name / p.data_dir
    log_dir = LOG_PATH / p.data_name / (p.data_dir + '-' + p.log_dir)
    config_dir = CONFIG_PATH / p.data_name / p.conf
    print(config_dir)

    with open(config_dir) as f:
        conf = json.load(f)
    project_name = p.data_name + '-' + 'save'

    dataset = Dataset(data_dir=data_dir)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

 
    experiment = Experiment(project_name=project_name)
    experiment.log_parameters(conf)

    rmse, epoch, model = fit(experiment, p.model_name, dataset, log_dir, device, skip_eval=True) 
    experiment.log_metric("rmse", rmse)
    experiment.log_metric("epoch", epoch)
    experiment.log_others({
        "model_desc": p.log_dir
    })
    if torch.cuda.is_available:
        model = model.to(torch.device('cpu'))
    torch.save(model, str(log_dir)+'.pt')

    experiment.end()
