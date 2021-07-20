from comet_ml import Optimizer

import os
import numpy as np
import argparse
import torch
from pathlib import Path

import models
from utils.Params import Params
from utils.Dataset import Dataset
from utils.Logger import Logger
from utils.Evaluator import Evaluator
from utils.Trainer import Trainer
from utils.HPShelper import conf_dict_generator


DATA_PATH = Path("./data") 
LOG_PATH = Path("./saves") 
CONFIG_PATH = Path("./conf_hp_search") 

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
    return (trainer.best_score['RMSE'], trainer.best_epoch)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='VAEmultilayer')
    parser.add_argument('--data_name', type=str, default='yelp')
    parser.add_argument('--data_dir', type=str, default='fold0_valid/fold0')
    parser.add_argument('--log_dir', type=str, default='AE')
    parser.add_argument('--conf', type=str, default='VAE_beta_multilayer.config')
    parser.add_argument('--seed', type=int, default=201232)
    p = parser.parse_args()

    np.random.seed(p.seed)
    torch.random.manual_seed(p.seed)

    data_dir = DATA_PATH / p.data_name / p.data_dir
    log_dir = LOG_PATH / p.data_name / p.log_dir 
    config_dir = CONFIG_PATH / p.data_name / p.conf
    print(config_dir)
    project_name = p.data_name + '-' + 'valid-grid'

    opt = Optimizer(config_dir)
    dataset = Dataset(data_dir=data_dir)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    for experiment in opt.get_experiments(project_name=project_name):
        # Test the model
        print(experiment)
        experiment.log_others({
            "model_desc": p.log_dir
        })
        rmse, epoch = fit(experiment, p.model_name, dataset, log_dir, device,skip_eval=False) 
        experiment.log_metric("rmse", rmse)
        experiment.log_metric("epoch", epoch)

        

