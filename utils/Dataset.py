import os
import numpy as np
import torch
import scipy.sparse as sp
from pathlib import Path
import pickle
import pandas as pd

class Dataset:
    def __init__(self, data_dir, load_keyphrases = False):
        print('Read data from %s' % data_dir)
        self.train_matrix, self.test_matrix = self.load_data(data_dir)
        self.num_users = self.train_matrix.shape[0]
        self.num_items = self.train_matrix.shape[1]

        if load_keyphrases:
            self.train_item_keyphrase_matrix = self.load_keyphrases(data_dir)

    def load_data(self, data_path):
        
        with open(Path(data_path)/'tr_data.pkl', 'rb') as f:
            train_matrix = pickle.load(f)

        with open(Path(data_path)/'te_data.pkl', 'rb') as f:
            test_matrix = pickle.load(f)

        return train_matrix, test_matrix

    def load_keyphrases(self, data_path):
        df_tags = pd.read_csv(str(Path(data_path)/'tr_tags.csv'))
        rows, cols, values = df_tags.item, df_tags.tag, np.ones(len(df_tags))
        return sp.csr_matrix((values, (rows, cols)), dtype='float64')


    def eval_data(self):
        # eval_pos and eval_target
        return self.train_matrix, self.test_matrix

    def __str__(self):
        # return string representation of 'Dataset' class
        # print(Dataset) or str(Dataset)
        ret = '======== [Dataset] ========\n'
        # ret += 'Train file: %s\n' % self.train_file
        # ret += 'Test file : %s\n' % self.test_file
        ret += 'Number of Users : %d\n' % self.num_users
        ret += 'Number of items : %d\n' % self.num_items
        ret += '\n'
        return ret