"""
Yao Wu et al., Collaborative denoising auto-encoders for top-n recommender systems. WSDM 2016.
https://alicezheng.org/papers/wsdm16-cdae.pdf
"""
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.binomial import Binomial
from models.BaseModel import BaseModel
from utils.Tools import apply_activation

class AE(BaseModel):
    def __init__(self, model_conf, num_users, num_items, device):
        super(AE, self).__init__()
        self.hidden_dim = model_conf.hidden_dim
        self.act = model_conf.act
        self.sparse_normalization = model_conf.sparse_normalization
        self.num_users = num_users
        self.num_items = num_items
        self.device = device

        self.encoder = nn.Linear(self.num_items, self.hidden_dim)
        self.decoder = nn.Linear(self.hidden_dim, self.num_items)

        torch.nn.init.xavier_uniform_(self.encoder.weight)
        torch.nn.init.xavier_uniform_(self.decoder.weight)
        torch.nn.init.zeros_(self.encoder.bias)
        torch.nn.init.zeros_(self.decoder.bias)

        self.to(self.device)

    def forward(self, rating_matrix):
        # AE
        if self.sparse_normalization:
            deno = torch.sum(rating_matrix>0, axis=1, keepdim=True) + 1e-5
            rating_matrix = rating_matrix / deno 
        enc = self.encoder(rating_matrix)
        enc = apply_activation(self.act, enc)

        dec = self.decoder(enc)

        return dec

    def train_one_epoch(self, dataset, optimizer, batch_size, verbose):
        self.train()
        
        # user, item, rating pairs
        train_matrix = dataset.train_matrix

        num_training = train_matrix.shape[0]
        num_batches = int(np.ceil(num_training / batch_size))

        perm = np.random.permutation(num_training)

        loss = 0.0
        for b in range(num_batches):
            optimizer.zero_grad()

            if (b + 1) * batch_size >= num_training:
                batch_idx = perm[b * batch_size:]
            else:
                batch_idx = perm[b * batch_size: (b + 1) * batch_size]

            batch_matrix = torch.FloatTensor(train_matrix[batch_idx].toarray()).to(self.device)
            pred_matrix = self.forward(batch_matrix)

            # MMSE
            mask = batch_matrix != 0
            batch_loss = torch.sum((batch_matrix-(pred_matrix*mask))**2) / torch.sum(mask)
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss

            if verbose and b % 50 == 0:
                print('(%3d / %3d) loss = %.4f' % (b, num_batches, batch_loss))
        return loss

    def predict(self, dataset, test_batch_size):
        total_preds = []
        total_ys = []
        with torch.no_grad():
            input_matrix = dataset.train_matrix
            test_matrix = dataset.test_matrix

            num_data = input_matrix.shape[0]
            num_batches = int(np.ceil(num_data / test_batch_size))
            perm = list(range(num_data))
            for b in range(num_batches):
                if (b + 1) * test_batch_size >= num_data:
                    batch_idx = perm[b * test_batch_size:]
                else:
                    batch_idx = perm[b * test_batch_size: (b + 1) * test_batch_size]
                
                input_batch_matrix = torch.FloatTensor(input_matrix[batch_idx].toarray()).to(self.device)
                test_batch_matrix = torch.FloatTensor(test_matrix[batch_idx].toarray())

                pred_batch_matrix = self.forward(input_batch_matrix).cpu().numpy()
                preds = pred_batch_matrix[test_batch_matrix != 0]
                ys = test_batch_matrix[test_batch_matrix != 0]
                if len(ys) > 0:
                    total_preds.append(preds)
                    total_ys.append(ys)

        total_preds = np.concatenate(total_preds)
        total_ys = np.concatenate(total_ys)
        
        return total_preds, total_ys