"""
Dawen Liang et al., Variational Autoencoders for Collaborative Filtering. WWW 2018.
https://arxiv.org/pdf/1802.05814
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.BaseModel import BaseModel
from utils.Tools import apply_activation

class VAEmultilayer(BaseModel):
    def __init__(self, model_conf, num_users, num_items, device):
        super(VAEmultilayer, self).__init__()

        self.hidden_dim = model_conf.hidden_dim

        self.num_users = num_users
        self.num_items = num_items
        self.sparse_normalization = model_conf.sparse_normalization
        self.dropout_ratio = model_conf.dropout_ratio

        self.encoder = nn.ModuleList()
        self.encoder.append(nn.Linear(self.num_items, self.hidden_dim*4))
        self.encoder.append(nn.Tanh())
        self.encoder.append(nn.Linear(self.hidden_dim*4, self.hidden_dim*2))
        for layer in self.encoder:
            if 'weight' in dir(layer):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
                
        self.decoder = nn.Linear(self.hidden_dim, self.num_items)
        torch.nn.init.xavier_uniform_(self.decoder.weight)
        torch.nn.init.zeros_(self.decoder.bias)
        
        self.total_anneal_steps = model_conf.total_anneal_steps
        self.anneal_cap = model_conf.anneal_cap

        self.anneal = 0.
        self.update_count = 0

        self.device = device

        self.to(self.device)

    def forward(self, rating_matrix):
        # encoder

        mu_q, logvar_q = self.get_mu_logvar(rating_matrix)
        std_q = self.logvar2std(logvar_q)
        eps = torch.randn_like(std_q)
        sampled_z = mu_q + self.training * eps * std_q

        output = self.decoder(sampled_z)

        if self.training:
            # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            #not averaged yet
            kl_loss = -0.5 * torch.sum(1 + logvar_q - mu_q.pow(2) - logvar_q.exp())
            return output, kl_loss
        else:
            return output

    def get_mu_logvar(self, rating_matrix):

        if self.training and self.dropout_ratio >0 :
            rating_matrix = F.dropout(rating_matrix, p=self.dropout_ratio) * (1 - self.dropout_ratio)

        if self.sparse_normalization:
            deno = torch.sum(rating_matrix>0, axis=1, keepdim=True) + 1e-5
            rating_matrix = rating_matrix / deno
        h = rating_matrix
        for layer in self.encoder:
            h = layer(h)
        mu_q = h[:, :self.hidden_dim]
        logvar_q = h[:, self.hidden_dim:]  # log sigmod^2  
        return mu_q, logvar_q

    def logvar2std(self, logvar):
        return torch.exp(0.5 * logvar)  # sigmod 




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

            if self.total_anneal_steps > 0:
                self.anneal = min(self.anneal_cap, 1. * self.update_count / self.total_anneal_steps)
            else:
                self.anneal = self.anneal_cap

            pred_matrix, kl_loss = self.forward(batch_matrix)

            # Gaussian log-likelihood loss
            mask = batch_matrix != 0
            log_sigma = torch.zeros([], device=pred_matrix.device)
            recon_loss = torch.sum(gaussian_nll(pred_matrix, log_sigma, batch_matrix) * mask)            
            
            # vae loss with annealing
            batch_loss = (recon_loss + kl_loss * self.anneal) / torch.sum(mask)

            batch_loss.backward()
            optimizer.step()

            self.update_count += 1

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

def gaussian_nll(mu, log_sigma, x):
    return 0.5 * torch.pow((x - mu) / log_sigma.exp(), 2) + log_sigma + 0.5 * np.log(2 * np.pi)