import numpy as np
import torch
from .SimulatorUtils import get_select_query_func, update_posterior, update_posterior_logistic
from tqdm import tqdm
class Simulator:
    def __init__(self, dataset, model, keyphrase_embeddings, item_keyphrase_matrix, sim_conf,alpha):
        self.dataset = dataset
        self.model = model
        self.keyphrase_embeddings = keyphrase_embeddings
        self.embedding_size = keyphrase_embeddings.shape[1]
        self.item_keyphrase_matrix = item_keyphrase_matrix

        self.steps = sim_conf['steps']
        self.select_query = get_select_query_func[sim_conf['query_type']]

        self.rating_threshold = sim_conf['rating_threshold']
        self.keyphrase_threshold = sim_conf['keyphrase_threshold']
        self.diff = sim_conf['diff']
        self.response_noise = sim_conf['response_noise']
        self.k = sim_conf['k']
        self.pos_prec = sim_conf['pos_prec']
        self.neg_prec = sim_conf['neg_prec']

        self.sim_conf = sim_conf
        self.alpha=alpha

    def simulate_hr(self):
        result = {'HR@{}'.format(_k):[] for _k in self.k}
        result['Steps'] = list(range(self.steps+1))
        test_matrix = self.dataset.test_matrix >= self.rating_threshold
        item_keyphrase_matrix = self.item_keyphrase_matrix > 0
        print(np.sum(item_keyphrase_matrix))
        
        users, items = np.nonzero(test_matrix)
        print(len(items))
        
        pos = (item_keyphrase_matrix.sum(axis=1) >= self.keyphrase_threshold).nonzero()[0]
        print(len(pos))
        print(self.keyphrase_threshold)
        print(item_keyphrase_matrix.shape)
        mask = np.isin(items, pos)
        items = items[mask]
        users = users[mask]
        t = 0
        for u, i in tqdm(zip(users, items),total=len(items)):

            #if self.dataset.test_matrix[u,i] - self.diff >= self.get_user_item_pred(u, i) :

            #r = self.simulate_user_hr_logistic(u, i,alpha=self.alpha)
            r = self.simulate_user_hr(u, i)
            for _k in self.k:
                result['HR@{}'.format(_k)].append(r[_k])


        for _k in self.k:
            avg = np.mean(result['HR@{}'.format(_k)],axis=0)
            ci = 1.96 * np.std(result['HR@{}'.format(_k)], axis=0) / np.sqrt(len(items))
            result['HR@{}'.format(_k)] = avg
            result['HR@{}_CI'.format(_k)] = ci

    
        return result
    


    def simulate_user_hr(self, user_id, target_item_id):
        result = {_k:[] for _k in self.k}
        asked_queries = [] # store indexes of keyphrase asked in prevous step to avoid redundant query
        mu, S = self.get_mu_S(user_id)
        #prec_y = np.array(np.linalg.norm(0.01/(S+1e-6)))
        

        neg = np.min(mu.T @ self.keyphrase_embeddings.T)
        pos = np.max(mu.T @ self.keyphrase_embeddings.T)
        _, relevant_keyphrases = np.nonzero(self.item_keyphrase_matrix[target_item_id])
        #print(relevant_keyphrases)

        # initial hr
        pred_sorted_items, _ = self.get_user_preds_using_mu(mu, user_id)
        top_item = pred_sorted_items[:10]
        
        for _k in self.k:
            result[_k].append(hr_k(pred_sorted_items,target_item_id, _k))
        
        for j in range(1, self.steps+1):
            # for each step:
            #   1. system queries
            #   2. get user response
            #   3. update user belief
            #   4. compute HR

            #generate query candidates
            sorted_query_candidates = self.select_query(
                item_keyphrase_matrix=self.item_keyphrase_matrix,
                items = top_item,
                target_item = target_item_id
                )
            #remove redundant queries
            reduns = np.isin(sorted_query_candidates, asked_queries).nonzero()[0]
            #query_idx = relevant_keyphrases[j-1]
            query_idx = np.delete(sorted_query_candidates, reduns)[0]
            #print(query_idx)
            #query embedding
            x = self.keyphrase_embeddings[query_idx][:,np.newaxis]
            
            #get user response
            s = np.random.uniform()
            if s < self.response_noise:
                y = pos if np.random.uniform() > 0.5 else neg
            else:
                y = pos if np.isin(query_idx, relevant_keyphrases) else neg
            
            prec_y = self.pos_prec if y == pos else self.neg_prec 
            #update user belief
            mu, S =  update_posterior(x, y, mu, S, prec_y)

            #new HR
            pred_sorted_items, _ = self.get_user_preds_using_mu(mu, user_id)
            top_item = pred_sorted_items[:10]
            
            for _k in self.k:
                result[_k].append(hr_k(pred_sorted_items,target_item_id, _k))

            asked_queries.append(query_idx)

        return result

    def simulate_user_hr_logistic(self, user_id, target_item_id, alpha):
        result = {_k:[] for _k in self.k}
        asked_queries = [] # store indexes of keyphrase asked in prevous step to avoid redundant query
        mu, S = self.get_mu_S(user_id)
        #prec_y = np.array(np.linalg.norm(0.01/(S+1e-6)))
        

        #neg = np.min(mu.T @ self.keyphrase_embeddings.T)
        #pos = np.max(mu.T @ self.keyphrase_embeddings.T)
        _, relevant_keyphrases = np.nonzero(self.item_keyphrase_matrix[target_item_id])
        #print(relevant_keyphrases)

        # initial hr
        pred_sorted_items, _ = self.get_user_preds_using_mu(mu, user_id)
        top_item = pred_sorted_items[:10]
        
        for _k in self.k:
            result[_k].append(hr_k(pred_sorted_items,target_item_id, _k))
        
        for j in range(1, self.steps+1):
            # for each step:
            #   1. system queries
            #   2. get user response
            #   3. update user belief
            #   4. compute HR

            #generate query candidates
            sorted_query_candidates = self.select_query(
                item_keyphrase_matrix=self.item_keyphrase_matrix,
                items = top_item,
                target_item = target_item_id
                )
            #remove redundant queries
            reduns = np.isin(sorted_query_candidates, asked_queries).nonzero()[0]
            #query_idx = relevant_keyphrases[j-1]
            query_idx = np.delete(sorted_query_candidates, reduns)[0]
            #print(query_idx)
            #query embedding
            x = self.keyphrase_embeddings[query_idx][:,np.newaxis]
            
            #get user response
            s = np.random.uniform()
            if s < self.response_noise:
                y = 1.0 if np.random.uniform() > 0.5 else 0.0
            else:
                y = 1.0 if np.isin(query_idx, relevant_keyphrases) else 0.0
            
            prec_y = self.pos_prec if y == 1.0 else self.neg_prec 
            #update user belief
            mu, S =  update_posterior_logistic(x, y, mu, S, prec_y, alpha)

            #new HR
            pred_sorted_items, _ = self.get_user_preds_using_mu(mu, user_id)
            top_item = pred_sorted_items[:10]
            
            for _k in self.k:
                result[_k].append(hr_k(pred_sorted_items,target_item_id, _k))

            asked_queries.append(query_idx)

        return result


    def get_user_preds_using_mu(self, user_mu, user_id=None):
        '''
        user_mu: hidden_dim by 1
        '''

        _mu = torch.FloatTensor(user_mu.T) # 1 by hidden_dim

        with torch.no_grad():
            preds = self.model.decoder(_mu)

        preds = np.asarray(preds).reshape(-1)

        if user_id is not None:
            _, user_input = np.nonzero(self.dataset.train_matrix[user_id])
            preds[user_input] = -np.inf    

        sorted_pred_items = preds.argsort()[::-1]
        sorted_pred_ratings = preds[sorted_pred_items] 
        return sorted_pred_items, sorted_pred_ratings

    def get_user_item_pred(self, user_id, item_id):
        user_input = self.dataset.train_matrix[user_id]
        i = torch.FloatTensor(user_input.toarray()).to(torch.device('cpu'))
        with torch.no_grad():
            preds = self.model.forward(i).cpu().numpy().reshape(-1)

        return preds[item_id]

    def get_mu_S(self, user_id):
        user_input = self.dataset.train_matrix[user_id]
        i = torch.FloatTensor(user_input.toarray()).to(torch.device('cpu'))
        with torch.no_grad():
            mu, logvar = self.model.get_mu_logvar(i)
            std = self.model.logvar2std(logvar)
        mu, std = mu.numpy().T, std.numpy()

        return mu, np.diagflat(std*std)

def hr_k(preds, target, k):
    if target in set(preds[:k]):
        return 1
    else:
        return 0

    