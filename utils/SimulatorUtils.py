import numpy as np
import torch
from scipy.stats import norm

threshold = 50

def rank_by_mean(keyphrase_embeddings, mu, S, pred_y, **unused):
    pred_means, _ = get_predictive_dist(keyphrase_embeddings.T, mu, S, pred_y)
    candidate_index = np.argpartition(-pred_means, threshold)[:threshold]
    rank = candidate_index[pred_means[candidate_index].argsort()[::-1]]
    return rank

def rank_by_var(keyphrase_embeddings, mu, S, pred_y, **unused):
    _, pred_vars = get_predictive_dist(keyphrase_embeddings.T, mu, S, pred_y)
    candidate_index = np.argpartition(-pred_vars, threshold)[:threshold]
    rank = candidate_index[pred_vars[candidate_index].argsort()[::-1]]
    return rank

def rank_by_ts(keyphrase_embeddings, mu, S, pred_y, **unused):
    pred_means, pred_vars = get_predictive_dist(keyphrase_embeddings.T, mu, S, pred_y)
    pred_sds = np.sqrt(pred_vars)
    score = np.random.normal(pred_means, pred_sds)
    candidate_index = np.argpartition(-score, threshold)[:threshold]
    rank = candidate_index[score[candidate_index].argsort()[::-1]]
    return rank

def rank_by_ucb(keyphrase_embeddings, mu, S, pred_y, alpha, **unused):
    pred_means, pred_vars = get_predictive_dist(keyphrase_embeddings.T, mu, S, pred_y)
    pred_sds = np.sqrt(pred_vars)
    score = alpha*pred_sds + pred_means
    candidate_index = np.argpartition(-score, threshold)[:threshold]
    rank = candidate_index[score[candidate_index].argsort()[::-1]]
    return rank

def rank_by_random(item_keyphrase_matrix, **unused):
    num_keyphrases = item_keyphrase_matrix.shape[1]
    keyphrases = np.arange(num_keyphrases)
    np.random.shuffle(keyphrases)
    return keyphrases[:threshold]

def rank_by_pop(item_keyphrase_matrix, **unused):

    score = np.asarray(item_keyphrase_matrix.sum(axis=0)).reshape(-1)

    candidate_index = np.argpartition(-score, threshold)[:threshold]
    rank = candidate_index[score[candidate_index].argsort()[::-1]]
    return rank

def rank_by_recommended_pop(item_keyphrase_matrix, item, **unused):
    item = np.delete(item, np.where(item >= item_keyphrase_matrix.shape[0]))
    item = np.array(item)

    if item is None:
        return rank_by_pop(item_keyphrase_matrix)
    score = np.asarray(item_keyphrase_matrix.sum(axis=0)).reshape(-1)

    keyphrases = item_keyphrase_matrix[item].nonzero()[1]
    if len(keyphrases) != 0:
        score[keyphrases] += np.max(score)
    
    candidate_index = np.argpartition(-score, threshold)[:threshold]
    rank = candidate_index[score[candidate_index].argsort()[::-1]]

    return rank

def rank_by_abs_diff(item_keyphrase_matrix, items, target_item, **unused):
    valid_items = np.unique(item_keyphrase_matrix.nonzero()[0])
    items = np.intersect1d(items,valid_items)

    if len(items) == 0:
        avg_candidate_keyphrase = np.asarray(np.mean(item_keyphrase_matrix[valid_items].todense(),axis=0)).squeeze()
    else:
        avg_candidate_keyphrase = np.asarray(np.mean(item_keyphrase_matrix[items].todense(),axis=0)).squeeze()
    target_keyphrase = np.asarray(item_keyphrase_matrix[target_item].todense()).squeeze()

    abs_diff = np.abs(target_keyphrase - avg_candidate_keyphrase)
    
    diff = target_keyphrase - avg_candidate_keyphrase
    abs_diff[np.logical_and(target_keyphrase>0, diff<0)] = 0
    
    score = abs_diff

    #target ++ cand +: pos ok
    #target ++ cand 0: pos ok
    #target 0 cand ++ : neg ok
    #target + cand ++ : neg but not ok
    #target 0 cand 0
    
    candidate_index = np.argpartition(-score, threshold)[:threshold]
    rank = candidate_index[score[candidate_index].argsort()[::-1]]

    return rank


def rank_by_evoi(keyphrase_embeddings, mu, S, pred_y, model, split, prec_y, **unused):
    pred_means, pred_vars = get_predictive_dist(keyphrase_embeddings.T, mu, S, pred_y)
    pred_sds = np.sqrt(pred_vars)

    score = []
    for (loc, scale, query_emb) in zip(pred_means, pred_sds, keyphrase_embeddings):
        query_emb = query_emb[:,np.newaxis]
        expected_u = compute_evoi(loc, scale, split, query_emb, mu, S, prec_y, model)    
        score.append(expected_u)

    score = np.array(score)
   
    candidate_index = np.argpartition(-score, threshold)[:threshold]
    rank = candidate_index[score[candidate_index].argsort()[::-1]]
   
    return rank

def compute_evoi(loc, scale, split, query_emb, mu, S, prec_y, model):
    expected_utilities = []
    for i in range(split):
        lb = norm.ppf(q=i/split, loc=loc, scale=scale)
        ub = norm.ppf(q=(i+1)/split, loc=loc, scale=scale)
        expected_y = norm.expect(loc=loc, scale=scale, lb=lb, ub=ub)

        #update user belief
        _mu, _ =  update_posterior(query_emb, expected_y, mu, S, prec_y)
        _mu = torch.FloatTensor(_mu.T) # 1 by hidden_dim
        with torch.no_grad():
            preds = model.decoder(_mu)
        preds = np.asarray(preds).reshape(-1)
        sorted_pred_ratings = np.sort(preds)[::-1]
        expected_utilities.append(sorted_pred_ratings[0])
        
    return np.mean(expected_utilities)

def noiseless_response(true_y):
    return true_y


def get_predictive_dist(x_pred, mu, S, prec_y):
    '''
    X_pred: dim by num_keyphrases
    S: dim by dim
    mu: dim by 1
    prec: = scalar
    '''

    pred_means = mu.T @ x_pred
    pred_means = pred_means.flatten()
    pred_vars = np.sum(1/prec_y + x_pred.T @ S * x_pred.T, axis=1)

    return pred_means, pred_vars

def update_posterior(x, y, mu_0, S_0, prec_y):
    '''
    x: newly observed data, in our case keyphrase: dim by 1
    y: value(observed score), scalar
    mu_0: dim by 1
    S_0: dim by dim
    prec_y: scalar
    '''
    S_0_inv = np.linalg.inv(S_0)
    S_1 = np.linalg.inv(S_0_inv +prec_y * (x @ x.T))    
    #print(np.swapaxes(x,-2,-1).shape)
    #S_1 = np.linalg.inv(S_0_inv +prec_y * np.matmul(x,np.swapaxes(x,-2,-1)))
    mu_1 = S_1 @ (S_0_inv @ mu_0 + prec_y * y * x)
    #print(m_1.shape)
    return mu_1, S_1

def update_posterior_logistic(x, y, mu_0, S_0, prec_y, alpha):
    '''
    x: newly observed data, in our case keyphrase: dim by 1
    y: value(observed score), scalar
    mu_0: dim by 1
    S_0: dim by dim
    prec_y: scalar: 1 or 0
    '''
    mu_1 = torch.rand(mu_0.shape, requires_grad=True)
    mu_0 = torch.from_numpy(mu_0)
    S_0_inv = np.linalg.inv(S_0)
    S_0_inv = torch.from_numpy(S_0_inv)
    x = torch.from_numpy(x).type(torch.FloatTensor)
    
    min_loss = 10^5
    best_m1 = mu_1.detach().numpy()
    for i in range(100000):
        loss = - (-0.5 * (mu_1 - mu_0).T @ S_0_inv @ (mu_1-mu_0) + \
                torch.tensor(y) * torch.log(torch.sigmoid(alpha * mu_1.T@x)) + (torch.tensor(1.0-y))* torch.log(1.0- torch.sigmoid(alpha * mu_1.T@x))).squeeze()
        # print(loss)
        if i > 0 and abs(prev_loss -loss) < 1e-1:
            break
        if torch.isnan(loss):
            print('NAN')
            print(min_loss)
            break

        prev_loss = loss
        

        loss.backward()

        with torch.no_grad():
            #yelp 1: 1e-2 
            mu_1.data.sub_(1e-2*mu_1.grad.data)
            mu_1.grad.zero_()
            if loss < min_loss:
                min_loss = loss
                best_m1 = mu_1.detach().numpy()

            #if i % 100 == 0:
            #    print(loss)

   # mu_1 = mu_1.detach().numpy()
    #print('XX')
    #print(min_loss)

    mu_1 = best_m1
    x = x.numpy()
    S_1_inv = S_0_inv + alpha * y*(1-y)*(x@x.T)
    S_1 = np.linalg.inv(S_1_inv)

    return mu_1, S_1


    





get_select_query_func = {
    'mean': rank_by_mean,
    'var': rank_by_var,
    'ts': rank_by_ts,
    'ucb': rank_by_ucb,
    'random': rank_by_random,
    'evoi' : rank_by_evoi,
    'pop' : rank_by_pop,
    'recommended_pop' : rank_by_recommended_pop,
    'abs_diff': rank_by_abs_diff
}
