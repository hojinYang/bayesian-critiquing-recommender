import os
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
import random
from pathlib import Path
from tqdm import tqdm
import json


def preprocess_movielens(data_dir, ratings, tags, movies, save_dir, num_folds=5, order_by_popularity=True, tag_freq_threshold=50):
    """
    Read raw movielens data.
    """
    print('Movielens preprocess starts.')
    print("Loading the dataset from \"%s\"" % data_dir)
    data_path = Path(data_dir) / ratings
    tags_path = Path(data_dir) / tags
    movies_path = Path(data_dir) / movies
    data = pd.read_csv(data_path, sep='::', names=['user', 'item', 'rating', 'timestamps'],
                                                dtype={'user': int, 'item': int, 'rating': float, 'timestamps': float},
                                                engine='python')

    tags = pd.read_csv(tags_path, sep='::', names=['user', 'item', 'tag', 'timestamps'],
                                                dtype={'user': int, 'item': int, 'tag' : str, 'timestamps': float},
                                                engine='python')
    tags['tag'] = tags['tag'].astype(str)
    tags['tag'] = tags['tag'].apply(lambda x: x.lower().replace('.', ''))
    movies = pd.read_csv(movies_path, sep='::', names=['item', 'title', 'genre'],
                                                dtype={'item': int, 'title': str, 'genre' : str},
                                                engine='python')

    # id re-assignment
    data, user_id_dict, item_id_dict = assign_id(data, order_by_popularity)
    tags, tag_id_dict = assign_tag_id(tags, data, user_id_dict, item_id_dict, tag_freq_threshold)
    title_id_dict = assign_movie_id(movies, item_id_dict)
    # split data and tag
    data = split_data(data, num_folds)
    tags = split_tags(tags, data)
    print(data)
    print(tags)

    num_users, num_items = len(user_id_dict), len(item_id_dict)
    save_folds(data, num_folds, num_users, num_items, save_dir)
    save_folds_tags(tags,num_folds, save_dir)

    num_unique_tags = len(tag_id_dict)
    num_ratings, num_tags = len(data), len(tags)

    info_lines = []
    info_lines.append('# users: %d, # items: %d, # ratings: %d' % (num_users, num_items, num_ratings))
    info_lines.append("Sparsity : %.2f%%" % ((1 - (num_ratings / (num_users * num_items))) * 100))
    info_lines.append('# tags: %d, # unique tags: %d' % (num_tags, num_unique_tags))
    
    stat_path = save_dir /'stat.txt'
    with open(stat_path, 'wt') as f:
        f.write('\n'.join(info_lines))

    with open(save_dir / 'user_id_dict.json', 'w') as fp:
        json.dump(user_id_dict, fp)

    with open(save_dir / 'item_id_dict.json', 'w') as fp:
        json.dump(item_id_dict, fp)

    with open(save_dir / 'tag_id_dict.json', 'w') as fp:
        json.dump(tag_id_dict, fp)

    with open(save_dir / 'title_id_dict.json', 'w') as fp:
        json.dump(title_id_dict, fp)

    generate_valid_set(data, tags, num_users, num_items, num_folds, save_dir)

    print('Preprocess finished.')


def preprocess_yelp(data_dir, ratings, tags, business, save_dir, num_folds=5, order_by_popularity=True, tag_freq_threshold=50):
    """
    Read raw movielens data.
    """
    print('Yelp preprocess starts.')
    print("Loading the dataset from \"%s\"" % data_dir)
    data_path = Path(data_dir) / ratings
    tags_path = Path(data_dir) / tags
    business_path = Path(data_dir) / business
    data = pd.read_csv(data_path, header=0, names=['user', 'item', 'rating'],
                                                dtype={'user': int, 'item': int, 'rating': float},
                                                engine='python')

    tags = pd.read_csv(tags_path, header=0, names=['user', 'item', 'tag'],
                                                dtype={'user': int, 'item': int, 'tag' : str},
                                                engine='python')
    tags['tag'] = tags['tag'].astype(str)

    business = pd.read_csv(business_path, header=0, names=['item', 'name'],
                                                dtype={'item': int, 'name': str},
                                                engine='python')
    business['name'] = business['name'].astype(str)


    # id re-assignment
    data, user_id_dict, item_id_dict = assign_id(data, order_by_popularity)
    tags, tag_id_dict = assign_tag_id(tags, data, user_id_dict, item_id_dict, tag_freq_threshold)
    id_business_dict = assign_business_id(business, item_id_dict)
    # split data and tag
    data = split_data(data, num_folds)
    tags = split_tags(tags, data)
    print(data)
    print(tags)

    num_users, num_items = len(user_id_dict), len(item_id_dict)
    save_folds(data, num_folds, num_users, num_items, save_dir)
    save_folds_tags(tags,num_folds, save_dir)

    num_unique_tags = len(tag_id_dict)
    num_ratings, num_tags = len(data), len(tags)

    info_lines = []
    info_lines.append('# users: %d, # items: %d, # ratings: %d' % (num_users, num_items, num_ratings))
    info_lines.append("Sparsity : %.2f%%" % ((1 - (num_ratings / (num_users * num_items))) * 100))
    info_lines.append('# tags: %d, # unique tags: %d' % (num_tags, num_unique_tags))
    
    stat_path = save_dir /'stat.txt'
    with open(stat_path, 'wt') as f:
        f.write('\n'.join(info_lines))

    with open(save_dir / 'user_id_dict.json', 'w') as fp:
        json.dump(user_id_dict, fp)

    with open(save_dir / 'item_id_dict.json', 'w') as fp:
        json.dump(item_id_dict, fp)

    with open(save_dir / 'tag_id_dict.json', 'w') as fp:
        json.dump(tag_id_dict, fp)

    with open(save_dir / 'id_business_dict.json', 'w') as fp:
        json.dump(id_business_dict, fp)

    generate_valid_set(data, tags, num_users, num_items, num_folds, save_dir)

    print('Preprocess finished.')


def generate_valid_set(data, tags, num_users, num_items, num_folds, save_dir, holdout_fold=0):
    save_dir = save_dir / ('fold%d_valid'%(holdout_fold)) 
    save_dir.mkdir(parents=True, exist_ok=True)
    data = data[data.fold != holdout_fold][['user','item','rating']]
    tags = tags[tags.fold != holdout_fold][['user','item','tag']]

    data = split_data(data, num_folds)
    tags = split_tags(tags, data)

    save_folds(data, 1, num_users, num_items, save_dir)
    save_folds_tags(tags, 1, save_dir)

def assign_id(data, order_by_popularity):
    """
    Assign old user/item id into new consecutive ids.    
    """

    # initial # user, items
    num_users = len(pd.unique(data.user))
    num_items = len(pd.unique(data.item))

    print('initial user, item:', num_users, num_items)
    
    num_items_by_user = data.groupby('user', as_index=False).size()
    num_items_by_user = num_items_by_user.set_index('user')
    
    num_users_by_item = data.groupby('item', as_index=False).size()
    num_users_by_item = num_users_by_item.set_index('item')

    # assign new user id
    print('Assign new user id...')
    user_frame = num_items_by_user
    user_frame.columns = ['item_cnt']
    if order_by_popularity: 
        user_frame = user_frame.sort_values(by='item_cnt', ascending=False)
    user_frame['new_id'] = list(range(num_users))

    frame_dict = user_frame.to_dict()
    user_id_dict = frame_dict['new_id']
    user_frame = user_frame.set_index('new_id')
    user_to_num_items = user_frame.to_dict()['item_cnt']

    data.user = [user_id_dict[x] for x in  data.user.tolist()]
    
    # assign new item id
    print('Assign new item id...')
    item_frame = num_users_by_item
    item_frame.columns = ['user_cnt']
    if order_by_popularity: 
        item_frame = item_frame.sort_values(by='user_cnt', ascending=False)
    item_frame['new_id'] = range(num_items)

    frame_dict = item_frame.to_dict()
    item_id_dict = frame_dict['new_id']
    item_frame = item_frame.set_index('new_id')
    item_to_num_users = item_frame.to_dict()['user_cnt']

    data.item = [item_id_dict[x] for x in  data.item.tolist()]

    return data, user_id_dict, item_id_dict

def assign_tag_id(tags, data, user_id_dict, item_id_dict, tag_freq_threshold):
    print('Assign new tag id...')
    print('Original # rows of tags dataset: %d'%(len(tags)))
    tags.user = [user_id_dict.get(x, -1) for x in tags.user.tolist()]
    tags.item = [item_id_dict.get(x, -1) for x in tags.item.tolist()]
    
    tags = pd.merge(left=tags, right=data, how='inner', on=['user','item'])
    print('After removing rows only in tag datset: %d'%(len(tags)))
    
    if tag_freq_threshold > 1:
        #limit the vocabulary of tags to those that have been applied by at least "tag_item_threshold" items
        counter = tags.groupby('tag')['item'].apply(lambda x: len(set(x))).to_frame('count').reset_index()
        counter = counter[counter['count']>=tag_freq_threshold]
        tags = pd.merge(tags,counter,on='tag')[['user','item','tag']]

        #limit the vocabulary of tags to those that have been applied by at least "tag_user_threshold" users
        counter = tags.groupby('tag')['user'].apply(lambda x: len(set(x))).to_frame('count').reset_index()
        counter = counter[counter['count']>=tag_freq_threshold]
        tags = pd.merge(tags,counter,on='tag')[['user','item','tag']]
    
    print('After removing rows below thresholds: %d'%(len(tags)))
    tag_id_dict = {tag: id for id, tag in enumerate(pd.unique(tags['tag']))}
    tags.tag = [tag_id_dict[x] for x in tags.tag.tolist()]
    return tags, tag_id_dict

def assign_movie_id(movies, item_id_dict):
    print('Assign new movie id...')
    movies.item = [item_id_dict.get(x, -1) for x in movies.item.tolist()]
    movies = movies[movies.item != -1]
    title_id_dict = {title:id for title, id in zip(movies.title.tolist(), movies.item.tolist())}
    return title_id_dict

def assign_business_id(business, item_id_dict):
    print('Assign new business id...')

    business.item = [item_id_dict.get(x, -1) for x in business.item.tolist()]

    business = business[business.item != -1]

    id_business_dict = {id:title for title, id in zip(business.name.tolist(), business.item.tolist())}

    return id_business_dict

    
def split_data(data, num_folds):

    """
    Proprocess UIRT raw data into trainable form.
    Holdout feedbacks for test per user.
    Save preprocessed data.
    """

    # Split data into n-folds
    print('Split data into %d folds.' %(num_folds))
    data_group = data.groupby('user')

    fold_list = []
    num_passed_users = 0

    for _, group in tqdm(data_group):
        num_items_user = len(group)

        if num_items_user < num_folds:
            num_passed_users += 1 
            continue

        assigned_fold = np.zeros(num_items_user, dtype=int)        
        idx = list(range(num_items_user))
        random.shuffle(idx)
        
        delta = num_items_user//num_folds
 
        for k in range(num_folds):
            assigned_fold[idx[k*delta: (k+1)*delta]] = k

        group['fold'] = assigned_fold
        fold_list.append(group)

    data_with_fold_id = pd.concat(fold_list)
    print('# users rated less then %d items are removed: %d' % (num_folds, num_passed_users))
    return data_with_fold_id

def split_tags(tags, data):
    print('Split tag data into folds.')
    tags = pd.merge(left=tags, right=data, how='inner', on=['user','item'])[['user','item','tag','fold']]
    return tags 

def save_folds(fold_data, num_folds, num_users, num_items, save_path):
    for fold in range(num_folds):
        te_data = fold_data[fold_data['fold'] == fold]
        tr_data = fold_data[fold_data['fold'] != fold]
        fold_save_dir = save_path / ('fold%d'%(fold)) 
        fold_save_dir.mkdir(parents=True, exist_ok=True)
        save_data_to_sparse(te_data, num_users, num_items, fold_save_dir/'te_data.pkl')
        save_data_to_sparse(tr_data, num_users, num_items, fold_save_dir/'tr_data.pkl')

def save_folds_tags(fold_tags, num_folds, save_path):
    for fold in range(num_folds):
        te_tags = fold_tags[fold_tags['fold'] == fold]
        tr_tags = fold_tags[fold_tags['fold'] != fold]
        fold_save_dir = save_path / ('fold%d'%(fold)) 
        te_tags.to_csv(fold_save_dir / 'te_tags.csv')
        tr_tags.to_csv(fold_save_dir / 'tr_tags.csv')

    
def save_data_to_sparse(data, num_users, num_items, fold_save_path):
    sparse = df_to_sparse(data, shape=(num_users, num_items))
    print(sparse.shape)
    with open(fold_save_path, 'wb') as f:
        pickle.dump(sparse , f)

def save_tags(tags, save_path):
    tags.to_csv(save_path / 'tags.csv')


def df_to_sparse(df, shape):
    rows, cols = df.user, df.item
    values = df.rating

    sp_data = sp.csr_matrix((values, (rows, cols)), dtype='float64', shape=shape)

    #num_nonzeros = np.diff(sp_data.indptr)
    #rows_to_drop = num_nonzeros == 0
    #if sum(rows_to_drop) > 0:
    #    print('%d empty users are dropped from matrix.' % sum(rows_to_drop))
    #    sp_data = sp_data[num_nonzeros != 0]

    return sp_data


if __name__ == "__main__":

    np.random.seed(201231)
    random.seed(201231)
    '''
    data_dir = Path(__file__).resolve().parents[1] / "data" /"ml-10M100K"
    ratings = "ratings.dat"
    movies = "movies.dat"
    tags = "tags.dat"
    save_dir = Path(__file__).resolve().parents[1] / "data" / "ml10"
    save_dir.mkdir(parents=True, exist_ok=True)
    preprocess_movielens(data_dir, ratings, tags, movies, save_dir, num_folds=5, order_by_popularity=True, tag_freq_threshold=15)
    '''

    data_dir = Path(__file__).resolve().parents[1] / "data" /"yelp-toronto"
    ratings = "ratings.csv"
    business = "business_name.csv"
    tags = "tags.csv"
    save_dir = Path(__file__).resolve().parents[1] / "data" / "yelp"
    save_dir.mkdir(parents=True, exist_ok=True)
    preprocess_yelp(data_dir, ratings, tags, business, save_dir, num_folds=5, order_by_popularity=True, tag_freq_threshold=15)