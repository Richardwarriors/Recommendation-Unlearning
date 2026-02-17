import random
import time

import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from utils import kmeans_InBP, ot_cluster

def delete(ratings, del_type, del_per, min_inter_per_user=2):

    #np.random.seed(42)

    total_interactions = len(ratings)
    num_delete = int(total_interactions * del_per / 100)

    #user interactions count    
    user_counts = ratings['user'].value_counts()

    if del_type == 'random':
        target_users = user_counts.index.tolist()

    elif del_type == 'core':
        # top 5% user
        num_active = int(len(user_counts) * 0.05)
        target_users = user_counts.sort_values(ascending=False).index[:num_active]

    elif del_type == 'edge':
        # bottom 95% user
        num_active = int(len(user_counts) * 0.05)
        target_users = user_counts.sort_values(ascending=False).index[num_active:]

    else:
        raise ValueError("Unknown del_type")

    #delete interactions 
    deletable_idx = []

    for user in target_users:
        user_idx = ratings[ratings['user'] == user].index.tolist()
        max_delete = len(user_idx) - min_inter_per_user

        if max_delete > 0:
            deletable_idx.extend(user_idx[:max_delete])

    #random delete interactions from deletable_idx

    num_delete = min(num_delete, len(deletable_idx))
    delete_idx = np.random.choice(deletable_idx, num_delete, replace=False)

    return ratings.drop(index=delete_idx).reset_index(drop=True)


def readRating_full(train_dir, test_dir, del_type='random', del_per=5):
    #train_dir: /home/jiajie/Richard_He/CURE4Rec/data/ml-100k/train.csv
    train_ratings = pd.read_csv(train_dir, sep=',')
    train_ratings['rating'] = 1
    test_ratings = pd.read_csv(test_dir, sep=',')
    test_ratings['rating'] = 1


    if del_per > 0:
        train_ratings = delete(train_ratings, del_type, del_per, min_inter_per_user=2)

    # active and inactive
    user_counts = train_ratings['user'].value_counts()

    #计算5%的活跃用户和非活跃用户
    num_active_users = int(len(user_counts) * 5 / 100)

    active_users = user_counts.index[:num_active_users].tolist()
    inactive_users = user_counts.index[num_active_users:].tolist()
    
    active_ratings = test_ratings[test_ratings['user'].isin(active_users)].reset_index(drop=True)
    inactive_ratings = test_ratings[test_ratings['user'].isin(inactive_users)].reset_index(drop=True)

    train_rating_lists = [train_ratings['user'], train_ratings['item'], train_ratings['rating']]
    test_rating_lists = [test_ratings['user'], test_ratings['item'], test_ratings['rating']]

    active_ratings = [active_ratings['user'], active_ratings['item'], active_ratings['rating']]
    inactive_ratings = [inactive_ratings['user'], inactive_ratings['item'], inactive_ratings['rating']]

    return train_rating_lists, test_rating_lists, active_ratings, inactive_ratings


def readRating_group(train_dir, test_dir, del_type='random', del_per=5, learn_type='sisa', num_groups=5,
                     dataset='ml-1m', model_type='wmf'):
    
    train_ratings = pd.read_csv(train_dir, sep=',')
    train_ratings['rating'] = 1
    test_ratings = pd.read_csv(test_dir, sep=',')
    test_ratings['rating'] = 1

    if del_per > 0:
        train_ratings = delete(train_ratings, del_type, del_per, min_inter_per_user=2)

    ensemble_train = [train_ratings['user'], train_ratings['item'], train_ratings['rating']]
    ensemble_test = [test_ratings['user'], test_ratings['item'], test_ratings['rating']]

    # active and inactive
    user_counts = train_ratings['user'].value_counts()
    num_active_users = int(len(user_counts) * 5 / 100)

    active_users = user_counts.index[:num_active_users].tolist()
    inactive_users = user_counts.index[num_active_users:].tolist()

    if learn_type == 'sisa':
        #shuffle interaction
        train_ratings = train_ratings.sample(frac=1, random_state=42).reset_index(drop=True)

        total_inter = len(train_ratings)
        shard_size = total_inter // num_groups

        train_rating_groups = []
        test_rating_groups = []

        for i in range(num_groups):

            start = i * shard_size

            if i == num_groups - 1:
                end = total_inter
            else:
                end = (i + 1) * shard_size

            #interaction average shard
            train_g = train_ratings.iloc[start:end].reset_index(drop=True)

            #user in shard
            shard_users = train_g['user'].unique()

            #user in the shard test
            test_g = test_ratings[
                test_ratings['user'].isin(shard_users)
            ].reset_index(drop=True)

            train_rating_groups.append(train_g)
            test_rating_groups.append(test_g)
            
    #user embedding clustering
    elif learn_type == 'receraser':
        train_rating_groups, test_rating_groups = kmeans_InBP(train_ratings, test_ratings, dataset, num_groups, model_type)

    elif learn_type == 'ultrare':
        start_time = time.time()
        user_embeddings = np.load(f'results/user_emb/{dataset}_wmf_emb.npy', allow_pickle=True).item()
        unique_users = train_ratings['user'].unique()
        user_mat = np.array([user_embeddings[user_id][0] for user_id in unique_users])

        _, labels = ot_cluster(user_mat, num_groups)


        user_groups = [[] for _ in range(num_groups)]
        for user_id, label in zip(unique_users, labels):
            user_groups[int(label)].append(user_id)

        train_rating_groups = [train_ratings[train_ratings['user'].isin(group)].reset_index(drop=True) for group in
                               user_groups]
        test_rating_groups = [test_ratings[test_ratings['user'].isin(group)].reset_index(drop=True) for group in
                              user_groups]

        print(f'Grouping time: {time.time() - start_time}')

    active_groups = []
    inactive_groups = []

    for i, ratings in enumerate(test_rating_groups):
        active_ratings = ratings[ratings['user'].isin(active_users)].reset_index(drop=True)
        inactive_ratings = ratings[ratings['user'].isin(inactive_users)].reset_index(drop=True)
        active_groups.append(active_ratings)
        inactive_groups.append(inactive_ratings)
        print(f"Group {i + 1} active users: {len(active_ratings['user'].unique())}")
        print(f"Group {i + 1} inactive users: {len(inactive_ratings['user'].unique())}")

    train_rating_groups = [[ratings['user'], ratings['item'], ratings['rating']] for ratings in train_rating_groups]
    test_rating_groups = [[ratings['user'], ratings['item'], ratings['rating']] for ratings in test_rating_groups]
    active_groups = [[ratings['user'], ratings['item'], ratings['rating']] for ratings in active_groups]
    inactive_groups = [[ratings['user'], ratings['item'], ratings['rating']] for ratings in inactive_groups]

    return train_rating_groups, test_rating_groups, active_groups, inactive_groups, ensemble_test, ensemble_train


class RatingData(Dataset):
    def __init__(self, rating_array):
        super(RatingData, self).__init__()
        self.rating_array = rating_array

        self.users = self.rating_array[0].astype(int)
        self.items = self.rating_array[1].astype(int)
        self.ratings = self.rating_array[2].astype(float)

        self.total_users = self.users
        self.total_items = self.items
        self.total_ratings = self.ratings

        self.pos_dict = {user: set() for user in self.users}
        for user, item in zip(self.users, self.items):
            self.pos_dict[user].add(item)
        
        self.num_item = int(self.items.max()) + 1

    def __len__(self):
        return len(self.total_users)

    def __getitem__(self, idx):
        user = self.total_users[idx]
        item = self.total_items[idx]
        rating = self.total_ratings[idx]
        return (torch.tensor(user, dtype=torch.long),
                torch.tensor(item, dtype=torch.long),
                torch.tensor(rating, dtype=torch.float32))
    
    def ng_sample(self, num_neg=4):
        """
        negative sampling: for each positive sample, randomly sample num_neg negative samples with 4
        """
        neg_users = []
        neg_items = []
        neg_ratings = []

        for u, i in zip(self.users, self.items):
            for _ in range(num_neg):
                j = np.random.randint(self.num_item)
                while j in self.pos_dict[u]:
                    j = np.random.randint(self.num_item)
                neg_users.append(u)
                neg_items.append(j)
                neg_ratings.append(0)

        #positive samples + negative samples
        self.total_users = np.concatenate([self.users, np.array(neg_users)])
        self.total_items = np.concatenate([self.items, np.array(neg_items)])
        self.total_ratings = np.concatenate([self.ratings, np.array(neg_ratings)])


class PairData(Dataset):
    def __init__(self, rating_array, pos_dir):
        super(PairData, self).__init__()
        self.pos_dir = pos_dir
        self.rating_array = rating_array

        self.users = self.rating_array[0].astype(int)
        self.pos_items = self.rating_array[1].astype(int)
        self.neg_items = self.rating_array[1].astype(int)

        self.user_mapping = {index: i for i, index in enumerate(self.rating_array[0].unique())}
        self.pos_mapping = {index: i for i, index in enumerate(self.rating_array[1].unique())}
        src = [self.user_mapping[index] for index in rating_array[0]]
        dst = [self.pos_mapping[index] for index in rating_array[1]]

        self.edge_index = [[], []]
        for i in range(len(self.users)):
            self.edge_index[0].append(src[i])
            self.edge_index[1].append(dst[i])

        # self.ratings = self.rating_array[2].astype(float)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        pos = self.pos_items[idx]
        neg = self.neg_items[idx]
        return (torch.tensor(user, dtype=torch.long),
                torch.tensor(pos, dtype=torch.long),
                torch.tensor(neg, dtype=torch.long))

    def ng_sample(self, ng_sample):

        user_list = []

        neg_item_list = []

        new_rating_array = []
        num_item = max(self.rating_array[1].unique())

        pos_dict = np.load(self.pos_dir, allow_pickle=True).item()
        for userid in self.users:
            for i in range(ng_sample):
                j = np.random.randint(num_item)
                while j in pos_dict[userid]:
                    j = np.random.randint(num_item)
                user_list.append(userid)
                neg_item_list.append(j)

        # new_rating_array.append(pd.Series(neg_item_list))

        # self.neg_items = new_rating_array[1].astype(int)

        self.neg_items = neg_item_list


def loadData(data, batch=30000, n_worker=24, shuffle=True):
    '''
    Parameters
    ----------
    data:   RatingData object 
    '''

    return DataLoader(data, batch_size=batch, shuffle=shuffle, num_workers=n_worker)


def readSparseMat(dir, n_user, n_item, max_rating=5):
    ratings = pd.read_csv(dir, header=None, sep=',')

    row = ratings[0].astype(int).values
    col = ratings[1].astype(int).values
    val = ratings[2].astype(float).values / max_rating
    # ind = np.ones_like(val, dtype=int)

    val_mat = coo_matrix((val, (row, col)), shape=(n_user, n_item), dtype=np.float16)
    # ind_mat = coo_matrix((ind, (row, col)), shape=(n_user, n_item), dtype=np.float16)  # set to float! int will cause error in kmeans

    return val_mat.tocsr()  # , ind_mat.tocsr()


'''
if learn_type == 'sisa':

    unique_users = train_ratings['user'].unique()
    random.shuffle(unique_users)

    group_size = len(unique_users) // num_groups
    #average shard
    user_groups = [unique_users[i * group_size: (i + 1) * group_size] for i in range(num_groups)]

    if len(unique_users) % num_groups != 0:
        for i in range(len(unique_users) % num_groups):
            user_groups[i] = np.append(user_groups[i], unique_users[num_groups * group_size + i])

    train_rating_groups = [train_ratings[train_ratings['user'].isin(group)].reset_index(drop=True) for group in
                        user_groups]
    test_rating_groups = [test_ratings[test_ratings['user'].isin(group)].reset_index(drop=True) for group in
                        user_groups]

elif learn_type == 'receraser':
    start_time = time.time()
    #user_embeddings = [[][]]
    user_embeddings = np.load(f'results/user_emb/{dataset}_wmf_emb.npy', allow_pickle=True).item()
    unique_users = train_ratings['user'].unique()
    #item_embeddings = np.load(f'results/item_emb/{dataset}_wmf_emb.npy', allow_pickle=True).item()
    user_mat = np.array([user_embeddings[user_id][0] for user_id in unique_users])

    _, labels = kmeans(user_mat, num_groups, True, 30)

    user_groups = [[] for _ in range(num_groups)]
    for user_id, label in zip(unique_users, labels):
        user_groups[int(label)].append(user_id)

    train_rating_groups = [train_ratings[train_ratings['user'].isin(group)].reset_index(drop=True) for group in
                            user_groups]
    test_rating_groups = [test_ratings[test_ratings['user'].isin(group)].reset_index(drop=True) for group in
                            user_groups]

    print(f'Grouping time: {time.time() - start_time}')
'''
