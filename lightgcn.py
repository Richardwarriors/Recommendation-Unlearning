import argparse
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn, optim, Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import structured_negative_sampling
from torch_sparse import SparseTensor, matmul

from utils import kmeans_InBP, kmeans_ot_InBP


def load_data(root, index_col):
    df = pd.read_csv(root)
    unique_ids = df[index_col].unique()
    mapping = {raw_id: i for i, raw_id in enumerate(unique_ids)}
    return mapping

#remove interactions of users
def delete_interactions(ratings, del_type, del_per, min_inter_per_user=2):

    total_interactions = len(ratings)
    num_delete = int(total_interactions * del_per / 100)

    user_counts = ratings['user'].value_counts()

    if del_type == 'random':
        target_users = user_counts.index.tolist()

    elif del_type == 'core':
        num_active = int(len(user_counts) * 0.05)
        target_users = user_counts.sort_values(ascending=False).index[:num_active]

    elif del_type == 'edge':
        num_active = int(len(user_counts) * 0.05)
        target_users = user_counts.sort_values(ascending=False).index[num_active:]

    else:
        raise ValueError("Unknown del_type")

    deletable_idx = []

    for user in target_users:
        user_idx = ratings[ratings['user'] == user].index.tolist()
        max_delete = len(user_idx) - min_inter_per_user

        if max_delete > 0:
            deletable_idx.extend(user_idx[:max_delete])

    num_delete = min(num_delete, len(deletable_idx))
    delete_idx = np.random.choice(deletable_idx, num_delete, replace=False)

    return ratings.drop(index=delete_idx).reset_index(drop=True)

def load_edge(df, src_mapping, dst_mapping, rating_threshold=1):

    if 'rating' in df.columns:
        df = df[df['rating'] >= rating_threshold]

    src = df['user'].map(src_mapping).astype(np.int64).values
    dst = df['item'].map(dst_mapping).astype(np.int64).values

    edge_np = np.stack((src, dst), axis=0)  # [2, E]

    return torch.from_numpy(edge_np).long()

#if retrain
def readRating_full_lightgcn(train_dir, test_dir,
                             user_mapping, item_mapping,
                             del_type='random', del_per=5):

    train_ratings = pd.read_csv(train_dir)
    train_ratings['rating'] = 1

    test_ratings = pd.read_csv(test_dir)
    test_ratings['rating'] = 1

    # interaction delete
    if del_per > 0:
        train_ratings = delete_interactions(train_ratings, del_type, del_per)

    # active / inactive 
    user_counts = train_ratings['user'].value_counts()
    num_active_users = int(len(user_counts) * 5 / 100)

    active_users = user_counts.index[:num_active_users].tolist()
    inactive_users = user_counts.index[num_active_users:].tolist()

    active_test = test_ratings[test_ratings['user'].isin(active_users)].reset_index(drop=True)
    inactive_test = test_ratings[test_ratings['user'].isin(inactive_users)].reset_index(drop=True)

    # ==== strucutre graph in lightgcn edge_index ====
    train_edge = load_edge(train_ratings, user_mapping, item_mapping)
    test_edge = load_edge(test_ratings, user_mapping, item_mapping)
    active_edge = load_edge(active_test, user_mapping, item_mapping)
    inactive_edge = load_edge(inactive_test, user_mapping, item_mapping)

    return train_edge, test_edge, active_edge, inactive_edge

def readRating_group_lightgcn(train_dir, test_dir,
                              user_mapping, item_mapping,
                              del_type='random',
                              del_per=5,
                              learn_type='sisa',
                              num_groups=10,
                              dataset='ml-1m',
                              model_type='lightgcn'):

    train_ratings = pd.read_csv(train_dir)
    train_ratings['rating'] = 1

    test_ratings = pd.read_csv(test_dir)
    test_ratings['rating'] = 1

    # ========================
    # interaction delete
    # ========================

    if del_per > 0:
        train_ratings = delete_interactions(train_ratings, del_type, del_per)

    
    ensemble_train_df = train_ratings.copy()
    ensemble_test_df = test_ratings.copy()

    # ========================
    # active / inactive
    # ========================

    user_counts = train_ratings['user'].value_counts()
    num_active_users = int(len(user_counts) * 5 / 100)

    active_users = user_counts.index[:num_active_users].tolist()
    inactive_users = user_counts.index[num_active_users:].tolist()

    # ========================
    # shard
    # ========================

    if learn_type == 'sisa':

        train_ratings = train_ratings.sample(frac=1, random_state=42).reset_index(drop=True)

        total_inter = len(train_ratings)
        shard_size = total_inter // num_groups

        train_rating_groups = []
        test_rating_groups = []

        for i in range(num_groups):

            start = i * shard_size
            end = total_inter if i == num_groups - 1 else (i + 1) * shard_size

            train_g = train_ratings.iloc[start:end].reset_index(drop=True)

            shard_users = train_g['user'].unique()

            test_g = test_ratings[
                test_ratings['user'].isin(shard_users)
            ].reset_index(drop=True)

            train_rating_groups.append(train_g)
            test_rating_groups.append(test_g)

    elif learn_type == 'receraser':
        train_rating_groups, test_rating_groups = \
            kmeans_InBP(train_ratings, test_ratings, dataset, num_groups, model_type)

    elif learn_type == 'ultrare':
        train_rating_groups, test_rating_groups = \
            kmeans_ot_InBP(train_ratings, test_ratings, dataset, num_groups, model_type)

    else:
        raise ValueError("Unknown learn_type")

    # ========================
    # active / inactive per group
    # ========================

    active_groups = []
    inactive_groups = []

    for ratings in test_rating_groups:
        active_r = ratings[ratings['user'].isin(active_users)].reset_index(drop=True)
        inactive_r = ratings[ratings['user'].isin(inactive_users)].reset_index(drop=True)

        active_groups.append(active_r)
        inactive_groups.append(inactive_r)

    # ========================
    # edge_index
    # ========================

    train_edge_groups = [load_edge(r, user_mapping, item_mapping)
                         for r in train_rating_groups]

    test_edge_groups = [load_edge(r, user_mapping, item_mapping)
                        for r in test_rating_groups]

    active_edge_groups = [load_edge(r, user_mapping, item_mapping)
                          for r in active_groups]

    inactive_edge_groups = [load_edge(r, user_mapping, item_mapping)
                            for r in inactive_groups]

    # ========================
    # ensemble edge - whole training set for receraser and ultrare, and testing for all ensemble unlearning types
    # ========================

    ensemble_train_edge = load_edge(ensemble_train_df, user_mapping, item_mapping)
    ensemble_test_edge = load_edge(ensemble_test_df, user_mapping, item_mapping)

    return (train_edge_groups,
            test_edge_groups,
            active_edge_groups,
            inactive_edge_groups,
            ensemble_train_edge,
            ensemble_test_edge)

def sample_batch(batch_size, edge_index):
    edges = structured_negative_sampling(edge_index)
    edges = torch.stack(edges, dim=0)
    indices = random.choices(
        [i for i in range(edges[0].shape[0])], k=batch_size)
    batch = edges[:, indices]
    user_indices, pos_item_indices, neg_item_indices = batch[0], batch[1], batch[2]
    return user_indices, pos_item_indices, neg_item_indices


class LightGCN(MessagePassing):
    def __init__(self, num_users, num_items, embedding_dim=64, K=3, add_self_loops=False, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.num_users, self.num_items = num_users, num_items
        self.embedding_dim, self.K = embedding_dim, K
        self.add_self_loops = add_self_loops

        self.users_emb = nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.embedding_dim)
        self.items_emb = nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.embedding_dim)

        nn.init.normal_(self.users_emb.weight, std=0.1)
        nn.init.normal_(self.items_emb.weight, std=0.1)

    def forward(self, edge_index: SparseTensor):
        emb_0 = torch.cat([self.users_emb.weight, self.items_emb.weight])
        embs = [emb_0]
        emb_k = emb_0

        for i in range(self.K):
            emb_k = self.propagate(edge_index, x=emb_k)
            embs.append(emb_k)

        embs = torch.stack(embs, dim=1)
        emb_final = torch.mean(embs, dim=1)

        users_emb_final, items_emb_final = torch.split(
            emb_final, [self.num_users, self.num_items])
        return users_emb_final, self.users_emb.weight, items_emb_final, self.items_emb.weight

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        # computes \tilde{A} @ x
        return matmul(adj_t, x)


def bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final, pos_items_emb_0, neg_items_emb_final, neg_items_emb_0,
             lambda_val):
    reg_loss = lambda_val * (users_emb_0.norm(2).pow(2) +
                             pos_items_emb_0.norm(2).pow(2) +
                             neg_items_emb_0.norm(2).pow(2))

    pos_scores = torch.mul(users_emb_final, pos_items_emb_final)
    pos_scores = torch.sum(pos_scores, dim=-1)
    neg_scores = torch.mul(users_emb_final, neg_items_emb_final)
    neg_scores = torch.sum(neg_scores, dim=-1)

    loss = -torch.mean(torch.nn.functional.softplus(pos_scores - neg_scores)) + reg_loss

    return loss


def get_positive_items(edge_index):
    user_pos_items = {}
    for i in range(edge_index.shape[1]):
        user = edge_index[0][i].item()
        item = edge_index[1][i].item()
        if user not in user_pos_items:
            user_pos_items[user] = []
        user_pos_items[user].append(item)
    return user_pos_items


def ndcgr(groundTruth, r, k):
    assert len(r) == len(groundTruth)

    test_matrix = torch.zeros((len(r), k))

    for i, items in enumerate(groundTruth):
        length = min(len(items), k)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = torch.sum(max_r * 1. / torch.log2(torch.arange(2, k + 2)), axis=1)
    dcg = r * (1. / torch.log2(torch.arange(2, k + 2)))
    dcg = torch.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[torch.isnan(ndcg)] = 0.
    return torch.mean(ndcg).item()


def hit(groundTruth, r, k):
    assert len(r) == len(groundTruth)

    hits = []
    for i, gt_items in enumerate(groundTruth):
        hits.append(any(item in gt_items for item in r[i][:k]))
    return torch.mean(torch.tensor(hits).float()).item()


def get_metrics(model, edge_index, exclude_edge_indices, k):
    user_embedding = model.users_emb.weight
    item_embedding = model.items_emb.weight

    rating = torch.matmul(user_embedding, item_embedding.T)

    for exclude_edge_index in exclude_edge_indices:
        user_pos_items = get_positive_items(exclude_edge_index)
        exclude_users = []
        exclude_items = []
        for user, items in user_pos_items.items():
            exclude_users.extend([user] * len(items))
            exclude_items.extend(items)

        rating[exclude_users, exclude_items] = -(1 << 10)

    _, top_K_items = torch.topk(rating, k=k)

    users = edge_index[0].unique()

    test_user_pos_items = get_positive_items(edge_index)

    test_user_pos_items_list = [
        test_user_pos_items[user.item()] for user in users]

    r = []
    for user in users:
        ground_truth_items = test_user_pos_items[user.item()]
        label = list(map(lambda x: x in ground_truth_items, top_K_items[user]))
        r.append(label)
    r = torch.Tensor(np.array(r).astype('float'))

    ndcg = ndcgr(test_user_pos_items_list, r, k)
    hr = hit(test_user_pos_items_list, r, k)

    return ndcg, hr


def evaluation(model, edge_index, sparse_edge_index, exclude_edge_indices, k, lambda_val):
    # get embeddings
    users_emb_final, users_emb_0, items_emb_final, items_emb_0 = model.forward(
        sparse_edge_index)
    edges = structured_negative_sampling(
        edge_index, contains_neg_self_loops=False)
    user_indices, pos_item_indices, neg_item_indices = edges[0], edges[1], edges[2]
    users_emb_final, users_emb_0 = users_emb_final[user_indices], users_emb_0[user_indices]
    pos_items_emb_final, pos_items_emb_0 = items_emb_final[
                                               pos_item_indices], items_emb_0[pos_item_indices]
    neg_items_emb_final, neg_items_emb_0 = items_emb_final[
                                               neg_item_indices], items_emb_0[neg_item_indices]

    loss = bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final, pos_items_emb_0,
                    neg_items_emb_final, neg_items_emb_0, lambda_val).item()

    ndcg, hr = get_metrics(
        model, edge_index, exclude_edge_indices, k)

    return loss, ndcg, hr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml-1m', help='dataset name')
    parser.add_argument('--learn', type=str, default='retrain', help='type of learning and unlearning')
    parser.add_argument('--deltype', type=str, default='random', help='unlearn data selection')
    parser.add_argument('--epoch', type=int, default=100, help='number of epochs')
    parser.add_argument('--worker', type=int, default=8, help='number of CPU workers')
    parser.add_argument('--verbose', type=int, default=2, help='verbose type')
    parser.add_argument('--group', type=int, default=10, help='number of groups')
    parser.add_argument('--delper', type=int, default=5, help='deleted user proportion')

    args = parser.parse_args()

    assert args.dataset in ['ml-1m', 'adm', 'book']
    dataset = args.dataset

    del_type = args.deltype
    method = args.learn
    groups = args.group
    del_per = args.delper

    rating_path = f'data/{dataset}/ratings.csv'
    train_path = f'data/{dataset}/train.csv'
    test_path = f'data/{dataset}/test.csv'

    user_mapping = load_data(rating_path, index_col='user')
    item_mapping = load_data(rating_path, index_col='item')

    num_users, num_movies = len(user_mapping), len(item_mapping)

    if args.learn == 'retrain':
        train_edge, test_edge, active_edge, inactive_edge = readRating_full_lightgcn(train_path, test_path, user_mapping, item_mapping, del_type, del_per)
    else:
        train_edge_groups, test_edge_groups, active_edge_groups, inactive_edge_groups, ensemble_train_edge, ensemble_test_edge = readRating_group_lightgcn(train_path, test_path, user_mapping, item_mapping, del_type, del_per,
                                  learn_type=method, num_groups=groups, dataset=dataset)
    

    ndcgs = [0] * groups
    hrs = [0] * groups
    active_ndcgs = [0] * groups
    inactive_ndcgs = [0] * groups
    times = [0] * groups

    if args.learn == 'retrain':
        epoch = args.epoch
        batch = 1024
        lr = 1e-3
        per_eval = 200
        per_lr_decay = 200
        K = 20
        LAMBDA = 1e-6
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

        best_ndcg = 0
        best_hr = 0
        count_dec = 0

        model = LightGCN(num_users, num_movies)
        model = model.to(device)
        model.train()

        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

        # edge_index = edge_index.to(device)
        train_edge_index = train_edge.to(device)
        train_sparse_edge_index = train_edge.to(device)

        test_edge_index = test_edge.to(device)
        test_sparse_edge_index = test_edge.to(device)

        active_edge_index = active_edge.to(device)
        active_sparse_edge_index = active_edge.to(device)

        inactive_edge_index = inactive_edge.to(device)
        inactive_sparse_edge_index = inactive_edge.to(device)

        train_losses = []
        val_losses = []

        start_time = time.time()
        total_time = 0

        for iter in range(epoch):
            # forward propagation
            users_emb_final, users_emb_0, items_emb_final, items_emb_0 = model.forward(
                train_sparse_edge_index)

            # mini batching
            user_indices, pos_item_indices, neg_item_indices = sample_batch(
                batch, train_edge_index)
            user_indices, pos_item_indices, neg_item_indices = user_indices.to(
                device), pos_item_indices.to(device), neg_item_indices.to(device)
            users_emb_final, users_emb_0 = users_emb_final[user_indices], users_emb_0[user_indices]
            pos_items_emb_final, pos_items_emb_0 = items_emb_final[
                                                    pos_item_indices], items_emb_0[pos_item_indices]
            neg_items_emb_final, neg_items_emb_0 = items_emb_final[
                                                    neg_item_indices], items_emb_0[neg_item_indices]

            # loss computation
            train_loss = bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final,
                                pos_items_emb_0, neg_items_emb_final, neg_items_emb_0, LAMBDA)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()


            if iter % per_eval == 0:
                model.eval()
                val_loss, ndcg, hr = evaluation(
                    model, test_edge_index, test_sparse_edge_index, [train_edge_index], K, LAMBDA)
                print(
                    f"[Iteration {iter}/{epoch}] train_loss: {round(train_loss.item(), 5)}, val_loss: {round(val_loss, 5)}, ndcg@{K}: {round(ndcg, 5)}, hr@{K}: {round(hr, 5)}")
                train_losses.append(train_loss.item())
                val_losses.append(val_loss)
                model.train()
                if ndcg > best_ndcg:
                    count_dec = 0
                    best_ndcg = ndcg
                    best_hr = hr
                    # if len(save_dir) > 0:
                    #     torch.save(model.state_dict(), save_dir + '/model' + '.pth')
                    #     torch.save(model.user_mat.weight.detach().cpu().numpy(), save_dir + '/user_mat' + '.npy')
                else:
                    count_dec += 1

                if count_dec > 5:
                    break

            if iter % per_lr_decay == 0 and iter != 0:
                scheduler.step()

        t_time = time.time() - start_time
        ndcgs[i] = best_ndcg
        hrs[i] = best_hr
        times[i] = t_time
        model.eval()
        if active_nums[i] > 0:
            val_loss, a_ndcg, a_hr = evaluation(
                model, active_edge_index, active_sparse_edge_index, [train_edge_index], K, LAMBDA)
        val_loss, i_ndcg, i_hr = evaluation(
            model, inactive_edge_index, inactive_sparse_edge_index, [train_edge_index], K, LAMBDA)
        print(f'Group {i}/{groups} Finish!')
        print("-------best---------")
        print(
            f"[best_ndcg@{K}: {round(best_ndcg, 5)}")
        if active_nums[i] > 0:
            print(
                f"[active_ndcg@{K}: {round(a_ndcg, 5)}")
        print(
            f"[inactive_ndcg@{K}: {round(i_ndcg, 5)}")
        if active_nums[i] > 0:
            active_ndcgs[i] = a_ndcg
        inactive_ndcgs[i] = i_ndcg
        print(np.mean(ndcgs))
        print(np.mean(hrs))
        active_ndcg = 0
        inactive_ndcg = 0
        for i in range(groups):
            active_ndcg += (active_nums[i] * active_ndcgs[i])
            inactive_ndcg += (inactive_nums[i] * inactive_ndcgs[i])

        active_num = np.sum(active_nums)
        active_ndcg = active_ndcg / active_num
        inactive_num = np.sum(inactive_nums)
        inactive_ndcg = inactive_ndcg / inactive_num
        fairness = active_ndcg - inactive_ndcg
        print(fairness)
        print(np.var(ndcgs))
        print(np.mean(times))
    else:
        for i in range(groups):
            epoch = args.epoch
            batch = 1024
            lr = 1e-3
            per_eval = 200
            per_lr_decay = 200
            K = 20
            LAMBDA = 1e-6
            device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

            best_ndcg = 0
            best_hr = 0
            count_dec = 0

            model = LightGCN(num_users, num_movies)
            model = model.to(device)
            model.train()

            optimizer = optim.Adam(model.parameters(), lr=lr)
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

            # edge_index = edge_index.to(device)
            train_edge_index = train_edge_groups[i].to(device)
            train_sparse_edge_index = train_edge_groups[i].to(device)

            test_edge_index = test_edge_groups[i].to(device)
            test_sparse_edge_index = test_edge_groups[i].to(device)

            active_edge_index = active_test_datasets[i].to(device)
            active_sparse_edge_index = active_test_datasets[i].to(device)

            inactive_edge_index = inactive_test_datasets[i].to(device)
            inactive_sparse_edge_index = inactive_test_datasets[i].to(device)

            train_losses = []
            val_losses = []

            start_time = time.time()
            total_time = 0

            for iter in range(epoch):
                # forward propagation
                users_emb_final, users_emb_0, items_emb_final, items_emb_0 = model.forward(
                    train_sparse_edge_index)

                # mini batching
                user_indices, pos_item_indices, neg_item_indices = sample_batch(
                    batch, train_edge_index)
                user_indices, pos_item_indices, neg_item_indices = user_indices.to(
                    device), pos_item_indices.to(device), neg_item_indices.to(device)
                users_emb_final, users_emb_0 = users_emb_final[user_indices], users_emb_0[user_indices]
                pos_items_emb_final, pos_items_emb_0 = items_emb_final[
                                                        pos_item_indices], items_emb_0[pos_item_indices]
                neg_items_emb_final, neg_items_emb_0 = items_emb_final[
                                                        neg_item_indices], items_emb_0[neg_item_indices]

                # loss computation
                train_loss = bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final,
                                    pos_items_emb_0, neg_items_emb_final, neg_items_emb_0, LAMBDA)

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()


                if iter % per_eval == 0:
                    model.eval()
                    val_loss, ndcg, hr = evaluation(
                        model, test_edge_index, test_sparse_edge_index, [train_edge_index], K, LAMBDA)
                    print(
                        f"[Iteration {iter}/{epoch}] train_loss: {round(train_loss.item(), 5)}, val_loss: {round(val_loss, 5)}, ndcg@{K}: {round(ndcg, 5)}, hr@{K}: {round(hr, 5)}")
                    train_losses.append(train_loss.item())
                    val_losses.append(val_loss)
                    model.train()
                    if ndcg > best_ndcg:
                        count_dec = 0
                        best_ndcg = ndcg
                        best_hr = hr
                        # if len(save_dir) > 0:
                        #     torch.save(model.state_dict(), save_dir + '/model' + '.pth')
                        #     torch.save(model.user_mat.weight.detach().cpu().numpy(), save_dir + '/user_mat' + '.npy')
                    else:
                        count_dec += 1

                    if count_dec > 5:
                        break

                if iter % per_lr_decay == 0 and iter != 0:
                    scheduler.step()

            t_time = time.time() - start_time
            ndcgs[i] = best_ndcg
            hrs[i] = best_hr
            times[i] = t_time
            model.eval()
            if active_nums[i] > 0:
                val_loss, a_ndcg, a_hr = evaluation(
                    model, active_edge_index, active_sparse_edge_index, [train_edge_index], K, LAMBDA)
            val_loss, i_ndcg, i_hr = evaluation(
                model, inactive_edge_index, inactive_sparse_edge_index, [train_edge_index], K, LAMBDA)
            print(f'Group {i}/{groups} Finish!')
            print("-------best---------")
            print(
                f"[best_ndcg@{K}: {round(best_ndcg, 5)}")
            if active_nums[i] > 0:
                print(
                    f"[active_ndcg@{K}: {round(a_ndcg, 5)}")
            print(
                f"[inactive_ndcg@{K}: {round(i_ndcg, 5)}")
            if active_nums[i] > 0:
                active_ndcgs[i] = a_ndcg
            inactive_ndcgs[i] = i_ndcg
        print(np.mean(ndcgs))
        print(np.mean(hrs))
        active_ndcg = 0
        inactive_ndcg = 0
        for i in range(groups):
            active_ndcg += (active_nums[i] * active_ndcgs[i])
            inactive_ndcg += (inactive_nums[i] * inactive_ndcgs[i])

        active_num = np.sum(active_nums)
        active_ndcg = active_ndcg / active_num
        inactive_num = np.sum(inactive_nums)
        inactive_ndcg = inactive_ndcg / inactive_num
        fairness = active_ndcg - inactive_ndcg
        print(fairness)
        print(np.var(ndcgs))
        print(np.mean(times))
