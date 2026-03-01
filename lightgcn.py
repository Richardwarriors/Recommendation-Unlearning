import argparse
import time
import random
import numpy as np
import pandas as pd
import torch
import os
from torch import nn, optim
from torch_sparse import SparseTensor, matmul

import ot  # POT

# -----------------------------
# Utils: mapping / loading
# -----------------------------
def load_mapping(root, index_col):
    df = pd.read_csv(root)
    unique_ids = df[index_col].unique()
    mapping = {raw_id: i for i, raw_id in enumerate(unique_ids)}
    return mapping


def delete_interactions(ratings, del_type, del_per, min_inter_per_user=2, seed=42):
    rng = np.random.default_rng(seed)
    total_interactions = len(ratings)
    num_delete = int(total_interactions * del_per / 100)

    user_counts = ratings["user"].value_counts()

    if del_type == "random":
        target_users = user_counts.index.tolist()
    elif del_type == "core":
        num_active = int(len(user_counts) * 0.05)
        target_users = user_counts.sort_values(ascending=False).index[:num_active]
    elif del_type == "edge":
        num_active = int(len(user_counts) * 0.05)
        target_users = user_counts.sort_values(ascending=False).index[num_active:]
    else:
        raise ValueError("Unknown del_type")

    deletable_idx = []
    for user in target_users:
        user_idx = ratings[ratings["user"] == user].index.tolist()
        max_delete = len(user_idx) - min_inter_per_user
        if max_delete > 0:
            deletable_idx.extend(user_idx[:max_delete])

    num_delete = min(num_delete, len(deletable_idx))
    if num_delete <= 0:
        return ratings.reset_index(drop=True)

    delete_idx = rng.choice(deletable_idx, num_delete, replace=False)
    return ratings.drop(index=delete_idx).reset_index(drop=True)


def load_edge(df, src_mapping, dst_mapping, rating_threshold=1):
    if "rating" in df.columns:
        df = df[df["rating"] >= rating_threshold]

    src = df["user"].map(src_mapping).astype(np.int64).values
    dst = df["item"].map(dst_mapping).astype(np.int64).values
    edge_np = np.stack((src, dst), axis=0)  # [2, E]
    return torch.from_numpy(edge_np).long()


def readRating_full_lightgcn(train_dir, test_dir, user_mapping, item_mapping, del_type="random", del_per=0):
    train_ratings = pd.read_csv(train_dir)
    train_ratings["rating"] = 1
    test_ratings = pd.read_csv(test_dir)
    test_ratings["rating"] = 1

    if del_per > 0:
        train_ratings = delete_interactions(train_ratings, del_type, del_per)

    train_edge = load_edge(train_ratings, user_mapping, item_mapping)
    test_edge = load_edge(test_ratings, user_mapping, item_mapping)
    return train_edge, test_edge

# =========================================================
# (RecEraser) Balanced K-means InBP (your version)
# =========================================================
def kmeans_InBP(train_ratings, test_ratings, dataset, num_groups, model_type, max_iters=10):
    """
    Interaction-based Balanced Partition (RecEraser InBP)
    Returns:
        train_rating_groups: list[pd.DataFrame]  (ONLY assigned interactions)
        test_rating_groups:  list[pd.DataFrame]  (users appearing in this shard's train interactions)
    """
    start_time = time.time()
    train_ratings = train_ratings.reset_index(drop=True)

    user_emb = np.load(f"results/user_emb/{dataset}_{model_type}_user_emb.npy", allow_pickle=True).item()
    item_emb = np.load(f"results/item_emb/{dataset}_{model_type}_item_emb.npy", allow_pickle=True).item()

    data = [(int(row["user"]), int(row["item"])) for _, row in train_ratings.iterrows()]
    n = len(data)
    k = int(num_groups)
    max_data = int(np.ceil(1.2 * n / k))

    centroids = random.sample(data, k)
    centroembs = []
    for u, i in centroids:
        pu = user_emb[np.int64(u)][0]
        qi = item_emb[np.int64(i)][0]
        centroembs.append((pu, qi))

    # keep shard_indices from last iter
    shard_indices = [[] for _ in range(k)]

    for _ in range(max_iters):
        C = [{} for _ in range(k)]
        C_num = [0] * k
        shard_indices = [[] for _ in range(k)]
        Scores = {}

        for idx, (u, i) in enumerate(data):
            pu = user_emb[np.int64(u)][0]
            qi = item_emb[np.int64(i)][0]

            for j in range(k):
                cu, ci = centroembs[j]
                dist_u = np.sum((pu - cu) ** 2)
                dist_i = np.sum((qi - ci) ** 2)
                Scores[(idx, j)] = -(dist_u * dist_i)

        Scores = sorted(Scores.items(), key=lambda x: x[1], reverse=True)
        assigned = set()

        for (idx, shard_id), _ in Scores:
            if idx in assigned:
                continue
            if C_num[shard_id] >= max_data:
                continue

            u, i = data[idx]
            shard_indices[shard_id].append(idx)

            if u not in C[shard_id]:
                C[shard_id][u] = [i]
            else:
                C[shard_id][u].append(i)

            C_num[shard_id] += 1
            assigned.add(idx)

            if len(assigned) == n:
                break

        new_centroembs = []
        for shard_id in range(k):
            temp_u, temp_i = [], []
            for u in C[shard_id]:
                for i in C[shard_id][u]:
                    temp_u.append(user_emb[np.int64(u)][0])
                    temp_i.append(item_emb[np.int64(i)][0])

            if len(temp_u) == 0:
                new_centroembs.append(centroembs[shard_id])
            else:
                new_centroembs.append((np.mean(temp_u, axis=0), np.mean(temp_i, axis=0)))

        if all(
            np.allclose(new_centroembs[i][0], centroembs[i][0]) and np.allclose(new_centroembs[i][1], centroembs[i][1])
            for i in range(k)
        ):
            centroembs = new_centroembs
            break

        centroembs = new_centroembs

    train_rating_groups, test_rating_groups = [], []
    for shard_id in range(k):
        shard_idx = shard_indices[shard_id]
        train_group = train_ratings.iloc[shard_idx].reset_index(drop=True)
        shard_users = train_group["user"].unique()
        test_group = test_ratings[test_ratings["user"].isin(shard_users)].reset_index(drop=True)
        train_rating_groups.append(train_group)
        test_rating_groups.append(test_group)

    print(f"Grouping time: {time.time() - start_time:.2f}s")
    return train_rating_groups, test_rating_groups


def ot_assignment(trans: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    """
    STRICT balanced hard assignment from OT plan.
    trans: [n, k]
    Output label: [n] with counts differing by at most 1.
    """
    rng = np.random.default_rng(seed)
    n = trans.shape[0]

    q, r = divmod(n, k)
    targets = np.array([q + 1 if j < r else q for j in range(k)], dtype=int)

    label = -np.ones(n, dtype=np.int64)
    assigned = np.zeros(n, dtype=bool)
    shard_count = np.zeros(k, dtype=int)

    shard_order = np.arange(k)
    rng.shuffle(shard_order)

    # Greedy fill each shard by highest mass to that shard
    for j in shard_order:
        need = targets[j]
        if need <= 0:
            continue

        idx_sorted = np.argsort(-trans[:, j], kind="mergesort")

        cnt = 0
        for i in idx_sorted:
            if not assigned[i]:
                label[i] = j
                assigned[i] = True
                shard_count[j] += 1
                cnt += 1
                if cnt >= need:
                    break

    # Fill any leftover (rare)
    if not assigned.all():
        remaining_idx = np.where(~assigned)[0]
        cap = targets - shard_count

        ptrs = []
        for j in range(k):
            if cap[j] > 0:
                ptrs.extend([j] * cap[j])
        ptrs = np.array(ptrs, dtype=np.int64)

        if len(ptrs) < len(remaining_idx):
            extra = len(remaining_idx) - len(ptrs)
            ptrs = np.concatenate([ptrs, np.tile(np.arange(k), int(np.ceil(extra / k)))[:extra]])

        for i, j in zip(remaining_idx, ptrs[:len(remaining_idx)]):
            label[i] = j

    counts = np.bincount(label, minlength=k)
    assert counts.sum() == n
    assert counts.max() - counts.min() <= 1, f"Not strictly balanced! counts={counts}"
    return label


def kmeans_ot_InBP(
    train_ratings,
    test_ratings,
    dataset,
    num_groups,
    model_type,
    max_iters=20,
    reg=2e-1, #unlearn 0 和 5 5e-2; 10 1e-1 or 2e-1
    seed=42,
    verbose=True,
):
    """
    ✅ TRUE interaction-level UltraRE(OBC):
      - sample = interaction (u,i)
      - cost = ||p_u - c_u||^2 * ||q_i - c_i||^2  (multiplicative, NOT concat L2)
      - OT(Sinkhorn) => soft plan
      - STRICT balanced hard assignment via ot_assignment
      - train shard built by interaction indices (NOT user isin)
    """
    t0 = time.time()
    rng = np.random.default_rng(seed)

    # -------- 1) load pretrained embeddings (dict: id -> (1,d) array) --------
    user_emb = np.load(f"results/user_emb/{dataset}_neumf_user_emb.npy", allow_pickle=True).item()
    item_emb = np.load(f"results/item_emb/{dataset}_neumf_item_emb.npy", allow_pickle=True).item()

    # -------- 2) build interaction arrays aligned with train_ratings row order --------
    users = train_ratings["user"].to_numpy(dtype=np.int64)
    items = train_ratings["item"].to_numpy(dtype=np.int64)
    n = len(users)
    k = int(num_groups)

    # Pull dimensions
    du = user_emb[np.int64(users[0])][0].shape[0]
    di = item_emb[np.int64(items[0])][0].shape[0]

    # Vectorize: build Pu [n,du], Qi [n,di]
    # (loop over n is still O(n) but avoids O(n*k) nested loops; the heavy part is OT and cost matrix anyway)
    Pu = np.empty((n, du), dtype=np.float32)
    Qi = np.empty((n, di), dtype=np.float32)
    for idx in range(n):
        Pu[idx] = user_emb[np.int64(users[idx])][0]
        Qi[idx] = item_emb[np.int64(items[idx])][0]

    # Optional but strongly recommended: L2 normalize for stable OT scale
    Pu = Pu / (np.linalg.norm(Pu, axis=1, keepdims=True) + 1e-12)
    Qi = Qi / (np.linalg.norm(Qi, axis=1, keepdims=True) + 1e-12)

    # -------- 3) init interaction anchors: pick k interactions -> centroid_u, centroid_i --------
    init_idx = rng.choice(n, size=k, replace=False)
    Cu = Pu[init_idx].copy()  # [k,du]
    Ci = Qi[init_idx].copy()  # [k,di]

    # OT marginals (balanced shards)
    a = np.ones(n, dtype=np.float64) / n
    b = np.ones(k, dtype=np.float64) / k

    last_label = None

    # -------- 4) OBC loop --------
    for it in range(max_iters):
        # dist_u: [n,k] = ||Pu - Cu||^2
        # dist_i: [n,k] = ||Qi - Ci||^2
        # cost:   [n,k] = dist_u * dist_i  (multiplicative interaction distance)
        dist_u = ((Pu[:, None, :] - Cu[None, :, :]) ** 2).sum(axis=2).astype(np.float64)
        dist_i = ((Qi[:, None, :] - Ci[None, :, :]) ** 2).sum(axis=2).astype(np.float64)
        cost = dist_u * dist_i

        # Normalize cost for numerical stability (important!)
        cost = cost / (cost.max() + 1e-12)

        # Sinkhorn stabilized (avoid overflow/div0 you saw)
        # If this still warns, try reg=1e-1 or 2e-1
        trans = ot.bregman.sinkhorn_stabilized(
            a, b, cost, reg,
            numItermax=2000,
            stopThr=1e-9,
            verbose=False
        )  # [n,k]

        # STRICT balanced hard assignment
        label = ot_assignment(trans, k=k, seed=seed + it)
        #label = np.argmax(trans, axis=1)

        # Update centroids (mean of assigned interactions)
        new_Cu = Cu.copy()
        new_Ci = Ci.copy()
        for j in range(k):
            mask = (label == j)
            if np.any(mask):
                new_Cu[j] = Pu[mask].mean(axis=0)
                new_Ci[j] = Qi[mask].mean(axis=0)
            #else:
            #    print(f"Warning: shard {j} is empty at iteration {it}! Reinitializing centroid randomly.")
            #    # avoid empty cluster
            #    rand_id = rng.integers(0, n)
            #    new_Cu[j] = Pu[rand_id]
            #    new_Ci[j] = Qi[rand_id]

        # Convergence
        if last_label is not None and np.array_equal(label, last_label):
            Cu, Ci = new_Cu, new_Ci
            break
        if np.allclose(Cu, new_Cu) and np.allclose(Ci, new_Ci):
            Cu, Ci = new_Cu, new_Ci
            break

        Cu, Ci = new_Cu, new_Ci
        last_label = label

    # -------- 5) build shard datasets (CRITICAL: by interaction indices) --------
    train_rating_groups = []
    test_rating_groups = []

    for shard_id in range(k):
        shard_idx = np.where(label == shard_id)[0]

        #train shard = ONLY assigned interactions
        train_group = train_ratings.iloc[shard_idx].reset_index(drop=True)

        #test shard = users appearing in this shard's train interactions (common practice)
        shard_users = train_group["user"].unique()
        test_group = test_ratings[test_ratings["user"].isin(shard_users)].reset_index(drop=True)

        train_rating_groups.append(train_group)
        test_rating_groups.append(test_group)

    if verbose:
        counts = np.bincount(label, minlength=k)
        print("\n[UltraRE Interaction-level OBC] STRICT balanced shard sizes (train interactions):")
        for j in range(k):
            print(f"  Shard {j}: {counts[j]}")
        print(f"  Mean={counts.mean():.2f}, Std={counts.std():.2f}, Max-Min={counts.max()-counts.min()}")
        print(f"UltraRE OBC Grouping time: {time.time() - t0:.2f}s")

    return train_rating_groups, test_rating_groups

def readRating_group_lightgcn(train_dir, test_dir,
                              user_mapping, item_mapping,
                              del_type='random',
                              del_per=5,
                              learn_type='sisa',
                              num_groups=10,
                              dataset='ml-1m',
                              model_type='neumf'):

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
    # edge_index
    # ========================

    train_edge_groups = [load_edge(r, user_mapping, item_mapping)
                         for r in train_rating_groups]

    test_edge_groups = [load_edge(r, user_mapping, item_mapping)
                        for r in test_rating_groups]

    # ========================
    # ensemble edge - whole training set for receraser and ultrare, and testing for all ensemble unlearning types
    # ========================

    ensemble_train_edge = load_edge(ensemble_train_df, user_mapping, item_mapping)
    ensemble_test_edge = load_edge(ensemble_test_df, user_mapping, item_mapping)

    return train_edge_groups, test_edge_groups, ensemble_train_edge,ensemble_test_edge
# -----------------------------
# Build normalized adjacency
# -----------------------------
def build_norm_adj(edge_index, num_users, num_items, device):
    """
    edge_index: [2, E] user,item
    return: normalized SparseTensor adjacency of shape [(U+I),(U+I)]
    """
    users = edge_index[0]
    items = edge_index[1] + num_users  # shift items

    row = torch.cat([users, items], dim=0)
    col = torch.cat([items, users], dim=0)

    total_nodes = num_users + num_items

    adj = SparseTensor(row=row, col=col, sparse_sizes=(total_nodes, total_nodes))
    deg = adj.sum(dim=1).to(torch.float)

    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    row2, col2, _ = adj.coo()
    norm_val = deg_inv_sqrt[row2] * deg_inv_sqrt[col2]

    adj_norm = SparseTensor(row=row2, col=col2, value=norm_val, sparse_sizes=(total_nodes, total_nodes))
    return adj_norm.to(device)


# -----------------------------
# LightGCN Model
# -----------------------------
class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, K=3):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.K = K

        self.users_emb = nn.Embedding(num_users, embedding_dim)
        self.items_emb = nn.Embedding(num_items, embedding_dim)

        nn.init.normal_(self.users_emb.weight, std=0.1)
        nn.init.normal_(self.items_emb.weight, std=0.1)

    def forward(self, adj_sparse: SparseTensor):
        emb0 = torch.cat([self.users_emb.weight, self.items_emb.weight], dim=0)  # [U+I, d]
        embs = [emb0]

        curr = emb0
        for _ in range(self.K):
            curr = matmul(adj_sparse, curr)
            embs.append(curr)

        final = torch.mean(torch.stack(embs, dim=1), dim=1)  # [U+I, d]
        u_final, i_final = torch.split(final, [self.num_users, self.num_items], dim=0)
        return u_final, self.users_emb.weight, i_final, self.items_emb.weight


def bpr_loss(u_final, p_final, n_final, u0, p0, n0, decay):
    pos_scores = torch.sum(u_final * p_final, dim=1)
    neg_scores = torch.sum(u_final * n_final, dim=1)
    loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

    # L2 regularization on ego embeddings only (official)
    reg = 0.5 * (u0.norm(2).pow(2) + p0.norm(2).pow(2) + n0.norm(2).pow(2)) / float(u_final.size(0))
    return loss + decay * reg


# -----------------------------
# Build user->pos (official style sampler needs list)
# -----------------------------
def build_user_pos_lists(train_edge, num_users):
    """
    returns: list[list[int]] length num_users
    """
    user_pos = [[] for _ in range(num_users)]
    u = train_edge[0].cpu().numpy()
    i = train_edge[1].cpu().numpy()
    for uu, ii in zip(u, i):
        user_pos[int(uu)].append(int(ii))
    return user_pos


def build_user_pos_sets(train_edge, num_users):
    """
    returns: list[set] length num_users (for fast membership check)
    """
    user_pos = [set() for _ in range(num_users)]
    u = train_edge[0].cpu().numpy()
    i = train_edge[1].cpu().numpy()
    for uu, ii in zip(u, i):
        user_pos[int(uu)].add(int(ii))
    return user_pos


# -----------------------------
# Official-style sampler: generate (u,pos,neg) triples per epoch
# Supports neg_k = 1 or 4 (or any k)
# -----------------------------
def uniform_sample_epoch(num_users, num_items, train_size, user_pos_lists, user_pos_sets, neg_k=1, seed=None):
    """
    Mimic LightGCN UniformSample_original (python/C++): 
      - sample users uniformly: size=train_size
      - sample 1 positive for each user
      - sample neg_k negatives not in positives

    Return:
      users [S], pos [S], neg [S*neg_k] if neg_k>1 (flattened)
      and also expanded users/pos matched to negs for BPR.
    """
    rng = np.random.default_rng(seed)

    users = rng.integers(0, num_users, size=train_size, endpoint=False)
    pos = np.empty(train_size, dtype=np.int64)

    # pick one positive for each sampled user
    valid_mask = np.ones(train_size, dtype=bool)
    for idx, u in enumerate(users):
        plist = user_pos_lists[u]
        if len(plist) == 0:
            valid_mask[idx] = False
            continue
        pos[idx] = plist[rng.integers(0, len(plist))]

    users = users[valid_mask]
    pos = pos[valid_mask]

    S = users.shape[0]
    # expand for 1:k negatives
    users_exp = np.repeat(users, neg_k)
    pos_exp = np.repeat(pos, neg_k)

    neg = rng.integers(0, num_items, size=S * neg_k, endpoint=False)
    # rejection
    for t in range(S * neg_k):
        u = int(users_exp[t])
        while int(neg[t]) in user_pos_sets[u]:
            neg[t] = rng.integers(0, num_items, endpoint=False)

    return users_exp, pos_exp, neg


# -----------------------------
# Evaluation: full-ranking
# -----------------------------
@torch.no_grad()
def evaluation_full(model, test_edge, train_edge, adj_norm, num_items, k=10, batch_size=256):
    model.eval()
    u_final, _, i_final, _ = model(adj_norm)

    # test targets
    test_dict = {}
    for u, it in zip(test_edge[0].cpu().numpy(), test_edge[1].cpu().numpy()):
        test_dict.setdefault(int(u), set()).add(int(it))

    # train mask
    train_dict = {}
    for u, it in zip(train_edge[0].cpu().numpy(), train_edge[1].cpu().numpy()):
        train_dict.setdefault(int(u), set()).add(int(it))

    users = list(test_dict.keys())
    all_hr, all_ndcg = [], []

    for start in range(0, len(users), batch_size):
        u_ids = users[start:start + batch_size]
        u_batch = u_final[u_ids]  # [b,d]
        scores = torch.matmul(u_batch, i_final.t())  # [b, I]

        for row_i, u in enumerate(u_ids):
            if u in train_dict:
                scores[row_i, list(train_dict[u])] = -1e10

        _, topk = torch.topk(scores, k=k, dim=1)
        topk = topk.cpu().numpy()

        for row_i, u in enumerate(u_ids):
            targets = test_dict[u]
            hits = [1 if int(it) in targets else 0 for it in topk[row_i]]

            all_hr.append(1.0 if sum(hits) > 0 else 0.0)
            dcg = sum(h / np.log2(idx + 2) for idx, h in enumerate(hits))
            idcg = sum(1.0 / np.log2(idx + 2) for idx in range(min(len(targets), k)))
            all_ndcg.append(dcg / idcg if idcg > 0 else 0.0)

    return float(np.mean(all_hr)), float(np.mean(all_ndcg))


# -----------------------------
# Evaluation: sampled 1 pos + 99 neg (often used in papers)
# -----------------------------
@torch.no_grad()
def evaluation_sampled_1p99n(model, test_edge, train_edge, adj_norm, num_items, k=10, neg_num=99, seed=42):
    """
    For each user: candidate set = {true_test_item} + 99 negatives (not in train positives, not equal to test item).
    Compute HR/NDCG on this 100-item ranking.
    """
    rng = np.random.default_rng(seed)
    model.eval()
    u_final, _, i_final, _ = model(adj_norm)

    # build dicts
    test_items = {}
    for u, it in zip(test_edge[0].cpu().numpy(), test_edge[1].cpu().numpy()):
        # if multiple test items per user, we'll treat as a set (still ok)
        test_items.setdefault(int(u), set()).add(int(it))

    train_dict = {}
    for u, it in zip(train_edge[0].cpu().numpy(), train_edge[1].cpu().numpy()):
        train_dict.setdefault(int(u), set()).add(int(it))

    users = list(test_items.keys())
    all_hr, all_ndcg = [], []

    for u in users:
        uemb = u_final[u]  # [d]
        gt_set = test_items[u]

        # if multiple test items, pick one (leave-one-out should be 1)
        gt = next(iter(gt_set))

        banned = train_dict.get(u, set()).copy()
        banned.add(gt)

        negs = []
        while len(negs) < neg_num:
            x = int(rng.integers(0, num_items))
            if x in banned:
                continue
            negs.append(x)

        cand = [gt] + negs  # size 100
        cand_t = torch.tensor(cand, device=uemb.device, dtype=torch.long)

        scores = torch.matmul(i_final[cand_t], uemb)  # [100]
        rank = torch.argsort(scores, descending=True).tolist()
        topk = [cand[r] for r in rank[:k]]

        hits = [1 if it == gt else 0 for it in topk]
        all_hr.append(1.0 if sum(hits) > 0 else 0.0)

        dcg = sum(h / np.log2(idx + 2) for idx, h in enumerate(hits))
        idcg = 1.0  # only 1 ground truth
        all_ndcg.append(dcg / idcg)

    return float(np.mean(all_hr)), float(np.mean(all_ndcg))

def train_lightgcn(
    train_edge, test_edge,
    num_users, num_items,
    device,
    epochs=1000,
    embed_dim=64,
    K_layers=3,
    lr=1e-3,
    decay=1e-4,
    batch_size=1024,
    eval_every=50,
    neg_k=1,
    eval_mode="sampled",
    learn_type='retrain',
    dataset='ml-1m',
    seed=42,
):

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = LightGCN(num_users, num_items, embedding_dim=embed_dim, K=K_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_edge = train_edge.to(device)
    test_edge = test_edge.to(device)

    adj_norm = build_norm_adj(train_edge, num_users, num_items, device)

    user_pos_lists = build_user_pos_lists(train_edge.cpu(), num_users)
    user_pos_sets = build_user_pos_sets(train_edge.cpu(), num_users)

    train_size = train_edge.size(1)

    best_ndcg = 0.0
    t0 = time.time()

    for ep in range(1, epochs + 1):

        model.train()

        # 2️⃣ 采样
        u_np, p_np, n_np = uniform_sample_epoch(
            num_users,
            num_items,
            train_size,
            user_pos_lists,
            user_pos_sets,
            neg_k,
            seed=seed + ep,
        )

        u = torch.from_numpy(u_np).long().to(device)
        p = torch.from_numpy(p_np).long().to(device)
        n = torch.from_numpy(n_np).long().to(device)

        perm = torch.randperm(u.size(0), device=device)
        u, p, n = u[perm], p[perm], n[perm]

        epoch_loss = 0.0
        steps = 0

        for start in range(0, u.size(0), batch_size):

            idx = slice(start, start + batch_size)

            u_b = u[idx]
            p_b = p[idx]
            n_b = n[idx]

            optimizer.zero_grad()

            # ✅ 每个 batch forward 一次
            u_f, u0, i_f, i0 = model(adj_norm)

            loss = bpr_loss(
                u_f[u_b], i_f[p_b], i_f[n_b],
                u0[u_b], i0[p_b], i0[n_b],
                decay
            )

            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            steps += 1

        epoch_loss /= max(steps, 1)

        if ep % eval_every == 0:

            if eval_mode == "full":
                hr, ndcg = evaluation_full(
                    model, test_edge, train_edge, adj_norm, num_items
                )
            else:
                hr, ndcg = evaluation_sampled_1p99n(
                    model, test_edge, train_edge, adj_norm, num_items
                )

            print(f"Epoch {ep} | loss={epoch_loss:.4f} | HR={hr:.4f} | NDCG={ndcg:.4f}")
    # ==============================
    # Save final propagated embeddings
    # ==============================
    if learn_type == 'retrain':
        model.eval()
        with torch.no_grad():
            u_final, _, i_final, _ = model(adj_norm)

        u_np = u_final.cpu().numpy().astype(np.float32)
        i_np = i_final.cpu().numpy().astype(np.float32)

        user_dict = {}
        for uid in range(u_np.shape[0]):
            user_dict[np.int64(uid)] = u_np[uid:uid+1]

        item_dict = {}
        for iid in range(i_np.shape[0]):
            item_dict[np.int64(iid)] = i_np[iid:iid+1]

        np.save(f"./results/user_emb/{dataset}_lightgcn_user_emb.npy", user_dict)
        np.save(f"./results/item_emb/{dataset}_lightgcn_item_emb.npy", item_dict)

        print("Final user embeddings saved to user_embedding.npy")
        print("Final item embeddings saved to item_embedding.npy")

    return model

@torch.no_grad()
def SISA_SEnsemble(models,train_edge,test_edge,num_users,num_items,device,neg_num=99,k=10,seed=42):

    rng = np.random.default_rng(seed)

    # Build adjacency
    adj_norm = build_norm_adj(
        train_edge.to(device),
        num_users,
        num_items,
        device
    )

    # Precompute embeddings
    shard_user_embs = []
    shard_item_embs = []

    for m in models:
        m.eval()
        u_final, _, i_final, _ = m(adj_norm)
        shard_user_embs.append(u_final)
        shard_item_embs.append(i_final)

    # Build dicts
    test_dict = {}
    for u, it in zip(test_edge[0].cpu().numpy(),
                     test_edge[1].cpu().numpy()):
        test_dict.setdefault(int(u), set()).add(int(it))

    train_dict = {}
    for u, it in zip(train_edge[0].cpu().numpy(),
                     train_edge[1].cpu().numpy()):
        train_dict.setdefault(int(u), set()).add(int(it))

    all_hr = []
    all_ndcg = []

    for u in test_dict.keys():

        gt = next(iter(test_dict[u]))

        banned = train_dict.get(u, set()).copy()
        banned.add(gt)

        negs = []
        while len(negs) < neg_num:
            x = int(rng.integers(0, num_items))
            if x in banned:
                continue
            negs.append(x)

        cand = [gt] + negs
        cand_t = torch.tensor(cand, device=device)

        scores = 0

        for g in range(len(models)):
            user_emb = shard_user_embs[g][u]
            item_emb = shard_item_embs[g][cand_t]
            scores += torch.matmul(item_emb, user_emb)

        scores /= len(models)

        rank = torch.argsort(scores, descending=True).tolist()
        topk = [cand[r] for r in rank[:k]]

        hits = [1 if it == gt else 0 for it in topk]

        all_hr.append(1.0 if sum(hits) > 0 else 0.0)

        dcg = sum(h / np.log2(idx + 2)
                  for idx, h in enumerate(hits))

        all_ndcg.append(dcg)   # IDCG=1

    return float(np.mean(all_hr)), float(np.mean(all_ndcg))

@torch.no_grad()
def SISA_FEnsemble(models,train_edge,test_edge,num_users,num_items,device,k=10,batch_size=256):

    # Build full adjacency
    adj_norm = build_norm_adj(
        train_edge.to(device),
        num_users,
        num_items,
        device
    )

    # Precompute embeddings for each shard
    shard_user_embs = []
    shard_item_embs = []

    for m in models:
        m.eval()
        u_final, _, i_final, _ = m(adj_norm)
        shard_user_embs.append(u_final)
        shard_item_embs.append(i_final)

    # Build dicts
    test_dict = {}
    for u, it in zip(test_edge[0].cpu().numpy(),
                     test_edge[1].cpu().numpy()):
        test_dict.setdefault(int(u), set()).add(int(it))

    train_dict = {}
    for u, it in zip(train_edge[0].cpu().numpy(),
                     train_edge[1].cpu().numpy()):
        train_dict.setdefault(int(u), set()).add(int(it))

    users = list(test_dict.keys())

    all_hr = []
    all_ndcg = []

    for start in range(0, len(users), batch_size):

        u_ids = users[start:start + batch_size]

        # accumulate scores from shards
        scores = None

        for g in range(len(models)):
            u_batch = shard_user_embs[g][u_ids]
            s = torch.matmul(u_batch, shard_item_embs[g].t())

            if scores is None:
                scores = s
            else:
                scores += s

        scores /= len(models)

        # mask training items
        for row_i, u in enumerate(u_ids):
            if u in train_dict:
                scores[row_i, list(train_dict[u])] = -1e10

        _, topk = torch.topk(scores, k=k, dim=1)
        topk = topk.cpu().numpy()

        for row_i, u in enumerate(u_ids):
            targets = test_dict[u]
            hits = [1 if int(it) in targets else 0 for it in topk[row_i]]

            all_hr.append(1.0 if sum(hits) > 0 else 0.0)

            dcg = sum(h / np.log2(idx + 2)
                      for idx, h in enumerate(hits))

            idcg = sum(1.0 / np.log2(idx + 2)
                       for idx in range(min(len(targets), k)))

            all_ndcg.append(dcg / idcg if idcg > 0 else 0.0)

    return float(np.mean(all_hr)), float(np.mean(all_ndcg))

class RecEraserAggregator(nn.Module):
    def __init__(self, emb_dim, num_shards, att_dim=64):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_shards = num_shards

        # shard transformation
        self.trans_W = nn.Parameter(torch.randn(num_shards, emb_dim, emb_dim) * 0.01)
        self.trans_b = nn.Parameter(torch.zeros(num_shards, emb_dim))

        # user attention
        self.WA = nn.Linear(emb_dim, att_dim)
        self.HA = nn.Linear(att_dim, 1)

        # item attention
        self.WB = nn.Linear(emb_dim, att_dim)
        self.HB = nn.Linear(att_dim, 1)

    def _attn(self, embs, is_user=True):
        # embs: [B, S, D]
        if is_user:
            x = torch.relu(self.WA(embs))
            score = self.HA(x)
        else:
            x = torch.relu(self.WB(embs))
            score = self.HB(x)

        w = torch.softmax(score, dim=1)  # [B,S,1]
        agg = torch.sum(w * embs, dim=1) # [B,D]
        return agg

    def forward(self, user_embs, item_embs):
        # transform
        u_trans = torch.einsum('bsd,sdh->bsh', user_embs, self.trans_W) + self.trans_b
        i_trans = torch.einsum('bsd,sdh->bsh', item_embs, self.trans_W) + self.trans_b

        u_agg = self._attn(u_trans, is_user=True)
        i_agg = self._attn(i_trans, is_user=False)

        return u_agg, i_agg

def train_receraser_aggregator_lightgcn(
    train_edge,
    models,
    aggregator,
    device,
    pos_dict,
    num_users,
    num_items,
    epochs_agg=5,
    batch_size=1048,
    num_neg=4,
    lr=1e-3,
):

    aggregator.train()
    opt = torch.optim.Adam(aggregator.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    # ===== build adjacency =====
    adj_norm = build_norm_adj(
        train_edge.to(device),
        num_users,
        num_items,
        device
    )

    # ===== precompute shard embeddings =====
    shard_user_embs = []
    shard_item_embs = []

    for m in models:
        m.eval()
        with torch.no_grad():
            u_final, _, i_final, _ = m(adj_norm)
        shard_user_embs.append(u_final)
        shard_item_embs.append(i_final)

    users = train_edge[0].cpu().numpy()
    items = train_edge[1].cpu().numpy()
    N = len(users)

    for ep in range(epochs_agg):

        idx = np.random.permutation(N)
        total_loss = 0.0
        steps = 0

        for st in range(0, N, batch_size):
            ed = min(st + batch_size, N)
            batch_idx = idx[st:ed]

            u_pos = users[batch_idx]
            i_pos = items[batch_idx]

            u_all = []
            i_all = []
            y_all = []

            for u, ip in zip(u_pos, i_pos):

                u_all.append(u)
                i_all.append(ip)
                y_all.append(1.0)

                neg_pool = list(set(range(num_items)) - set(pos_dict[int(u)]))
                if len(neg_pool) == 0:
                    continue

                negs = random.choices(neg_pool, k=num_neg)

                for ineg in negs:
                    u_all.append(u)
                    i_all.append(ineg)
                    y_all.append(0.0)

            u_all = torch.tensor(u_all, dtype=torch.long, device=device)
            i_all = torch.tensor(i_all, dtype=torch.long, device=device)
            y_all = torch.tensor(y_all, dtype=torch.float32, device=device)

            # ===== collect shard embeddings =====
            with torch.no_grad():
                user_embs = []
                item_embs = []

                for g in range(len(models)):
                    u_emb = shard_user_embs[g][u_all]
                    i_emb = shard_item_embs[g][i_all]

                    user_embs.append(u_emb.unsqueeze(1))
                    item_embs.append(i_emb.unsqueeze(1))

                user_embs = torch.cat(user_embs, dim=1)
                item_embs = torch.cat(item_embs, dim=1)

            # ===== aggregator forward =====
            u_agg, i_agg = aggregator(user_embs, item_embs)
            logits = torch.sum(u_agg * i_agg, dim=1)

            loss = bce(logits, y_all)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            steps += 1

        print(f"[Aggregator Epoch {ep+1}] loss={total_loss/max(steps,1):.6f}")

    aggregator.eval()
    return aggregator

@torch.no_grad()
def RecEraser_SEnsemble(
    models,
    aggregator,
    train_edge,
    test_edge,
    device,
    pos_dict,
    num_users,
    num_items,
    top_k=10,
):

    # ===== build adjacency =====
    adj_norm = build_norm_adj(
        train_edge.to(device),
        num_users,
        num_items,
        device
    )

    # ===== precompute shard embeddings =====
    shard_user_embs = []
    shard_item_embs = []

    for m in models:
        m.eval()
        u_final, _, i_final, _ = m(adj_norm)
        shard_user_embs.append(u_final)
        shard_item_embs.append(i_final)

    test_dict = {}
    for u, it in zip(test_edge[0].cpu().numpy(),
                     test_edge[1].cpu().numpy()):
        test_dict.setdefault(int(u), set()).add(int(it))

    HR, NDCG = [], []

    for u in test_dict.keys():

        gt = next(iter(test_dict[u]))

        neg_pool = list(set(range(num_items)) - set(pos_dict[u]))
        neg_items = random.sample(neg_pool, 99)

        candidates = [gt] + neg_items

        new_user = torch.tensor([u] * len(candidates), device=device)
        new_item = torch.tensor(candidates, device=device)

        user_embs = []
        item_embs = []

        for g in range(len(models)):
            u_emb = shard_user_embs[g][new_user]
            i_emb = shard_item_embs[g][new_item]

            user_embs.append(u_emb.unsqueeze(1))
            item_embs.append(i_emb.unsqueeze(1))

        user_embs = torch.cat(user_embs, dim=1)
        item_embs = torch.cat(item_embs, dim=1)

        u_agg, i_agg = aggregator(user_embs, item_embs)
        scores = torch.sum(u_agg * i_agg, dim=1)

        _, indices = torch.topk(scores, top_k)
        recommends = torch.take(new_item, indices).cpu().numpy().tolist()

        hit_val = 1.0 if gt in recommends else 0.0
        HR.append(hit_val)

        if gt in recommends:
            rank = recommends.index(gt)
            NDCG.append(1.0 / np.log2(rank + 2))
        else:
            NDCG.append(0.0)

    return float(np.mean(NDCG)), float(np.mean(HR))


@torch.no_grad()
def RecEraser_FEnsemble(
    models,
    aggregator,
    train_edge,
    test_edge,
    device,
    num_users,
    num_items,
    top_k=10,
    batch_size=256,
):

    adj_norm = build_norm_adj(
        train_edge.to(device),
        num_users,
        num_items,
        device
    )

    # ===== precompute shard embeddings =====
    shard_user_embs = []
    shard_item_embs = []

    for m in models:
        m.eval()
        u_final, _, i_final, _ = m(adj_norm)
        shard_user_embs.append(u_final)
        shard_item_embs.append(i_final)

    # ===== 聚合所有 item embedding 一次 =====
    item_embs = []
    for g in range(len(models)):
        item_embs.append(shard_item_embs[g].unsqueeze(1))

    item_embs = torch.cat(item_embs, dim=1)  # [I,S,D]

    # dummy user (不会用)
    dummy_user = torch.zeros_like(item_embs)

    _, i_agg_all = aggregator(dummy_user, item_embs)  # [I,D]

    # ===== build dict =====
    test_dict = {}
    for u, it in zip(test_edge[0].cpu().numpy(),
                     test_edge[1].cpu().numpy()):
        test_dict.setdefault(int(u), set()).add(int(it))

    train_dict = {}
    for u, it in zip(train_edge[0].cpu().numpy(),
                     train_edge[1].cpu().numpy()):
        train_dict.setdefault(int(u), set()).add(int(it))

    users = list(test_dict.keys())

    HR = []
    NDCG = []

    for start in range(0, len(users), batch_size):

        u_ids = users[start:start + batch_size]

        # ===== 聚合 user =====
        user_embs = []
        for g in range(len(models)):
            user_embs.append(
                shard_user_embs[g][u_ids].unsqueeze(1)
            )

        user_embs = torch.cat(user_embs, dim=1)  # [B,S,D]

        u_agg, _ = aggregator(user_embs, torch.zeros_like(user_embs))
        # 只取 user 部分

        scores = torch.matmul(u_agg, i_agg_all.t())

        # mask train
        for row_i, u in enumerate(u_ids):
            if u in train_dict:
                scores[row_i, list(train_dict[u])] = -1e10

        _, topk = torch.topk(scores, k=top_k, dim=1)
        topk = topk.cpu().numpy()

        for row_i, u in enumerate(u_ids):

            targets = test_dict[u]
            hits = [1 if int(it) in targets else 0 for it in topk[row_i]]

            HR.append(1.0 if sum(hits) > 0 else 0.0)

            dcg = sum(h / np.log2(idx + 2)
                      for idx, h in enumerate(hits))

            idcg = sum(1.0 / np.log2(idx + 2)
                       for idx in range(min(len(targets), top_k)))

            NDCG.append(dcg / idcg if idcg > 0 else 0.0)

    return float(np.mean(NDCG)), float(np.mean(HR))

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml-1m", choices=["ml-1m", "adm", "book"])
    parser.add_argument("--deltype", type=str, default="random", choices=["random", "core", "edge"])
    parser.add_argument('--learn', type=str, default='retrain', help='type of learning and unlearning')
    parser.add_argument("--group", type=int, default=10, help="number of groups for group-based unlearning (only for sisa/receraser/ultrare)")
    parser.add_argument("--delper", type=int, default=0, help="deletion percentage for retrain (0 means no deletion, 100 means all interactions deleted)")
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--embed", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--decay", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--eval_every", type=int, default=50)

    # NEW: 1:1 or 1:4 negative sampling
    parser.add_argument("--neg_k", type=int, default=4, help="negative per positive (1 means 1:1, 4 means 1:4)")

    # NEW: eval protocol
    parser.add_argument("--eval_mode", type=str, default="full", choices=["sampled", "full"],
                        help="sampled = 1 positive + 99 negatives; full = full-ranking")

    args = parser.parse_args()

    dataset = args.dataset
    rating_path = f"data/{dataset}/ratings.csv"
    train_path = f"data/{dataset}/train.csv"
    test_path = f"data/{dataset}/test.csv"

    user_mapping = load_mapping(rating_path, "user")
    item_mapping = load_mapping(rating_path, "item")
    num_users, num_items = len(user_mapping), len(item_mapping)

    if args.learn == 'retrain':
        train_edge, test_edge = readRating_full_lightgcn(train_path, test_path,user_mapping, item_mapping, del_type=args.deltype, del_per=args.delper)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_lightgcn(train_edge=train_edge, test_edge=test_edge, num_users=num_users,num_items=num_items, device=device, epochs=args.epoch, embed_dim=args.embed,K_layers=args.layers,
            lr=args.lr, decay=args.decay, batch_size=args.batch,
            eval_every=args.eval_every, neg_k=args.neg_k,
            eval_mode=args.eval_mode, learn_type=args.learn, dataset= dataset, seed=42)
    elif args.learn == 'sisa':
        group = args.group
        shard_model_path = []

        train_edge_groups, test_edge_groups, ensemble_train_edge,ensemble_test_edge = readRating_group_lightgcn(train_path, test_path,user_mapping, item_mapping, del_type=args.deltype, 
                                                                                                                del_per=args.delper, learn_type=args.learn, num_groups=group, dataset=dataset, model_type='neumf')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for i in range(group):
            print(f"\n=== Training group {i+1}/{group} ===")
            model = train_lightgcn(train_edge=train_edge_groups[i], test_edge=test_edge_groups[i], num_users=num_users,num_items=num_items, device=device, epochs=args.epoch, embed_dim=args.embed,K_layers=args.layers,
                lr=args.lr, decay=args.decay, batch_size=args.batch,
                eval_every=args.eval_every, neg_k=args.neg_k,
                eval_mode=args.eval_mode, learn_type=args.learn, seed=42)
            
            save_dir = f'results/{args.learn}'
            os.makedirs(save_dir, exist_ok=True)   
            shard_model_path.append(f"{save_dir}/{dataset}_lightgcn_group{i + 1}.pth")
            torch.save(model.state_dict(), shard_model_path[-1])
            print(f"Group {i+1} model saved to {shard_model_path[-1]}")

        # ==========================
        # SISA ENSEMBLE TESTING
        # ==========================

        print("\nStart SISA ensemble testing...")

        models = []

        for path in shard_model_path:
            m = LightGCN(num_users,num_items,embedding_dim=args.embed,K=args.layers)

            state_dict = torch.load(path, map_location=device, weights_only=True)
            m.load_state_dict(state_dict)
            m.to(device)
            m.eval()
            models.append(m)
        
        if args.eval_mode == "full":
            hr, ndcg = SISA_FEnsemble(models,ensemble_train_edge,ensemble_test_edge,num_users,num_items,device,k=10)
        else:
            hr, ndcg = SISA_SEnsemble(models, ensemble_train_edge, ensemble_test_edge, num_users,num_items,device,neg_num=99,k=10)

        print("SISA HR@10:", hr)
        print("SISA NDCG@10:", ndcg)

    elif args.learn == 'receraser':
        group = args.group
        shard_model_path = []

        train_edge_groups, test_edge_groups, ensemble_train_edge,ensemble_test_edge = readRating_group_lightgcn(train_path, test_path,user_mapping, item_mapping, del_type=args.deltype, 
                                                                                                                del_per=args.delper, learn_type=args.learn, num_groups=group, dataset=dataset, model_type='neumf')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for i in range(group):
            print(f"\n=== Training group {i+1}/{group} ===")
            model = train_lightgcn(train_edge=train_edge_groups[i], test_edge=test_edge_groups[i], num_users=num_users,num_items=num_items, device=device, epochs=args.epoch, embed_dim=args.embed,K_layers=args.layers,
                lr=args.lr, decay=args.decay, batch_size=args.batch,
                eval_every=args.eval_every, neg_k=args.neg_k,
                eval_mode=args.eval_mode, learn_type=args.learn, seed=42)
            
            save_dir = f'results/{args.learn}'
            os.makedirs(save_dir, exist_ok=True)   
            shard_model_path.append(f"{save_dir}/{dataset}_lightgcn_group{i + 1}.pth")
            torch.save(model.state_dict(), shard_model_path[-1])
            print(f"Group {i+1} model saved to {shard_model_path[-1]}")
        
            # ===== load shard models =====
        models = []

        for path in shard_model_path:
            m = LightGCN(num_users, num_items, args.embed, args.layers)
            state_dict = torch.load(path, map_location=device, weights_only=True)
            m.load_state_dict(state_dict)
            m.to(device)
            m.eval()
            models.append(m)

        # ===== build pos_dict =====
        pos_dict = build_user_pos_sets(ensemble_train_edge, num_users)

        # ===== initialize aggregator =====
        aggregator = RecEraserAggregator(
            emb_dim=args.embed,
            num_shards=len(models),
            att_dim=64
        ).to(device)

        # ===== train aggregator =====
        aggregator = train_receraser_aggregator_lightgcn(train_edge=ensemble_train_edge,models=models,aggregator=aggregator,device=device,
            pos_dict=pos_dict,num_users=num_users,num_items=num_items,epochs_agg=5,batch_size=2048,
            num_neg=4,lr=1e-3)

        # ===== final test =====
        if args.eval_mode == "full":
            ndcg, hr = RecEraser_FEnsemble(models=models,aggregator=aggregator,
                train_edge=ensemble_train_edge,test_edge=ensemble_test_edge,
                device=device,num_users=num_users,num_items=num_items,top_k=10)
        else:
            ndcg, hr = RecEraser_SEnsemble(models=models,aggregator=aggregator,train_edge=ensemble_train_edge,test_edge=ensemble_test_edge,
            device=device,pos_dict=pos_dict,num_users=num_users,num_items=num_items,top_k=10)

        print("\n====== RecEraser Final Result ======")
        print("HR@10:", hr)
        print("NDCG@10:", ndcg)
    
    elif args.learn == 'ultrare':

        group = args.group
        shard_model_path = []

        train_edge_groups, test_edge_groups, ensemble_train_edge, ensemble_test_edge = \
            readRating_group_lightgcn(
                train_path,
                test_path,
                user_mapping,
                item_mapping,
                del_type=args.deltype,
                del_per=args.delper,
                learn_type=args.learn,
                num_groups=group,
                dataset=dataset,
                model_type='neumf'
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ===== Train shard models =====
        for i in range(group):
            print(f"\n=== Training group {i+1}/{group} ===")

            model = train_lightgcn(
                train_edge=train_edge_groups[i],
                test_edge=test_edge_groups[i],
                num_users=num_users,
                num_items=num_items,
                device=device,
                epochs=args.epoch,
                embed_dim=args.embed,
                K_layers=args.layers,
                lr=args.lr,
                decay=args.decay,
                batch_size=args.batch,
                eval_every=args.eval_every,
                neg_k=args.neg_k,
                eval_mode=args.eval_mode,
                learn_type=args.learn,
                seed=42
            )

            save_dir = f'results/{args.learn}'
            os.makedirs(save_dir, exist_ok=True)

            path = f"{save_dir}/{dataset}_lightgcn_group{i + 1}.pth"
            torch.save(model.state_dict(), path)
            shard_model_path.append(path)

            print(f"Group {i+1} model saved to {path}")

        # ===== Load shard models =====
        models = []

        for path in shard_model_path:
            m = LightGCN(num_users, num_items, args.embed, args.layers)
            state_dict = torch.load(path, map_location=device, weights_only=True)
            m.load_state_dict(state_dict)
            m.to(device)
            m.eval()
            models.append(m)

        # ===== Train Combiner (UltraRE Stage III) =====
        pos_dict = build_user_pos_sets(ensemble_train_edge, num_users)

        # ===== initialize aggregator =====
        aggregator = RecEraserAggregator(
            emb_dim=args.embed,
            num_shards=len(models),
            att_dim=64
        ).to(device)

        aggregator = train_receraser_aggregator_lightgcn(train_edge=ensemble_train_edge,models=models,aggregator=aggregator,device=device,
            pos_dict=pos_dict,num_users=num_users,num_items=num_items,epochs_agg=5,batch_size=2048,
            num_neg=4,lr=1e-3)

        # ===== final test =====
        if args.eval_mode == "full":
            ndcg, hr = RecEraser_FEnsemble(models=models,aggregator=aggregator,
                train_edge=ensemble_train_edge,test_edge=ensemble_test_edge,
                device=device,num_users=num_users,num_items=num_items,top_k=10)
        else:
            ndcg, hr = RecEraser_SEnsemble(models=models,aggregator=aggregator,train_edge=ensemble_train_edge,test_edge=ensemble_test_edge,
            device=device,pos_dict=pos_dict,num_users=num_users,num_items=num_items,top_k=10)

        print("\n====== UltraRE Final Result ======")
        print("HR@10:", hr)
        print("NDCG@10:", ndcg)






'''
# -----------------------------
# Training loop: official-style sampling per epoch + neg_k switch
# -----------------------------
def train_lightgcn(
    train_edge, test_edge,
    num_users, num_items,
    device,
    epochs=1000,
    embed_dim=64,
    K_layers=3,
    lr=1e-3,
    decay=1e-4,
    batch_size=1024,
    eval_every=50,
    neg_k=1,
    eval_mode="sampled",  # "sampled" or "full"
    learn_type = 'retrain',
    dataset = 'ml-1m',
    seed=42,
):
    
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = LightGCN(num_users, num_items, embedding_dim=embed_dim, K=K_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_edge = train_edge.to(device)
    test_edge = test_edge.to(device)

    adj_norm = build_norm_adj(train_edge, num_users, num_items, device)

    # user pos structures for sampling
    user_pos_lists = build_user_pos_lists(train_edge.cpu(), num_users)
    user_pos_sets = build_user_pos_sets(train_edge.cpu(), num_users)

    # official uses trainDataSize as sampling count per epoch
    train_size = train_edge.size(1)  # E

    best_ndcg, best_hr = 0.0, 0.0
    t0 = time.time()

    print(f"[Train] edges={train_size} | batch={batch_size} | epochs={epochs} | K={K_layers} | lr={lr} | decay={decay} | neg_k={neg_k} | eval={eval_mode}")

    for ep in range(1, epochs + 1):
        model.train()

        # 1) sample triples for this epoch (official-style)
        u_np, p_np, n_np = uniform_sample_epoch(
            num_users=num_users,
            num_items=num_items,
            train_size=train_size,
            user_pos_lists=user_pos_lists,
            user_pos_sets=user_pos_sets,
            neg_k=neg_k,
            seed=seed + ep,
        )

        # convert to torch
        u = torch.from_numpy(u_np).long().to(device)
        p = torch.from_numpy(p_np).long().to(device)
        n = torch.from_numpy(n_np).long().to(device)

        # shuffle triples
        perm = torch.randperm(u.size(0), device=device)
        u, p, n = u[perm], p[perm], n[perm]

        epoch_loss = 0.0
        steps = 0

        for start in range(0, u.size(0), batch_size):
            idx = slice(start, start + batch_size)
            u_b = u[idx]
            p_b = p[idx]
            n_b = n[idx]

            optimizer.zero_grad()

            # forward
            u_f, u0, i_f, i0 = model(adj_norm)

            loss = bpr_loss(
                u_f[u_b], i_f[p_b], i_f[n_b],
                u0[u_b], i0[p_b], i0[n_b],
                decay
            )

            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            steps += 1

        epoch_loss /= max(steps, 1)

        if ep == 1 or ep % eval_every == 0:
            if eval_mode == "full":
                hr, ndcg = evaluation_full(model, test_edge, train_edge, adj_norm, num_items, k=10, batch_size=256)
            else:
                hr, ndcg = evaluation_sampled_1p99n(model, test_edge, train_edge, adj_norm, num_items, k=10, neg_num=99, seed=seed)

            dt = time.time() - t0
            print(f"Epoch {ep:04d} | loss={epoch_loss:.4f} | HR@10={hr:.4f} | NDCG@10={ndcg:.4f} | time={dt/60:.1f}m")

            if ndcg > best_ndcg:
                best_ndcg, best_hr = ndcg, hr
                print(f"  >>> New best: NDCG@10={best_ndcg:.4f}, HR@10={best_hr:.4f}")
        
    # ==============================
    # Save final propagated embeddings
    # ==============================
    if learn_type == 'retrain':
        model.eval()
        with torch.no_grad():
            u_final, _, i_final, _ = model(adj_norm)

        u_np = u_final.cpu().numpy().astype(np.float32)
        i_np = i_final.cpu().numpy().astype(np.float32)

        user_dict = {}
        for uid in range(u_np.shape[0]):
            user_dict[np.int64(uid)] = u_np[uid:uid+1]

        item_dict = {}
        for iid in range(i_np.shape[0]):
            item_dict[np.int64(iid)] = i_np[iid:iid+1]

        np.save(f"./results/user_emb/{dataset}_lightgcn_user_emb.npy", user_dict)
        np.save(f"./results/item_emb/{dataset}_lightgcn_item_emb.npy", item_dict)

        print("Final user embeddings saved to user_embedding.npy")
        print("Final item embeddings saved to item_embedding.npy")

    return model

'''
