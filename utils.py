import pickle
import random
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
import ot
import torch
from torch import nn
import torch.nn.functional as F

STD = 0.01


def hit(gt_items, pred_items):
    hr = 0
    for gt_item in gt_items:
        if gt_item in pred_items:
            hr = hr + 1

    return hr / len(gt_items)


def ndcg(gt_items, pred_items):
    dcg = 0
    idcg = 0

    for gt_item in gt_items:
        if gt_item in pred_items:
            index = pred_items.index(gt_item)
            dcg = dcg + np.reciprocal(np.log2(index + 2))

    for index in range(len(gt_items)):
        idcg = idcg + np.reciprocal(np.log2(index + 2))

    return dcg / idcg


##################### 
# model training
##################### 

# seed everything
def seed_all(seed):
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class WMF(nn.Module):
    def __init__(self, n_user, n_item, k=16):
        super(WMF, self).__init__()
        self.k = k
        self.user_mat = nn.Embedding(n_user, k)
        self.item_mat = nn.Embedding(n_item, k)
        self.init_weight()

    def init_weight(self):
        nn.init.normal_(self.user_mat.weight, std=0.01)
        nn.init.normal_(self.item_mat.weight, std=0.01)

    def forward(self, uid, iid):
        return (self.user_mat(uid) * self.item_mat(iid)).sum(dim=1)


class BPR(nn.Module):
    def __init__(self, n_user, n_item, k=16):
        super(BPR, self).__init__()
        self.k = k
        self.user_mat = nn.Embedding(n_user, k)
        self.item_mat = nn.Embedding(n_item, k)
        self.func = nn.Sigmoid()
        self.init_weight()

    def init_weight(self):
        nn.init.normal_(self.user_mat.weight, std=0.01)
        nn.init.normal_(self.item_mat.weight, std=0.01)

    def forward(self, uid, iid):
        return (self.user_mat(uid) * self.item_mat(iid)).sum(dim=1)

        # r_pos = (self.user_mat(uid) * self.item_mat(pos_id)).sum(dim=1)
        # r_neg = (self.user_mat(uid) * self.item_mat(neg_id)).sum(dim=1)

        # return self.func(r_pos - r_neg)

# build model DMF
class DMF(nn.Module):
    def __init__(self, n_user, n_item, k=16, layers=[64, 32]):
        super(DMF, self).__init__()
        self.k = k
        self.user_mat = nn.Embedding(n_user, k)
        self.item_mat = nn.Embedding(n_item, k)
        self.layers = [k]
        self.layers += layers
        self.user_fc = nn.ModuleList()
        self.item_fc = nn.ModuleList()
        self.cos = nn.CosineSimilarity()

        for (in_size, out_size) in zip(self.layers[:-1], self.layers[1:]):
            self.user_fc.append(nn.Linear(in_size, out_size))
            self.item_fc.append(nn.Linear(in_size, out_size))
            self.user_fc.append(nn.ReLU())
            self.item_fc.append(nn.ReLU())

        self.init_weight()

    def init_weight(self):
        nn.init.normal_(self.user_mat.weight, std=STD)
        nn.init.normal_(self.item_mat.weight, std=STD)

        for i in self.modules():
            if isinstance(i, nn.Linear):
                nn.init.xavier_uniform_(i.weight)
                if i.bias is not None:
                    i.bias.data.zero_()

    def forward(self, uid, iid):
        user_embedding = self.user_mat(uid)
        item_embedding = self.item_mat(iid)
        for i in range(len(self.user_fc)):
            user_embedding = self.user_fc[i](user_embedding)
            item_embedding = self.item_fc[i](item_embedding)
        rating = self.cos(user_embedding, item_embedding)
        return rating.squeeze()


# build model Neural MF -> batch size: 256, lr: 0.001, laryers: [64,32,16,8]
class NeuMF(nn.Module):
    def __init__(self, n_user, n_item, k=16, layser=[64, 32]):
        super(NeuMF, self).__init__()
        self.k = k
        self.k_mlp = int(layser[0] / 2)

        self.user_mat_mf = nn.Embedding(n_user, k)
        self.item_mat_mf = nn.Embedding(n_item, k)
        self.user_mat_mlp = nn.Embedding(n_user, self.k_mlp)
        self.item_mat_mlp = nn.Embedding(n_item, self.k_mlp)

        self.layers = layser
        self.fc = nn.ModuleList()
        for (in_size, out_size) in zip(self.layers[:-1], self.layers[1:]):
            self.fc.append(nn.Linear(in_size, out_size))
            self.fc.append(nn.ReLU())

        self.affine = nn.Linear(self.layers[-1] + self.k, 1)
        self.logistic = nn.Sigmoid()
        self.init_weight()

    def init_weight(self):
        nn.init.normal_(self.user_mat_mf.weight, std=STD)
        nn.init.normal_(self.item_mat_mf.weight, std=STD)
        nn.init.normal_(self.user_mat_mlp.weight, std=STD)
        nn.init.normal_(self.item_mat_mlp.weight, std=STD)

        for i in self.modules():
            if isinstance(i, nn.Linear):
                nn.init.xavier_uniform_(i.weight)
                if i.bias is not None:
                    i.bias.data.zero_()
        # for i in self.fc:
        #     if isinstance(i, nn.Linear):
        #         nn.init.xavier_uniform_(i.weight)
        #         if i.bias is not None:
        #             i.bias.data.zero_()

        # nn.init.xavier_uniform_(self.affine.weight)
        # if self.affine.bias is not None:
        #     self.affine.bias.data.zero_()

    def forward(self, uid, iid):
        user_embedding_mlp = self.user_mat_mlp(uid)
        item_embedding_mlp = self.item_mat_mlp(iid)

        user_embedding_mf = self.user_mat_mf(uid)
        item_embedding_mf = self.item_mat_mf(iid)

        mlp_vec = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)
        mf_vec = torch.mul(user_embedding_mf, item_embedding_mf)

        for i in range(len(self.fc)):
            mlp_vec = self.fc[i](mlp_vec)

        vec = torch.cat([mlp_vec, mf_vec], dim=-1)
        logits = self.affine(vec)
        #rating = self.logistic(logits)
        #logits.squeeze()
        return logits


# torch train
def baseTrain(dataloader, model, loss_fn, opt, device, verbose):
    size = len(dataloader.dataset)
    train_loss = 0

    dataloader.dataset.ng_sample(4)

    model.train(True)
    if loss_fn == 'point-wise':
        loss_func = nn.BCEWithLogitsLoss()
        for batch, (user, item, rating) in enumerate(dataloader):
            user = user.to(device)
            item = item.to(device)
            rating = rating.to(device)

            #pred = model(user, item)
            #loss = loss_func(pred, rating)

            pred = model(user, item).view(-1)
            rating = rating.view(-1)
            loss = loss_func(pred, rating)


            train_loss += loss.item()

            opt.zero_grad()
            loss.backward()
            opt.step()

    elif loss_fn == 'pair-wise':
        sig_func = nn.Sigmoid()

        for batch, (user, pos, neg) in enumerate(dataloader):
            user = user.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            pred_pos = model(user, pos)
            pred_neg = model(user, neg)

            loss = (-torch.log(sig_func(pred_pos - pred_neg))).sum()

            train_loss += loss.item()

            opt.zero_grad()
            loss.backward()
            opt.step()

    return train_loss / size


# torch test
def baseTest(dataloader, model, loss_fn, device, verbose, pos_dict, n_items, top_k=10, user_mapping=None,
             pos_mapping=None):
    
    model.eval()

    full_items = [i for i in range(n_items)]

    HR = []
    NDCG = []

    with torch.no_grad():
        for user, item, rating in dataloader:
            all_users = user.unique()
            all_users = all_users.to(device)
            user = user.to(device)
            item = item.to(device)

            for uid in all_users:
                user_id = uid.item()
                user_indices = torch.where(user == uid)
                gt_items = item[user_indices].cpu().numpy().tolist()


                neg_pool = list(set(full_items) - set(pos_dict[user_id]))
                neg_items = random.sample(neg_pool, 99)

                candidates = gt_items + neg_items

                new_user = torch.tensor([user_id] * len(candidates), dtype=torch.long).to(device)
                new_item = torch.tensor(candidates, dtype=torch.long).to(device)
                #oreductuibs
                predictions = model(new_user, new_item).view(-1)
                _, indices = torch.topk(predictions, top_k)
                recommends = torch.take(new_item, indices).cpu().numpy().tolist()

                HR.append(hit(gt_items, recommends))
                NDCG.append(ndcg(gt_items, recommends))

    return np.mean(NDCG), np.mean(HR)

def SISATest_ensemble(dataloader, models, device, pos_dict, n_items, top_k=10):

    for m in models:
        m.eval()

    full_items = list(range(n_items))

    HR = []
    NDCG = []

    with torch.no_grad():

        for user, item, rating in dataloader:

            user = user.to(device)
            item = item.to(device)

            all_users = user.unique().to(device)

            for uid in all_users:

                user_id = uid.item()
                user_indices = torch.where(user == uid)
                gt_items = item[user_indices].cpu().numpy().tolist()

                #negative sampling
                neg_pool = list(set(full_items) - set(pos_dict[user_id]))
                neg_items = random.sample(neg_pool, 99)

                candidates = gt_items + neg_items

                new_user = torch.tensor([user_id] * len(candidates),dtype=torch.long).to(device)
                new_item = torch.tensor(candidates,dtype=torch.long).to(device)

                #Ensemble average logits
                total_logits = None

                for m in models:
                    #logits = m(new_user, new_item)
                    logits = m(new_user, new_item).view(-1)
                    
                    if total_logits is None:
                        total_logits = logits
                    else:
                        total_logits += logits

                avg_logits = total_logits / len(models)

                _, indices = torch.topk(avg_logits, top_k)

                recommends = torch.take(new_item, indices).cpu().numpy().tolist()

                HR.append(hit(gt_items, recommends))
                NDCG.append(ndcg(gt_items, recommends))

    return np.mean(NDCG), np.mean(HR)

# shrink and perturb
def spTrick(model, shrink=0.5, sigma=0.01):
    for (name, param) in model.named_parameters():
        if 'weight' in name:
            for i in range(param.shape[0]):
                for j in range(param.shape[1]):
                    param.data[i][j] = shrink * param.data[i][j] + torch.normal(0.0, sigma, size=(1, 1))
    return model



#RecEraser Ebsembling model
def RecEraserTest_ensemble(dataloader, models, aggregator, device, pos_dict, n_items, top_k=10):

    for m in models:
        m.eval()
    aggregator.eval()

    full_items = list(range(n_items))
    HR, NDCG = [], []

    with torch.no_grad():
        for user, item, rating in dataloader:
            user = user.to(device)
            item = item.to(device)

            for uid in user.unique():
                user_id = uid.item()
                user_indices = torch.where(user == uid)
                gt_items = item[user_indices].cpu().numpy().tolist()

                neg_pool = list(set(full_items) - set(pos_dict[user_id]))
                neg_items = random.sample(neg_pool, 99)
                candidates = gt_items + neg_items

                new_user = torch.tensor([user_id] * len(candidates), dtype=torch.long).to(device)
                new_item = torch.tensor(candidates, dtype=torch.long).to(device)

                # collect shard embeddings
                user_embs, item_embs = [], []
                for m in models:
                    if hasattr(m, "user_mat_mlp"):
                        u_emb = torch.cat([m.user_mat_mlp(new_user), m.user_mat_mf(new_user)], dim=1)
                        i_emb = torch.cat([m.item_mat_mlp(new_item), m.item_mat_mf(new_item)], dim=1)
                    elif hasattr(m, "user_mat"):
                        u_emb = m.user_mat(new_user)
                        i_emb = m.item_mat(new_item)
                        # DMF
                    elif hasattr(m, "user_fc"):
                        u_emb = m.user_mat(new_user)
                        i_emb = m.item_mat(new_item)
                    else:
                        raise NotImplementedError
                    user_embs.append(u_emb.unsqueeze(1))
                    item_embs.append(i_emb.unsqueeze(1))

                user_embs = torch.cat(user_embs, dim=1)  # [C,S,D]
                item_embs = torch.cat(item_embs, dim=1)  # [C,S,D]

                u_agg, i_agg = aggregator(user_embs, item_embs)
                scores = torch.sum(u_agg * i_agg, dim=1)

                _, indices = torch.topk(scores, top_k)
                recommends = torch.take(new_item, indices).cpu().numpy().tolist()

                HR.append(hit(gt_items, recommends))
                NDCG.append(ndcg(gt_items, recommends))

    return float(np.mean(NDCG)), float(np.mean(HR))


##################### 
# object saving
##################### 

def saveObject(filename, obj):
    with open(filename + '.pkl', 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def loadObject(filename):
    with open(filename + '.pkl', 'rb') as input:
        obj = pickle.load(input)
    return obj

def kmeans_InBP(train_ratings, test_ratings, dataset, num_groups, model_type, max_iters=50):
    """
    Interaction-based Balanced Partition (RecEraser InBP)
    Returns:
        train_rating_groups: list[pd.DataFrame]  (ONLY assigned interactions)
        test_rating_groups:  list[pd.DataFrame]  (users appearing in this shard's train interactions)
    """

    start_time = time.time()

    #IMPORTANT: reset index so that "idx" aligns with iloc positions
    train_ratings = train_ratings.reset_index(drop=True)

    # ===== load embeddings =====
    user_emb = np.load(
        f'results/user_emb/{dataset}_neumf_user_emb.npy',
        allow_pickle=True
    ).item()

    item_emb = np.load(
        f'results/item_emb/{dataset}_neumf_item_emb.npy',
        allow_pickle=True
    ).item()

    # ===== build interaction list aligned with train_ratings rows =====
    # data[idx] corresponds to train_ratings.iloc[idx]
    data = [(int(row['user']), int(row['item'])) for _, row in train_ratings.iterrows()]

    n = len(data)
    k = num_groups
    max_data = int(np.ceil(1.2 * n / k))

    # ===== randomly initialize anchors =====
    centroids = random.sample(data, k)

    centroembs = []
    for u, i in centroids:
        pu = user_emb[np.int64(u)][0]
        qi = item_emb[np.int64(i)][0]
        centroembs.append((pu, qi))

    # ===== main loop =====
    for _ in range(max_iters):

        # C is still useful for updating anchors (user->items), but NOT for building train_group
        C = [{} for _ in range(k)]
        C_num = [0] * k

        #This is the key: store assigned interaction indices for each shard
        shard_indices = [[] for _ in range(k)]

        Scores = {}

        # compute distance score for every (interaction idx, shard j)
        for idx, (u, i) in enumerate(data):

            pu = user_emb[np.int64(u)][0]
            qi = item_emb[np.int64(i)][0]

            for j in range(k):
                cu, ci = centroembs[j]

                dist_u = np.sum((pu - cu) ** 2)
                dist_i = np.sum((qi - ci) ** 2)

                # larger score = closer (because we sort descending)
                Scores[(idx, j)] = -(dist_u * dist_i)

        Scores = sorted(Scores.items(), key=lambda x: x[1], reverse=True)

        assigned = set()

        for (idx, shard_id), _ in Scores:

            if idx in assigned:
                continue
            if C_num[shard_id] >= max_data:
                continue

            u, i = data[idx]

            # record interaction index
            shard_indices[shard_id].append(idx)

            # maintain C for centroid update
            if u not in C[shard_id]:
                C[shard_id][u] = [i]
            else:
                C[shard_id][u].append(i)

            C_num[shard_id] += 1
            assigned.add(idx)

            if len(assigned) == n:
                break

        # ===== update anchors =====
        new_centroembs = []
        for shard_id in range(k):

            temp_u = []
            temp_i = []

            for u in C[shard_id]:
                for i in C[shard_id][u]:
                    temp_u.append(user_emb[np.int64(u)][0])
                    temp_i.append(item_emb[np.int64(i)][0])

            if len(temp_u) == 0:
                # empty shard fallback: keep old centroid
                new_centroembs.append(centroembs[shard_id])
            else:
                new_centroembs.append((
                    np.mean(temp_u, axis=0),
                    np.mean(temp_i, axis=0)
                ))

        # convergence check
        if all(
            np.allclose(new_centroembs[i][0], centroembs[i][0]) and
            np.allclose(new_centroembs[i][1], centroembs[i][1])
            for i in range(k)
        ):
            centroembs = new_centroembs
            break

        centroembs = new_centroembs

    # ===== build shard datasets (FIXED: interaction-based slicing) =====
    train_rating_groups = []
    test_rating_groups = []

    for shard_id in range(k):

        #train shard = ONLY assigned interactions
        shard_idx = shard_indices[shard_id]
        train_group = train_ratings.iloc[shard_idx].reset_index(drop=True)

        #test shard = users appearing in this shard’s train interactions
        shard_users = train_group['user'].unique()
        test_group = test_ratings[test_ratings['user'].isin(shard_users)].reset_index(drop=True)

        train_rating_groups.append(train_group)
        test_rating_groups.append(test_group)

    print(f'Grouping time: {time.time() - start_time:.2f}s')

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
    max_iters=50,
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


class RecEraserAggregator(nn.Module):
    def __init__(self, emb_dim, num_shards, att_dim=64):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_shards = num_shards

        # shard -> shared space
        self.trans_W = nn.Parameter(torch.randn(num_shards, emb_dim, emb_dim) * 0.01)
        self.trans_b = nn.Parameter(torch.zeros(num_shards, emb_dim))

        # user attention
        self.WA = nn.Linear(emb_dim, att_dim)
        self.HA = nn.Linear(att_dim, 1)

        # item attention
        self.WB = nn.Linear(emb_dim, att_dim)
        self.HB = nn.Linear(att_dim, 1)

    def _attn(self, embs, is_user: bool):
        # embs: [B, S, D]
        if is_user:
            x = torch.relu(self.WA(embs))
            score = self.HA(x)  # [B,S,1]
        else:
            x = torch.relu(self.WB(embs))
            score = self.HB(x)

        w = torch.softmax(score, dim=1)          # [B,S,1]
        agg = torch.sum(w * embs, dim=1)         # [B,D]
        return agg, w

    def aggregate_users(self, user_embs):
        # user_embs: [B,S,D]
        u_trans = torch.einsum('bsd,sdh->bsh', user_embs, self.trans_W) + self.trans_b
        u_agg, _ = self._attn(u_trans, is_user=True)
        return u_agg

    def aggregate_items(self, item_embs):
        # item_embs: [B,S,D]
        i_trans = torch.einsum('bsd,sdh->bsh', item_embs, self.trans_W) + self.trans_b
        i_agg, _ = self._attn(i_trans, is_user=False)
        return i_agg

    def forward(self, user_embs, item_embs):
        return self.aggregate_users(user_embs), self.aggregate_items(item_embs)

def train_receraser_aggregator(
    train_df,               # pandas.DataFrame with columns ['user','item']
    models,                 # list of shard models (already loaded)
    aggregator,             # RecEraserAggregator
    device,
    pos_dict,               # dict: user -> set(pos_items)  (你已有 npy)
    n_items: int,
    epochs_agg: int = 5,
    batch_size: int = 2048,
    num_neg: int = 4,
    lr: float = 1e-3
):
    """
    训练 aggregator，让它学会对不同 shard 的 embeddings 分配权重。
    shard 模型参数固定（stop-gradient 类似），只更新 aggregator。
    """
    aggregator.train()
    opt = torch.optim.Adam(aggregator.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    users = train_df['user'].to_numpy()
    items = train_df['item'].to_numpy()
    N = len(users)

    for m in models:
        m.eval()  # shard 冻结

    for ep in range(epochs_agg):
        idx = np.random.permutation(N)

        total_loss = 0.0
        steps = 0

        for st in range(0, N, batch_size):
            ed = min(st + batch_size, N)
            batch_idx = idx[st:ed]
            u_pos = users[batch_idx]
            i_pos = items[batch_idx]

            # ===== negative sampling =====
            u_all = []
            i_all = []
            y_all = []

            for u, ip in zip(u_pos, i_pos):
                u_all.append(u); i_all.append(ip); y_all.append(1.0)

                neg_pool = list(set(range(n_items)) - set(pos_dict[int(u)]))
                # 防止某些用户正样本太多导致 neg_pool 太小
                if len(neg_pool) == 0:
                    continue
                negs = random.choices(neg_pool, k=num_neg)
                for ineg in negs:
                    u_all.append(u); i_all.append(ineg); y_all.append(0.0)

            u_all = torch.tensor(u_all, dtype=torch.long, device=device)
            i_all = torch.tensor(i_all, dtype=torch.long, device=device)
            y_all = torch.tensor(y_all, dtype=torch.float32, device=device)

            # ===== collect shard embeddings (no grad for shard) =====
            with torch.no_grad():
                user_embs = []
                item_embs = []
                for m in models:
                    if hasattr(m, "user_mat_mlp"):   # NeuMF
                        u_mlp = m.user_mat_mlp(u_all)
                        u_mf  = m.user_mat_mf(u_all)
                        u_emb = torch.cat([u_mlp, u_mf], dim=1)

                        it_mlp = m.item_mat_mlp(i_all)
                        it_mf  = m.item_mat_mf(i_all)
                        it_emb = torch.cat([it_mlp, it_mf], dim=1)

                    elif hasattr(m, "user_mat"):     # WMF / BPR
                        u_emb = m.user_mat(u_all)
                        it_emb = m.item_mat(i_all)

                    else:
                        raise NotImplementedError

                    #u_mlp = m.user_mat_mlp(u_all)
                    #u_mf  = m.user_mat_mf(u_all)
                    #u_emb = torch.cat([u_mlp, u_mf], dim=1)

                    #it_mlp = m.item_mat_mlp(i_all)
                    #it_mf  = m.item_mat_mf(i_all)
                    #it_emb = torch.cat([it_mlp, it_mf], dim=1)

                    user_embs.append(u_emb.unsqueeze(1))
                    item_embs.append(it_emb.unsqueeze(1))

                user_embs = torch.cat(user_embs, dim=1)  # [B,S,D]
                item_embs = torch.cat(item_embs, dim=1)  # [B,S,D]

            # ===== aggregator forward (grad) =====
            u_agg, i_agg = aggregator(user_embs, item_embs)
            logits = torch.sum(u_agg * i_agg, dim=1)  # dot product score

            loss = bce(logits, y_all)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            steps += 1

        print(f"[agg epoch {ep+1}/{epochs_agg}] loss={total_loss/max(steps,1):.6f}")

    aggregator.eval()
    return aggregator


#Heatmap
def visualize_all_shard_heatmaps(
    train_rating_groups,
    dataset,
    model_type,
    save_dir="results/shard_heatmaps",
    max_samples=300
):
    """
    Visualize interaction similarity heatmap for ALL shards.

    train_rating_groups: list of pd.DataFrame
    """

    os.makedirs(save_dir, exist_ok=True)

    # ===== load embeddings =====
    user_emb = np.load(
        f'results/user_emb/{dataset}_{model_type}_user_emb.npy',
        allow_pickle=True
    ).item()

    item_emb = np.load(
        f'results/item_emb/{dataset}_{model_type}_item_emb.npy',
        allow_pickle=True
    ).item()

    print("\nGenerating shard heatmaps...\n")

    for shard_id, train_group in enumerate(train_rating_groups):

        if len(train_group) == 0:
            print(f"Shard {shard_id} empty, skip.")
            continue

        # ===== sample interactions to avoid huge matrix =====
        if len(train_group) > max_samples:
            train_group = train_group.sample(max_samples, random_state=42)

        Z = []

        for _, row in train_group.iterrows():
            u = int(row["user"])
            i = int(row["item"])

            pu = user_emb[np.int64(u)][0]
            qi = item_emb[np.int64(i)][0]

            z = np.concatenate([pu, qi])
            Z.append(z)

        Z = np.array(Z)

        # ===== normalize =====
        Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)

        # ===== similarity matrix =====
        sim_matrix = cosine_similarity(Z)

        # ===== compute structural score =====
        mean_sim = sim_matrix[np.triu_indices(len(sim_matrix), 1)].mean()

        # ===== plot =====
        plt.figure(figsize=(6, 5))
        sns.heatmap(sim_matrix, cmap="viridis", cbar=False)
        plt.title(f"Shard {shard_id} | MeanSim={mean_sim:.3f}")
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"shard_{shard_id}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"Shard {shard_id} saved → {save_path} | MeanSim={mean_sim:.4f}")

    print("\nAll shard heatmaps generated.\n")










'''
def kmeans_UBP(X, k, balance=False, max_iters=10):
    # Initialize centroids randomly
    n, _ = X.shape
    group_len = int(np.ceil(n / k))
    centroid = X[np.random.choice(n, size=k, replace=False)]

    dist = ((X - centroid[:, np.newaxis]) ** 2).sum(axis=2)  # [k, n]

    inertia = np.min(dist, axis=0).sum()

    # Iterate until convergence or maximum iterations
    for _ in range(max_iters):
        # print(_)
        # Assign each sample to the nearest centroid
        dist = ((X - centroid[:, np.newaxis]) ** 2).sum(axis=2)  # [k, n]
        if balance:
            label_count = [group_len] * k
            assinged_sample = []
            assigned_dict = set()

            label = np.zeros(n)
            inertia = 0

            flat_idx_sorted = np.argsort(dist.ravel())[::-1]

            # print(np.argsort(dist.ravel())[::-1].shape)

            row_idx, col_idx = np.unravel_index(flat_idx_sorted, dist.shape)

            for val, cen_idx, sample_idx in zip(dist[row_idx, col_idx], row_idx, col_idx):

                if len(assigned_dict) == n:
                    break
                # if sample_idx in assinged_sample:
                if sample_idx in assigned_dict:
                    continue
                if label_count[cen_idx] > 0:
                    label[sample_idx] = cen_idx
                    # assinged_sample.append(sample_idx)
                    assigned_dict.add(sample_idx)
                    label_count[cen_idx] -= 1
                    inertia += val
        else:
            label = np.argmin(dist, axis=0)
            inertia = np.min(dist, axis=0).sum()

        # Update centroids to the mean of assigned samples
        new_centroid = np.array([X[label == i].mean(axis=0) for i in range(k)])

        # Check if centroids have converged
        if np.allclose(centroid, new_centroid):
            break

        centroid = new_centroid

    print(f'{inertia:.3f}', end=' ')
    return inertia, label  # , centroid


def ot_cluster(X, k, max_iters=10):
    # Initialize centroids randomly
    n, _ = X.shape
    centroid = X[np.random.choice(n, size=k, replace=False)]

    # Iterate until convergence or maximum iterations

    for _ in range(max_iters):
        # compute distance
        dist = ((X - centroid[:, np.newaxis]) ** 2).sum(axis=2)  # [k, n]

        # print(dist.shape)
        inertia = np.min(dist, axis=0).sum()

        # compute sinkhorn distance
        lam = 1e-3
        a = np.ones(n) / n
        b = np.ones(k) / k
        # b = np.array([0.1, 0.2, 0.3, 0.4])

        trans = ot.emd(a, b, dist.T, lam)

        # Update centroids to the mean of assigned samples
        label = np.argmax(trans, axis=1)
        new_centroid = np.array([X[label == i].mean(axis=0) for i in range(k)])

        # Check if centroids have converged
        if np.allclose(centroid, new_centroid):
            break

        centroid = new_centroid
    return inertia, label  # , centroids


def ot_assignment(trans: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    """
    Given OT transport plan trans of shape [n, k],
    produce a HARD assignment label with STRICT balance:
      - each shard gets either floor(n/k) or ceil(n/k) samples
      - assignment prioritizes larger transport mass
    """
    rng = np.random.default_rng(seed)
    n = trans.shape[0]

    # target sizes: first r shards get (q+1), others get q
    q, r = divmod(n, k)
    targets = np.array([q + 1 if j < r else q for j in range(k)], dtype=int)

    label = -np.ones(n, dtype=np.int64)
    assigned = np.zeros(n, dtype=bool)
    shard_count = np.zeros(k, dtype=int)

    # To reduce bias, process shards in a random order each time
    shard_order = np.arange(k)
    rng.shuffle(shard_order)

    for j in shard_order:
        # sort interactions by transport mass to shard j (descending)
        idx_sorted = np.argsort(-trans[:, j], kind="mergesort")

        need = targets[j]
        if need == 0:
            continue

        cnt = 0
        for i in idx_sorted:
            if not assigned[i]:
                label[i] = j
                assigned[i] = True
                shard_count[j] += 1
                cnt += 1
                if cnt >= need:
                    break

    # If anything unassigned (should be rare), assign remaining to shards with remaining capacity
    if not assigned.all():
        remaining_idx = np.where(~assigned)[0]
        # compute remaining capacities
        cap = targets - shard_count
        # fill leftover
        ptrs = []
        for j in range(k):
            if cap[j] > 0:
                ptrs.extend([j] * cap[j])
        ptrs = np.array(ptrs, dtype=np.int64)

        # If still mismatch due to some edge case, just cycle
        if len(ptrs) < len(remaining_idx):
            extra = len(remaining_idx) - len(ptrs)
            ptrs = np.concatenate([ptrs, np.tile(np.arange(k), int(np.ceil(extra / k)))[:extra]])

        for i, j in zip(remaining_idx, ptrs[:len(remaining_idx)]):
            label[i] = j

    # Sanity check
    counts = np.bincount(label, minlength=k)
    assert counts.sum() == n
    assert counts.max() - counts.min() <= 1, f"Not strictly balanced! counts={counts}"

    return label


def kmeans_ot_InBP(train_ratings,test_ratings,dataset,num_groups,model_type,max_iters=20,reg=1e-2,seed=42,verbose=True):
    """
    Interaction-level Optimal Balanced Clustering (OBC-ish):
    - Build interaction embeddings from pretrained user/item embeddings.
    - Iteratively:
        (1) compute cost to centroids
        (2) compute OT transport plan with balanced marginals
        (3) produce STRICT balanced hard labels from OT transport
        (4) update centroids by assigned interactions
    - Build train shards by interaction indices.
    - Build test shards by users appearing in each train shard.
    """
    start_time = time.time()
    rng = np.random.default_rng(seed)

    # ===== 1) load pretrained embeddings =====
    user_emb = np.load(f"results/user_emb/{dataset}_{model_type}_user_emb.npy", allow_pickle=True).item()
    item_emb = np.load(f"results/item_emb/{dataset}_{model_type}_item_emb.npy", allow_pickle=True).item()

    # ===== 2) build interaction embedding matrix X =====
    users = train_ratings["user"].to_numpy(dtype=np.int64)
    items = train_ratings["item"].to_numpy(dtype=np.int64)
    n = len(users)
    k = int(num_groups)

    # emb dict values shape (1,d) => take [0]
    d_u = user_emb[np.int64(users[0])][0].shape[0]
    d_i = item_emb[np.int64(items[0])][0].shape[0]

    X = np.empty((n, d_u + d_i), dtype=np.float32)
    for idx in range(n):
        X[idx, :d_u] = user_emb[np.int64(users[idx])][0]
        X[idx, d_u:] = item_emb[np.int64(items[idx])][0]

    # ===== 3) init centroids =====
    init_idx = rng.choice(n, size=k, replace=False)
    centroid = X[init_idx].copy()  # [k, 2d]

    # ===== 4) OT loop =====
    a = np.ones(n, dtype=np.float64) / n
    b = np.ones(k, dtype=np.float64) / k

    last_label = None
    for it in range(max_iters):
        # cost matrix [n,k] (squared Euclidean)
        dist = ((X[:, None, :] - centroid[None, :, :]) ** 2).sum(axis=2).astype(np.float64)

        # OT transport plan [n,k]
        trans = ot.sinkhorn(a, b, dist, reg)

        # STRICT balanced hard assignment (replaces argmax)
        label = ot_assignment(trans, k=k, seed=seed + it)

        # update centroids
        new_centroid = centroid.copy()
        for j in range(k):
            mask = (label == j)
            if np.any(mask):
                new_centroid[j] = X[mask].mean(axis=0)

        if last_label is not None and np.array_equal(label, last_label):
            centroid = new_centroid
            break

        if np.allclose(centroid, new_centroid):
            centroid = new_centroid
            break

        centroid = new_centroid
        last_label = label

    # ===== 5) build shard datasets =====
    train_rating_groups = []
    test_rating_groups = []

    for shard_id in range(k):
        shard_idx = np.where(label == shard_id)[0]

        train_group = train_ratings.iloc[shard_idx].reset_index(drop=True)

        shard_users = train_group["user"].unique()
        test_group = test_ratings[test_ratings["user"].isin(shard_users)].reset_index(drop=True)

        train_rating_groups.append(train_group)
        test_rating_groups.append(test_group)

    if verbose:
        counts = np.bincount(label, minlength=k)
        print("\n[OBC Interaction-level] Strict balanced shard sizes (train interactions):")
        for j in range(k):
            print(f"  Shard {j}: {counts[j]}")
        print(f"  Mean={counts.mean():.2f}, Std={counts.std():.2f}, Max-Min={counts.max()-counts.min()}")

        test_counts = [len(g) for g in test_rating_groups]
        print("\nTest interactions per shard (after user-filter):")
        for j, c in enumerate(test_counts):
            print(f"  Shard {j}: {c}")
        print(f"  Mean={np.mean(test_counts):.2f}, Std={np.std(test_counts):.2f}, Max-Min={np.max(test_counts)-np.min(test_counts)}")

        print(f"\nUltraRE OBC Grouping time: {time.time() - start_time:.2f}s")

    return train_rating_groups, test_rating_groups
'''
