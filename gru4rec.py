import os
import math
import numpy as np
import pandas as pd
import torch
from torch import autograd, nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from collections import OrderedDict
import time

from utils import ot_cluster, kmeans_InBP

def read_session_data(path, del_users=[], del_type='random', del_per=0.0, seed=42):
    """
    TODO: del_users might be deprecated if we don't do user-level deletion.
    Reads sequential data from a CSV file into a pandas DataFrame.
    It can perform interaction-level deletion and user-level deletion.

    Args:
        path (str): Path to the CSV data file.
        del_users (list): A list of user IDs to remove from the data.
        del_type (str): Type of deletion ('random', 'core', 'edge').
        del_per (float): Percentage of interactions to remove.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: A tuple containing:
               - pd.DataFrame: The processed data.
               - list: A list of user IDs affected by the deletion.
    """
    df = pd.read_csv(path, sep=',')
    # Rename columns from converted CSV format (uid, iid, timestamp) to (SessionId, ItemId, Time)
    df.rename(columns={'uid': 'SessionId', 'iid': 'ItemId', 'timestamp': 'Time'}, inplace=True)
    
    affected_users = []

    # --- Interaction-level deletion ---
    if del_per > 0:
        total_rows = len(df)
        num_delete = int(total_rows * del_per / 100)
        
        # Determine target sessions based on del_type
        session_counts = df['SessionId'].value_counts()
        if del_type == 'random':
            target_sessions = session_counts.index.tolist()
        elif del_type == 'core':
            # top 5% users with most interactions
            num_active = max(1, int(len(session_counts) * 0.05))
            target_sessions = session_counts.sort_values(ascending=False).index[:num_active]
        elif del_type == 'edge':
            # sessions outside top 5%
            num_active = int(len(session_counts) * 0.05)
            target_sessions = session_counts.sort_values(ascending=False).index[num_active:]
        else:
            raise ValueError(f"Unknown del_type: {del_type}")

        # Leave 2 interactions per session
        deletable_idx = []
        min_inter = 2
        
        target_df = df[df['SessionId'].isin(target_sessions)]
        for session, group in target_df.groupby('SessionId'):
            indices = group.index.tolist()
            if len(indices) > min_inter:
                deletable_idx.extend(indices[:(len(indices) - min_inter)])

        if len(deletable_idx) > 0:
            rng = np.random.default_rng(seed)
            num_to_drop = min(num_delete, len(deletable_idx))
            indices_to_remove = rng.choice(deletable_idx, size=num_to_drop, replace=False)
            
            # Identify affected users (using SessionId as proxy)
            affected_users = list(df.loc[indices_to_remove]['SessionId'].unique())
            
            df = df.drop(indices_to_remove).reset_index(drop=True)
            print(f"Removed {num_to_drop} interactions ({del_per:.2f}%, type={del_type}) from {path}")

    # Add UserId as a proxy for SessionId for sharding
    if 'UserId' not in df.columns:
        df['UserId'] = df['SessionId']

    # --- User-level deletion ---
    if del_users:
        df = df[~df['UserId'].isin(del_users)]
        # The affected users are the ones we just deleted
        affected_users = list(set(affected_users + del_users))
        print(f"Removed sessions for {len(del_users)} users.")

    return df, affected_users


def readSequence(path, del_per=0, seed=42): # Added del_per and seed
    """
    Reads sequential data from a CSV file, groups by session,
    generates (input_sequence, target_item) pairs,
    and re-indexes item IDs to be 0-based and contiguous.
    Can also perform interaction-level deletion.

    Args:
        path (str): Path to the CSV data file (e.g., train.csv).
        del_per (float): Percentage of interactions to remove (0.0 to 100.0).
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: A tuple containing:
               - list: A list of tuples, where each tuple is
                       (list_of_input_item_ids, target_item_id).
               - int: The total number of unique, re-indexed items.
    """
    df, _ = read_session_data(path, del_per=del_per, seed=seed)

    # Create itemidmap for 0-indexed, contiguous item IDs
    unique_items = df['ItemId'].unique()
    n_items = len(unique_items)
    item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
    df['ItemId'] = df['ItemId'].map(item_to_idx)

    # Sort by session and then by time to ensure correct sequence order
    df = df.sort_values(by=['SessionId', 'Time'])

    all_sequences = []
    # Group by SessionId and process each session
    for session_id, group in df.groupby('SessionId'):
        items = group['ItemId'].tolist()
        
        # Generate (input_sequence, target_item) pairs for each session
        if len(items) > 1: # A sequence needs at least two items (one input, one target)
            for i in range(1, len(items)):
                input_seq = items[:i]
                target_item = items[i]
                all_sequences.append((input_seq, target_item))
    
    return all_sequences, n_items

class SequentialData(Dataset):
    def __init__(self, sequences):
        """
        Args:
            sequences (list): A list of tuples, where each tuple is
                              (list_of_input_item_ids, target_item_id).
        """
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        input_sequence, target_item = self.sequences[idx]
        
        # The collate_fn will handle padding these to the same length
        return (torch.tensor(input_sequence, dtype=torch.long),
                torch.tensor(target_item, dtype=torch.long))

class GRU4RecWrapper: 
    def __init__(self, param):
        self.param = param
        
        # We need to manually set device for the wrapper, as Scratch used to do this.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Map parameters from the UltraRE 'param' object to what GRU4RecOriginal expects.
        self.model = GRU4RecOriginal(
            layers=self.param.layers,
            loss=self.param.loss,
            batch_size=self.param.batch,
            learning_rate=self.param.lr,
            momentum=self.param.momentum,
            n_epochs=self.param.epoch,
            device=self.device,
            constrained_embedding=False,
            embedding=self.param.layers[0], # Use first layer's size as embedding dim
            dropout_p_embed=self.param.dropout_p_embed,
            dropout_p_hidden=self.param.dropout_p_hidden,
            n_sample=self.param.n_sample,
            sample_alpha=self.param.sample_alpha,
            bpreg=self.param.bpreg,
            logq=self.param.logq,
            elu_param=self.param.elu_param
        )

    def train(self, training_df: pd.DataFrame, save_dir: str = '', id: int = 0, verbose: int = 1):
        # This method now accepts a pre-loaded pandas DataFrame directly.
        if verbose > 0:
            print(f"GRU4RecWrapper: Training with a DataFrame of {len(training_df)} rows for GRU4RecOriginal's fit method...")

        # The 'fit' method contains its own training loop.
        # Get the global itemidmap from params and pass it to fit
        itemidmap = getattr(self.param, 'itemidmap', None)
        self.model.fit(training_df, itemidmap=itemidmap, verbose=verbose)

        if save_dir and len(save_dir) > 0:
            model_path = os.path.join(save_dir, f'model{id}.pth')
            if verbose > 0:
                print(f"Saving GRU4Rec model to {model_path}")
            self.model.savemodel(model_path)

    def test(self, test_df: pd.DataFrame, verbose: int = 1):
        """
        Evaluates the trained GRU4Rec model by calling the main evaluation
        function with itself in a list.
        """
        # The evaluation function now expects a list of models.
        return evaluate_sessions_and_print([self], test_df, self.param, verbose=verbose)




@torch.no_grad()
def evaluate_sessions_and_print(gru_list, test_df: pd.DataFrame, params, cutoff=10, batch_size=512, mode='conservative', item_key='ItemId', session_key='SessionId', time_key='Time', verbose=1):
    """
    Evaluates one or more GRU4Rec models on a given test dataset.
    If multiple models are provided, it evaluates them as an ensemble by averaging their prediction scores.
    """
    if not gru_list:
        if verbose > 0:
            print("Warning: No models to evaluate.")
        return 0.0, 0.0, 0.0

    # Assume all models share the same itemidmap, get it from the first one
    first_model_obj = gru_list[0]
    itemidmap = first_model_obj.model.data_iterator.itemidmap if hasattr(first_model_obj.model, 'data_iterator') else None

    # metrics
    recall = 0.0
    mrr = 0.0
    ndcg = 0.0
    n = 0
    
    data_iterator = SessionDataIterator(test_df, batch_size, 0, 0, 0, item_key, session_key, time_key, device=first_model_obj.device, itemidmap=itemidmap, verbose=verbose)
    
    # Create hidden states for each model
    hidden_states = {i: [torch.zeros((batch_size, layer_size), device=m.device) for layer_size in m.model.layers] for i, m in enumerate(gru_list)}

    for in_idxs, out_idxs in data_iterator(enable_neg_samples=False):
        current_batch_size = in_idxs.shape[0]
        ensembled_scores = []
        
        for i, model_wrapper in enumerate(gru_list):
            H = hidden_states[i]
            
            # Ensure hidden state batch size matches current input batch size
            if H[0].shape[0] != current_batch_size:
                H = [torch.zeros((current_batch_size, layer_size), device=model_wrapper.device) for layer_size in model_wrapper.model.layers]
                hidden_states[i] = H

            for h_layer in H: h_layer.detach_()
            
            # Forward pass for one model
            # model_wrapper is GRU4RecWrapper, .model is GRU4RecOriginal, .model.model is GRU4RecModel
            O = model_wrapper.model.model.forward(in_idxs, H, None, training=False)
            ensembled_scores.append(O.T)

        # Average the scores from all models
        final_scores = torch.stack(ensembled_scores).mean(dim=0)
        
        tscores = torch.diag(final_scores[out_idxs])
        
        if mode == 'standard': ranks = (final_scores > tscores).sum(dim=0) + 1
        elif mode == 'conservative': ranks = (final_scores >= tscores).sum(dim=0)
        elif mode == 'median':  ranks = (final_scores > tscores).sum(dim=0) + 0.5*((final_scores == tscores).sum(dim=0) - 1) + 1
        else: raise NotImplementedError

        hits_mask = (ranks <= cutoff)
        recall += hits_mask.sum().cpu().numpy()
        
        hit_ranks = ranks[hits_mask]
        mrr += (1 / hit_ranks.float()).sum().cpu().numpy()
        ndcg += (1 / torch.log2(hit_ranks.float() + 1)).sum().cpu().numpy()
        n += final_scores.shape[1]

    if n > 0:
        recall /= n
        mrr /= n
        ndcg /= n

    # Print with a note if it's an ensembled result
    eval_type = "Ensembled" if len(gru_list) > 1 else "Single Model"
    print(f'Test ({eval_type}) - Recall@{cutoff}: {recall:.4f}, MRR@{cutoff}: {mrr:.4f}, NDCG@{cutoff}: {ndcg:.4f}')

    return recall, mrr, ndcg


from torch.optim import Optimizer
class IndexedAdagradM(Optimizer):

    def __init__(self, params, lr=0.05, momentum=0.0, eps=1e-6):
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if eps <= 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))

        defaults = dict(lr=lr, momentum=momentum, eps=eps)
        super(IndexedAdagradM, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['acc'] = torch.full_like(p, 0, memory_format=torch.preserve_format)
                if momentum > 0: state['mom'] = torch.full_like(p, 0, memory_format=torch.preserve_format)

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['acc'].share_memory_()
                if group['momentum'] > 0: state['mom'].share_memory_()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                clr = group['lr']
                momentum = group['momentum']
                if grad.is_sparse:
                    grad = grad.coalesce()
                    grad_indices = grad._indices()[0]
                    grad_values = grad._values()
                    accs = state['acc'][grad_indices] + grad_values.pow(2)
                    state['acc'].index_copy_(0, grad_indices, accs)
                    accs.add_(group['eps']).sqrt_().mul_(-1/clr)
                    if momentum > 0:
                        moma = state['mom'][grad_indices]
                        moma.mul_(momentum).add_(grad_values / accs)
                        state['mom'].index_copy_(0, grad_indices, moma)
                        p.index_add_(0, grad_indices, moma)
                    else:
                        p.index_add_(0, grad_indices, grad_values / accs)
                else:
                    state['acc'].add_(grad.pow(2))
                    accs = state['acc'].add(group['eps'])
                    accs.sqrt_()
                    if momentum > 0:
                        mom = state['mom']
                        mom.mul_(momentum).addcdiv_(grad, accs, value=-clr)
                        p.add_(mom)
                    else:
                        p.addcdiv_(grad, accs, value=-clr)
        return loss

def init_parameter_matrix(tensor: torch.Tensor, dim0_scale: int = 1, dim1_scale: int = 1):
    sigma = math.sqrt(6.0 / float(tensor.size(0) / dim0_scale + tensor.size(1) / dim1_scale))
    return nn.init._no_grad_uniform_(tensor, -sigma, sigma)

class GRUEmbedding(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(GRUEmbedding, self).__init__()
        self.Wx0 = nn.Embedding(dim_in, dim_out * 3, sparse=True)
        self.Wrz0 = nn.Parameter(torch.empty((dim_out, dim_out * 2), dtype=torch.float))
        self.Wh0 = nn.Parameter(torch.empty((dim_out, dim_out * 1), dtype=torch.float))
        self.Bh0 = nn.Parameter(torch.zeros(dim_out * 3, dtype=torch.float))
        self.reset_parameters()
    def reset_parameters(self):
        init_parameter_matrix(self.Wx0.weight, dim1_scale = 3)
        init_parameter_matrix(self.Wrz0, dim1_scale = 2)
        init_parameter_matrix(self.Wh0, dim1_scale = 1)
        nn.init.zeros_(self.Bh0)
    def forward(self, X, H):
        Vx = self.Wx0(X) + self.Bh0
        Vrz = torch.mm(H, self.Wrz0)
        vx_x, vx_r, vx_z = Vx.chunk(3, 1)
        vh_r, vh_z = Vrz.chunk(2, 1)
        r = torch.sigmoid(vx_r + vh_r)
        z = torch.sigmoid(vx_z + vh_z)
        h = torch.tanh(torch.mm(r * H, self.Wh0) + vx_x)
        h = (1.0 - z) * H + z * h
        return h

class GRU4RecModel(nn.Module):
    def __init__(self, n_items, layers=[100], dropout_p_embed=0.0, dropout_p_hidden=0.0, embedding=0, constrained_embedding=True):
        super(GRU4RecModel, self).__init__()
        self.n_items = n_items
        self.layers = layers
        self.dropout_p_embed = dropout_p_embed
        self.dropout_p_hidden = dropout_p_hidden
        self.embedding = embedding
        self.constrained_embedding = constrained_embedding
        self.start = 0
        if constrained_embedding:
            n_input = layers[-1]
        elif embedding:
            self.E = nn.Embedding(n_items, embedding, sparse=True)
            n_input = embedding
        else:
            self.GE = GRUEmbedding(n_items, layers[0])
            n_input = n_items
            self.start = 1
        self.DE = nn.Dropout(dropout_p_embed)
        self.G = []
        self.D = []
        for i in range(self.start, len(layers)):
            self.G.append(nn.GRUCell(layers[i-1] if i > 0 else n_input, layers[i]))
            self.D.append(nn.Dropout(dropout_p_hidden))
        self.G = nn.ModuleList(self.G)
        self.D = nn.ModuleList(self.D)
        self.Wy = nn.Embedding(n_items, layers[-1], sparse=True)
        self.By = nn.Embedding(n_items, 1, sparse=True)
        self.reset_parameters()
    @torch.no_grad()
    def reset_parameters(self):
        if self.embedding:
            init_parameter_matrix(self.E.weight)
        elif not self.constrained_embedding:
            self.GE.reset_parameters()
        for i in range(len(self.G)):
            init_parameter_matrix(self.G[i].weight_ih, dim1_scale = 3)
            init_parameter_matrix(self.G[i].weight_hh, dim1_scale = 3)
            nn.init.zeros_(self.G[i].bias_ih)
            nn.init.zeros_(self.G[i].bias_hh)
        init_parameter_matrix(self.Wy.weight)
        nn.init.zeros_(self.By.weight)
    def _init_numpy_weights(self, shape):
        sigma = np.sqrt(6.0 / (shape[0] + shape[1]))
        m = (np.random.rand(*shape) * 2 * sigma - sigma).astype('float32')
        return m
    @torch.no_grad()
    def _reset_weights_to_compatibility_mode(self):
        np.random.seed(42)
        if self.constrained_embedding:
            n_input = self.layers[-1]
        elif self.embedding:
            n_input = self.embedding
            self.E.weight.set_(torch.tensor(self._init_numpy_weights((self.n_items, n_input)), device=self.E.weight.device))
        else:
            n_input = self.n_items
            m = []
            m.append(self._init_numpy_weights((n_input, self.layers[0])))
            m.append(self._init_numpy_weights((n_input, self.layers[0])))
            m.append(self._init_numpy_weights((n_input, self.layers[0])))
            self.GE.Wx0.weight.set_(torch.tensor(np.hstack(m), device=self.GE.Wx0.weight.device))
            m2 = []
            m2.append(self._init_numpy_weights((self.layers[0] , self.layers[0])))
            m2.append(self._init_numpy_weights((self.layers[0] , self.layers[0])))
            self.GE.Wrz0.set_(torch.tensor(np.hstack(m2), device=self.GE.Wrz0.device))
            self.GE.Wh0.set_(torch.tensor(self._init_numpy_weights((self.layers[0] , self.layers[0])), device=self.GE.Wh0.device))
            self.GE.Bh0.set_(torch.zeros((self.layers[0]*3,), device=self.GE.Bh0.device))
        for i in range(self.start, len(self.layers)):
            m = []
            m.append(self._init_numpy_weights((n_input, self.layers[i])))
            m.append(self._init_numpy_weights((n_input, self.layers[i])))
            m.append(self._init_numpy_weights((n_input, self.layers[i])))
            self.G[i].weight_ih.set_(torch.tensor(np.vstack(m), device=self.G[i].weight_ih.device))
            m2 = []
            m2.append(self._init_numpy_weights((self.layers[i] , self.layers[i])))
            m2.append(self._init_numpy_weights((self.layers[i] , self.layers[i])))
            m2.append(self._init_numpy_weights((self.layers[i] , self.layers[i])))
            self.G[i].weight_hh.set_(torch.tensor(np.vstack(m2), device=self.G[i].weight_hh.device))
            self.G[i].bias_hh.set_(torch.zeros((self.layers[i]*3,), device=self.G[i].bias_hh.device))
            self.G[i].bias_ih.set_(torch.zeros((self.layers[i]*3,), device=self.G[i].bias_ih.device))
        self.Wy.weight.set_(torch.tensor(self._init_numpy_weights((self.n_items, self.layers[-1])), device=self.Wy.weight.device))
        self.By.weight.set_(torch.zeros((self.n_items, 1), device=self.By.weight.device))
    def embed_constrained(self, X, Y=None):
        if Y is not None:
            XY = torch.cat([X, Y])
            EXY = self.Wy(XY)
            split = X.shape[0]
            E = EXY[:split]
            O = EXY[split:]
            B = self.By(Y)
        else:
            E = self.Wy(X)
            O = self.Wy.weight
            B = self.By.weight
        return E, O, B
    def embed_separate(self, X, Y=None):
        E = self.E(X)
        if Y is not None:
            O = self.Wy(Y)
            B = self.By(Y)
        else:
            O = self.Wy.weight
            B = self.By.weight
        return E, O, B
    def embed_gru(self, X, H, Y=None):
        E = self.GE(X, H)
        if Y is not None:
            O = self.Wy(Y)
            B = self.By(Y)
        else:
            O = self.Wy.weight
            B = self.By.weight
        return E, O, B
    def embed(self, X, H, Y=None):
        if self.constrained_embedding:
            E, O, B = self.embed_constrained(X, Y)
        elif self.embedding > 0:
            E, O, B = self.embed_separate(X, Y)
        else:
            E, O, B = self.embed_gru(X, H[0], Y)
        return E, O, B
    def hidden_step(self, X, H, training=False):
        for i in range(self.start, len(self.layers)):
            X = self.G[i](X, Variable(H[i]))
            if training:
                X = self.D[i](X)
            H[i] = X
        return X
    def score_items(self, X, O, B):
        O = torch.mm(X, O.T) + B.T
        return O
    def forward(self, X, H, Y, training=False):
        E, O, B = self.embed(X, H, Y)
        if training: 
            E = self.DE(E)
        if not (self.constrained_embedding or self.embedding):
            H[0] = E
        Xh = self.hidden_step(E, H, training=training)
        R = self.score_items(Xh, O, B)
        return R

class SampleCache:
    def __init__(self, n_sample, sample_cache_max_size, distr, device=torch.device('cuda:0'), verbose=1):
        self.device = device
        self.n_sample = n_sample
        self.generate_length = sample_cache_max_size // n_sample if n_sample > 0 else 0
        self.distr = distr
        self._refresh()
        if verbose > 0:
            print('Created sample store with {} batches of samples (type=GPU)'.format(self.generate_length))
    def _bin_search(self, arr, x):
        l = x.shape[0]
        a = torch.zeros(l, dtype=torch.int64, device=self.device)
        b = torch.zeros(l, dtype=torch.int64, device=self.device) + arr.shape[0]
        while torch.any(a != b):
            ab = torch.div((a + b), 2,  rounding_mode='trunc')
            val = arr[ab]
            amask = (val <= x)
            a[amask] = ab[amask] + 1
            b[~amask] = ab[~amask]
        return a
    def _refresh(self):
        if self.n_sample <= 0: return
        x = torch.rand(self.generate_length * self.n_sample, dtype=torch.float32, device=self.device)
        self.neg_samples = self._bin_search(self.distr, x).reshape((self.generate_length, self.n_sample))
        self.sample_pointer = 0
    def get_sample(self):
        if self.sample_pointer >= self.generate_length:
            self._refresh()
        sample = self.neg_samples[self.sample_pointer]
        self.sample_pointer += 1
        return sample

class SessionDataIterator:
    def __init__(self, data, batch_size, n_sample=0, sample_alpha=0.75, sample_cache_max_size=10000000, item_key='ItemId', session_key='SessionId', time_key='Time', session_order='time', device=torch.device('cuda:0'), itemidmap=None, verbose=1):
        self.device = device
        self.batch_size = batch_size
        self.verbose = verbose
        if itemidmap is None:
            itemids = data[item_key].unique()
            self.n_items = len(itemids)
            self.itemidmap = pd.Series(data=np.arange(self.n_items, dtype='int32'), index=itemids, name='ItemIdx')
        else:
            if self.verbose > 0:
                print('Using existing item ID map')
            self.itemidmap = itemidmap
            self.n_items = len(itemidmap)
            in_mask = data[item_key].isin(itemidmap.index.values)
            n_not_in = (~in_mask).sum()
            if n_not_in > 0:
                #print('{} rows of the data contain unknown items and will be filtered'.format(n_not_in))
                data = data.drop(data.index[~in_mask])
        self.sort_if_needed(data, [session_key, time_key])
        self.offset_sessions = self.compute_offset(data, session_key)
        
        # Add safeguard: Cap batch_size if it's larger than the number of sessions
        n_sessions = len(self.offset_sessions) - 1
        if self.batch_size > n_sessions:
            if self.verbose > 0:
                print(f"Warning: batch_size ({self.batch_size}) is larger than the number of sessions ({n_sessions}).")
                print(f"Setting batch_size to {n_sessions}.")
            self.batch_size = n_sessions

        if session_order == 'time':
            self.session_idx_arr = np.argsort(data.groupby(session_key)[time_key].min().values)
        else:
            self.session_idx_arr = np.arange(len(self.offset_sessions) - 1)
        self.data_items = self.itemidmap[data[item_key].values].values
        if n_sample > 0:
            pop = data.groupby(item_key).size()
            # Reindex pop to match the global itemidmap, filling missing items with 0 popularity
            pop = pop.reindex(self.itemidmap.index, fill_value=0)
            pop = pop.values**sample_alpha
            pop = pop.cumsum() / pop.sum()
            pop[-1] = 1
            distr = torch.tensor(pop, device=self.device, dtype=torch.float32)
            self.sample_cache = SampleCache(n_sample, sample_cache_max_size, distr, device=self.device, verbose=self.verbose)

    def sort_if_needed(self, data, columns, any_order_first_dim=False):
        is_sorted = True
        neq_masks = []
        for i, col in enumerate(columns):
            dcol = data[col]
            neq_masks.append(dcol.values[1:]!=dcol.values[:-1])
            if i == 0:
                if any_order_first_dim:
                    is_sorted = is_sorted and (dcol.nunique() == neq_masks[0].sum() + 1)
                else:
                    is_sorted = is_sorted and np.all(dcol.values[1:] >= dcol.values[:-1])
            else:
                is_sorted = is_sorted and np.all(neq_masks[i - 1] | (dcol.values[1:] >= dcol.values[:-1]))
            if not is_sorted:
                break
        if is_sorted:
            if self.verbose > 0:
                print('The dataframe is already sorted by {}'.format(', '.join(columns)))
        else:
            if self.verbose > 0:
                print('The dataframe is not sorted by {}, sorting now'.format(col))
            t0 = time.time()
            data.sort_values(columns, inplace=True)
            t1 = time.time()
            if self.verbose > 0:
                print('Data is sorted in {:.2f}'.format(t1 - t0))

    def compute_offset(self, data, column):
        offset = np.zeros(data[column].nunique() + 1, dtype=np.int32)
        offset[1:] = data.groupby(column).size().cumsum()
        return offset

    def __call__(self, enable_neg_samples, reset_hook=None):
        batch_size = self.batch_size
        iters = np.arange(batch_size)
        maxiter = iters.max()
        start = self.offset_sessions[self.session_idx_arr[iters]]
        end = self.offset_sessions[self.session_idx_arr[iters]+1]
        finished = False
        valid_mask = np.ones(batch_size, dtype='bool')
        n_valid = self.batch_size
        while not finished:
            minlen = (end-start).min()
            out_idx = torch.tensor(self.data_items[start], requires_grad=False, device=self.device)
            for i in range(minlen-1):
                in_idx = out_idx
                out_idx = torch.tensor(self.data_items[start+i+1], requires_grad=False, device=self.device)
                if enable_neg_samples:
                    sample = self.sample_cache.get_sample()
                    y = torch.cat([out_idx, sample])
                else:
                    y = out_idx
                yield in_idx, y
            start = start+minlen-1
            finished_mask = (end-start<=1)
            n_finished = finished_mask.sum()
            iters[finished_mask] = maxiter + np.arange(1,n_finished+1)
            maxiter += n_finished
            valid_mask = (iters < len(self.offset_sessions)-1)
            n_valid = valid_mask.sum()
            if n_valid == 0:
                finished = True
                break
            mask = finished_mask & valid_mask
            sessions = self.session_idx_arr[iters[mask]]
            start[mask] = self.offset_sessions[sessions]
            end[mask] = self.offset_sessions[sessions+1]
            iters = iters[valid_mask]
            start = start[valid_mask]
            end = end[valid_mask]
            if reset_hook is not None:
                finished = reset_hook(n_valid, finished_mask, valid_mask)

class GRU4RecOriginal:
    def __init__(self, layers=[100], loss='cross-entropy', batch_size=64, dropout_p_embed=0.0,
                 dropout_p_hidden=0.0, learning_rate=0.05, momentum=0.0, sample_alpha=0.5, n_sample=2048, embedding=0,
                 constrained_embedding=True, n_epochs=10, bpreg=1.0, elu_param=0.5, logq=0.0, device=torch.device('cuda:0')):
        self.device = device
        self.layers = layers
        self.loss = loss
        self.set_loss_function(loss)
        self.elu_param = elu_param
        self.bpreg = bpreg
        self.logq = logq
        self.batch_size = batch_size
        self.dropout_p_embed = dropout_p_embed
        self.dropout_p_hidden = dropout_p_hidden
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.sample_alpha = sample_alpha
        self.n_sample = n_sample
        if embedding == 'layersize':
            self.embedding = self.layers[0]
        else:
            self.embedding = embedding
        self.constrained_embedding = constrained_embedding
        self.n_epochs = n_epochs
    def set_loss_function(self, loss):
        if loss == 'cross-entropy': self.loss_function = self.xe_loss_with_softmax
        elif loss == 'bpr-max': self.loss_function = self.bpr_max_loss_with_elu
        else: raise NotImplementedError
    def set_params(self, **kvargs):
        maxk_len = np.max([len(str(x)) for x in kvargs.keys()])
        maxv_len = np.max([len(str(x)) for x in kvargs.values()])
        for k,v in kvargs.items():
            if not hasattr(self, k):
                print('Unkown attribute: {}'.format(k))
                raise NotImplementedError
            else:
                if type(v) == str and type(getattr(self, k)) == list: v = [int(l) for l in v.split('/')]
                if type(v) == str and type(getattr(self, k)) == bool:
                    if v == 'True' or v == '1': v = True
                    elif v == 'False' or v == '0': v = False
                    else:
                        print('Invalid value for boolean parameter: {}'.format(v))
                        raise NotImplementedError
                if k == 'embedding' and v == 'layersize':
                    self.embedding = 'layersize'
                setattr(self, k, type(getattr(self, k))(v))
                if k == 'loss': self.set_loss_function(self.loss)
                print('SET   {}{}TO   {}{}(type: {})'.format(k, ' '*(maxk_len-len(k)+3), getattr(self, k), ' '*(maxv_len-len(str(getattr(self, k)))+3), type(getattr(self, k))))
        if self.embedding == 'layersize':
            self.embedding = self.layers[0]
            print('SET   {}{}TO   {}{}(type: {})'.format('embedding', ' '*(maxk_len-len('embedding')+3), getattr(self, 'embedding'), ' '*(maxv_len-len(str(getattr(self, 'embedding')))+3), type(getattr(self, 'embedding'))))
    def xe_loss_with_softmax(self, O, Y, M):
        if self.logq > 0:
            O = O - self.logq * torch.log(torch.cat([self.P0[Y[:M]], self.P0[Y[M:]]**self.sample_alpha]))
        X = torch.exp(O - O.max(dim=1, keepdim=True)[0])
        X = X / X.sum(dim=1, keepdim=True)
        return -torch.sum(torch.log(torch.diag(X)+1e-24))
    def softmax_neg(self, X):
        hm = 1.0 - torch.eye(*X.shape, out=torch.empty_like(X))
        X = X * hm
        e_x = torch.exp(X - X.max(dim=1, keepdim=True)[0]) * hm
        return e_x / e_x.sum(dim=1, keepdim=True)
    def bpr_max_loss_with_elu(self, O, Y, M):
        if self.elu_param > 0:
            O = nn.functional.elu(O, self.elu_param)
        softmax_scores = self.softmax_neg(O)
        target_scores = torch.diag(O)
        target_scores = target_scores.reshape(target_scores.shape[0],-1)
        return torch.sum((-torch.log(torch.sum(torch.sigmoid(target_scores-O)*softmax_scores, dim=1)+1e-24)+self.bpreg*torch.sum((O**2)*softmax_scores, dim=1)))
    
    def fit(self, data, sample_cache_max_size=10000000, compatibility_mode=True, item_key='ItemId', session_key='SessionId', time_key='Time', itemidmap=None, verbose=1):
        self.error_during_train = False
        self.data_iterator = SessionDataIterator(data, self.batch_size, n_sample=self.n_sample, sample_alpha=self.sample_alpha, sample_cache_max_size=sample_cache_max_size, item_key=item_key, session_key=session_key, time_key=time_key, session_order='time', device=self.device, itemidmap=itemidmap, verbose=verbose)
        if self.logq and self.loss == 'cross-entropy':
            pop = data.groupby(item_key).size()
            # Reindex pop to match the global itemidmap, filling missing items with 0 popularity
            pop = pop.reindex(self.data_iterator.itemidmap.index, fill_value=0)
            self.P0 = torch.tensor(pop.values, dtype=torch.float32, device=self.device)
        model = GRU4RecModel(self.data_iterator.n_items, self.layers, self.dropout_p_embed, self.dropout_p_hidden, self.embedding, self.constrained_embedding).to(self.device)
        if compatibility_mode: 
            model._reset_weights_to_compatibility_mode()
        self.model = model
        opt = IndexedAdagradM(self.model.parameters(), self.learning_rate, self.momentum)
        for epoch in range(self.n_epochs):
            t0 = time.time()
            H = []
            for i in range(len(self.layers)):
                H.append(torch.zeros((self.batch_size, self.layers[i]), dtype=torch.float32, requires_grad=False, device=self.device))
            c = []
            cc = []
            n_valid = self.batch_size
            reset_hook = lambda n_valid, finished_mask, valid_mask: self._adjust_hidden(n_valid, finished_mask, valid_mask, H)
            for in_idx, out_idx in self.data_iterator(enable_neg_samples=(self.n_sample>0), reset_hook=reset_hook):
                for h in H: h.detach_()
                self.model.zero_grad()
                R = self.model.forward(in_idx, H, out_idx, training=True)
                L = self.loss_function(R, out_idx, n_valid) / self.batch_size
                L.backward()
                opt.step()
                L = L.cpu().detach().numpy()
                c.append(L)
                cc.append(n_valid)
                if np.isnan(L):
                    print(str(epoch) + ': NaN error!')
                    self.error_during_train = True
                    return
            c = np.array(c)
            cc = np.array(cc)
            avgc = np.sum(c * cc) / np.sum(cc)
            if np.isnan(avgc):
                print('Epoch {}: NaN error!'.format(str(epoch)))
                self.error_during_train = True
                return
            t1 = time.time()
            dt = t1 - t0
            if verbose > 0:
                print('Epoch{} --> loss: {:.6f} \t({:.2f}s) \t[{:.2f} mb/s | {:.0f} e/s]'.format(epoch+1, avgc, dt, len(c)/dt, np.sum(cc)/dt))
            
    def _adjust_hidden(self, n_valid, finished_mask, valid_mask, H):
        if (self.n_sample == 0) and (n_valid < 2):
            return True
        with torch.no_grad():
            for i in range(len(self.layers)):
                H[i][finished_mask] = 0
        if n_valid < len(valid_mask):
            for i in range(len(H)):
                H[i] = H[i][valid_mask]
        return False
    def to(self, device):
        if type(device) == str:
            device = torch.device(device)
        if device == self.device:
            return
        if hasattr(self, 'model'):
            self.model = self.model.to(device)
            self.model.eval()
        self.device = device
        if hasattr(self, 'data_iterator'):
            self.data_iterator.device = device
            if hasattr(self.data_iterator, 'sample_cache'):
                self.data_iterator.sample_cache.device = device
        pass
    def savemodel(self, path):
        torch.save(self, path)
    @classmethod
    def loadmodel(cls, path, device='cuda:0'):
        gru = torch.load(path, map_location=device, weights_only=False)
        gru.device = torch.device(device)
        if hasattr(gru, 'data_iterator'):
            gru.data_iterator.device = torch.device(device)
            if hasattr(gru.data_iterator, 'sample_cache'):
                gru.data_iterator.sample_cache.device = torch.device(device)
                gru.model.eval()
                return gru
        
# Attention-based Aggregator for RecEraser
class RecEraserAggregator(nn.Module):
    def __init__(self, emb_dim, num_shards, att_dim=64):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_shards = num_shards
        self.att_dim = att_dim

        # user attention
        self.WA = nn.Linear(emb_dim, att_dim)
        self.HA = nn.Linear(att_dim, 1)

    def forward(self, shard_scores):
        """
        shard_scores: [num_shards, batch, n_items] -> we want to aggregate them
        """
        # transpose to [batch, num_shards, n_items]
        scores = shard_scores.permute(1, 0, 2)
        
        # We treat the score vector as the "embedding" for the attention mechanism
        x = torch.relu(self.WA(scores))
        att_score = self.HA(x)
        weights = torch.softmax(att_score, dim=1)
        agg_scores = torch.sum(weights * scores, dim=1)
        
        return agg_scores

# Sharding Function for Sequential Methods (SISA, RecEraser)
def shard_data_sequential(train_df, args):
    """
    Partitions the training data into shards using the KMeans for Interaction
    Grouped two interactions together to ensure that each shard has enough data for training and deletion
    """
    print(f"Sharding data into {args.group} groups using {args.learn} ({args.level} level)...")
    
    # 1. Create Units
    units = []
    sorted_df = train_df.sort_values(['SessionId', 'Time'])
    for sid, group in sorted_df.groupby('SessionId'):
        indices = group.index.tolist()
        # Pair interactions: (1,2), (3,4)...
        for i in range(0, len(indices), 2):
            if i + 3 == len(indices):
                units.append(indices[i:])
                break
            elif i + 1 < len(indices):
                units.append(indices[i:i+2])
            else:
                units.append(indices[i:])
                
    shard_indices = [[] for _ in range(args.group)]

    if args.learn == 'sisa':
        if args.level == 'user':
            # Shard by SessionId
            unique_sids = train_df['SessionId'].unique()
            np.random.shuffle(unique_sids)
            sid_to_shard = {sid: i % args.group for i, sid in enumerate(unique_sids)}
            for unit in units:
                sid = train_df.loc[unit[0], 'SessionId']
                shard_indices[sid_to_shard[sid]].extend(unit)
        else:
            # Shard by Units (Interaction level)
            np.random.shuffle(units)
            for i, unit in enumerate(units):
                shard_indices[i % args.group].extend(unit)

    elif args.learn == 'receraser':
        # Balanced Partition based on Unit Embeddings (InBP logic) using Kmeans-InBP
        print("Using kmeans_InBP for RecEraser partitioning...")
        
        # Check available embeddings to avoid KeyError
        base_dataset = args.dataset.replace("-seq", "")
        item_emb_path = f'results/item_emb/{base_dataset}_neumf_item_emb.npy'
        if not os.path.exists(item_emb_path):
            item_emb_path = 'results/item_emb/ml-1m_neumf_item_emb.npy'
        item_embs = np.load(item_emb_path, allow_pickle=True).item()
        
        # Create a proxy DataFrame where each row is a Unit
        proxy_rows = []
        unshardable_units = []
        for i, unit in enumerate(units):
            sid = train_df.loc[unit[0], 'SessionId']
            rep_item = train_df.loc[unit[0], 'ItemId']
            
            # Adjust SessionId from 1-based to 0-based to match pre-trained embeddings
            sid_adjusted = sid - 1
            
            # Ensure the item exists in the pre-trained embeddings
            if rep_item in item_embs:
                proxy_rows.append({'user': sid_adjusted, 'item': rep_item, 'unit_idx': i})
            else:
                unshardable_units.append(i)
            
        if proxy_rows:
            proxy_df = pd.DataFrame(proxy_rows)
            proxy_df['rating'] = 1
            
            # Dummy test set for KMeans-InBP
            dummy_test = proxy_df.head(1).copy()
            
            train_groups, _ = kmeans_InBP(proxy_df, dummy_test, base_dataset, args.group, "neumf")
            
            for j in range(args.group):
                shard_proxy = train_groups[j]
                for _, row in shard_proxy.iterrows():
                    unit_idx = int(row['unit_idx'])
                    shard_indices[j].extend(units[unit_idx])
        
        # Distribute unshardable units (missing embeddings) via round-robin
        for i, unit_idx in enumerate(unshardable_units):
            shard_indices[i % args.group].extend(units[unit_idx])

    shards = []
    for i in range(args.group):
        shards.append(train_df.loc[shard_indices[i]].reset_index(drop=True))
        
    return shards

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    # Standard Benchmark Arguments
    parser.add_argument('--dataset', type=str, default='ml-1m-seq', help='dataset name')
    parser.add_argument('--learn', type=str, default='retrain', help='method: retrain, sisa, receraser, ultrare')
    parser.add_argument('--delper', type=int, default=5, help='deletion percentage')
    parser.add_argument('--deltype', type=str, default='random', help='selection: random, core, edge')
    parser.add_argument('--level', type=str, default='user', help='level: user, interaction')
    parser.add_argument('--group', type=int, default=10, help='number of groups/shards')
    parser.add_argument('--epoch', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--verbose', type=int, default=1, help='verbosity level')

    # GRU4Rec Specific Hyperparameters (Defaults)
    parser.add_argument('--layers', nargs='+', type=int, default=[100], help='GRU layers size')
    parser.add_argument('--loss', type=str, default='cross-entropy', help='loss function: cross-entropy, bpr-max')
    parser.add_argument('--dropout_p_embed', type=float, default=0.0, help='embedding dropout')
    parser.add_argument('--dropout_p_hidden', type=float, default=0.0, help='hidden layer dropout')
    parser.add_argument('--momentum', type=float, default=0.0, help='momentum for optimizer')
    parser.add_argument('--n_sample', type=int, default=2048, help='negative sampling size')
    parser.add_argument('--sample_alpha', type=float, default=0.75, help='sampling exponent')
    parser.add_argument('--bpreg', type=float, default=1.0, help='BPR regularization')
    parser.add_argument('--logq', type=float, default=0.0, help='LogQ correction')
    parser.add_argument('--elu_param', type=float, default=0.5, help='ELU parameter')

    args = parser.parse_args()
    
    print(f"Starting experiment: {args.learn} on {args.dataset} at {args.level} level")

    # Path to data files
    train_path = os.path.join('data', args.dataset, 'train.csv')
    test_path = os.path.join('data', args.dataset, 'test.csv')

    # Load initial data to build global item map
    raw_train = pd.read_csv(train_path, sep=',')
    raw_test = pd.read_csv(test_path, sep=',')
    all_items = pd.concat([raw_train['iid'], raw_test['iid']]).unique()
    itemidmap = pd.Series(data=np.arange(len(all_items)), index=all_items, name='ItemIdx')
    args.itemidmap = itemidmap

    if args.learn == 'retrain':
        # Handle deletion based on level
        if args.level == 'interaction':
            train_df, affected_users = read_session_data(train_path, del_per=args.delper, del_type=args.deltype)
        else:
            # User level deletion
            unique_users = raw_train['uid'].unique()
            num_del = int(len(unique_users) * args.delper / 100)
            del_users = list(np.random.choice(unique_users, num_del, replace=False))
            train_df, affected_users = read_session_data(train_path, del_users=del_users)
        
        test_df, _ = read_session_data(test_path)
        
        gru = GRU4RecWrapper(args)
        gru.train(train_df, verbose=args.verbose)
        gru.test(test_df, verbose=args.verbose)

    elif args.learn in ['sisa', 'receraser']:
        # Handle deletion first
        if args.level == 'interaction':
            reduced_train, _ = read_session_data(train_path, del_per=args.delper, del_type=args.deltype)
        else:
            unique_users = raw_train['uid'].unique()
            num_del = int(len(unique_users) * args.delper / 100)
            del_users = list(np.random.choice(unique_users, num_del, replace=False))
            reduced_train, _ = read_session_data(train_path, del_users=del_users)
            
        test_df, _ = read_session_data(test_path)
        
        # Perform Sharding
        shards = shard_data_sequential(reduced_train, args)
        
        # Train Shard Models
        gru_list = []
        save_dir = f'results/{args.learn}/{args.dataset}_{args.level}_{args.deltype}_{args.delper}'
        os.makedirs(save_dir, exist_ok=True)
        
        for i, shard_df in enumerate(shards):
            print(f"--- Training Shard {i+1}/{args.group} ({len(shard_df)} interactions) ---")
            gru = GRU4RecWrapper(args)
            gru.train(shard_df, save_dir=save_dir, id=i, verbose=args.verbose)
            gru_list.append(gru)
            
        # Ensemble Evaluation
        evaluate_sessions_and_print(gru_list, test_df, args, verbose=args.verbose)

    else:
        raise NotImplementedError(f"Method {args.learn} not supported for GRU4Rec yet.")