import os
import sys

import numpy as np
import time
from torch import nn
from torch import optim
import torch

from utils import NeuMF, seed_all, baseTrain, baseTest
from utils import WMF, DMF, NeuMF, BPR


class Scratch(object):
    def __init__(self, param, model_type):
        # model param
        self.n_user = param.n_user
        self.n_item = param.n_item
        self.k = param.k
        self.model_type = model_type

        # training param
        self.seed = param.seed
        self.lr = param.lr
        self.epochs = param.epochs
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.pos_dir = param.pos_data

        # log
        self.log = {'train_loss': [],
                    'test_rmse': [],
                    'test_ndcg': [],
                    'test_hr': [],
                    'total_rmse': [],
                    'total_ndcg': [],
                    'total_hr': [],
                    'time': []}

        if self.model_type in ['wmf']:
            self.loss_fn = 'point-wise'
        elif self.model_type == 'bpr':
            self.loss_fn = 'pair-wise'

        elif self.model_type in ['dmf', 'neumf']:
            self.layers = param.layers
            self.loss_fn = 'point-wise'
            self.is_rmse = False

    def train(self, train_data, test_data, active_test_data, inactive_test_data, verbose=2, id=0, given_model=''):
        print('Using device:', self.device)
        # seed for reproducibility
        seed_all(self.seed)

        # build model
        if given_model == '':
            if self.model_type == 'wmf':
                model = WMF(self.n_user, self.n_item, self.k).to(self.device)
            elif self.model_type == 'bpr':
                model = BPR(self.n_user, self.n_item, self.k).to(self.device)
            elif self.model_type == 'neumf':
                model = NeuMF(self.n_user, self.n_item, self.k, self.layers).to(self.device)
            elif self.model_type == 'dmf':
                model = DMF(self.n_user, self.n_item, self.k, self.layers).to(self.device)
        else:
            model = given_model.to(self.device)

        # set optimizer

        opt = optim.Adam(model.parameters(), lr=self.lr)

        # main loop
        best_ndcg = 0
        best_hr = 0
        count_dec = 0
        total_time = 0

        pos_dict = np.load(self.pos_dir, allow_pickle=True).item()

        best_user_emb = None
        best_item_emb = None
        best_model_state = None

        for t in range(self.epochs):
            # if verbose == 2:
            print(f'Epoch: [{t + 1:>3d}/{self.epochs:>3d}] --------------------')
            epoch_start = time.time()

            # train
            train_loss = baseTrain(train_data, model, self.loss_fn, opt, self.device, verbose)

            train_time = time.time() - epoch_start

            print(train_time)
            total_time += train_time
            user_mapping = None
            pos_mapping = None

            test_ndcg, test_hr = baseTest(test_data, model, self.loss_fn, self.device, verbose, pos_dict, self.n_item,
                                          10,user_mapping,pos_mapping)


            # print info
            epoch_time = time.strftime('%H:%M:%S', time.gmtime(time.time() - epoch_start))

            # if verbose == 2:
            print('Time:', epoch_time)
            print('train_loss:', train_loss)
            print('test_ndcg:', test_ndcg)
            print('test_hr:', test_hr)

            # save log
            self.log['train_loss'].append(train_loss)
            self.log['test_ndcg'].append(test_ndcg)
            self.log['test_hr'].append(test_hr)
            self.log['time'].append(epoch_time)

            if test_ndcg > best_ndcg:
                count_dec = 0
                best_ndcg = test_ndcg
                best_hr = test_hr

                #best model
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                
                #best model user and item embedding
                if self.model_type == 'neumf':
                    best_user_emb = torch.cat(
                        [model.user_mat_mlp.weight,
                        model.user_mat_mf.weight],
                        dim=1
                    ).detach().cpu().clone()

                    best_item_emb = torch.cat(
                        [model.item_mat_mlp.weight,
                        model.item_mat_mf.weight],
                        dim=1
                    ).detach().cpu().clone()
            else:
                count_dec += 1

            if count_dec > 5:
                break

        if active_test_data is not None:
            active_ndcg, active_hr = baseTest(active_test_data, model, self.loss_fn, self.device, verbose, pos_dict,
                                              self.n_item, 10,user_mapping,pos_mapping)
            print('active_test_ndcg:', active_ndcg)
        else:
            active_ndcg = 0
            active_hr = 0
        inactive_ndcg, inactive_hr = baseTest(inactive_test_data, model, self.loss_fn, self.device, verbose,
                                              pos_dict,
                                              self.n_item,
                                              10,user_mapping,pos_mapping)
        print('inactive_test_ndcg:', inactive_ndcg)
        print('-------best--------')

        # ===== save best model =====
        save_dir = f'results/best_{self.model_type}_{self.n_user}_{self.n_item}'
        os.makedirs(save_dir, exist_ok=True)

        if best_model_state is not None:
            torch.save(best_model_state, os.path.join(save_dir, 'best_model.pth'))
            print("Best model saved.")

        # ===== save best embedding =====
        if self.model_type == 'neumf' and best_user_emb is not None:

            user_dict = {}
            item_dict = {}

            user_np = best_user_emb.numpy()
            item_np = best_item_emb.numpy()

            for uid in range(user_np.shape[0]):
                user_dict[np.int64(uid)] = user_np[uid:uid+1]

            for iid in range(item_np.shape[0]):
                item_dict[np.int64(iid)] = item_np[iid:iid+1]

            np.save(os.path.join(save_dir, 'ml-1m_neumf_best_user_emb.npy'), user_dict)
            np.save(os.path.join(save_dir, 'ml-1m_neumf_best_item_emb.npy'), item_dict)

            print("Best embeddings saved.")

        result = {'time': total_time, 'ndcg': best_ndcg, 'hr': best_hr, 'active_ndcg': active_ndcg,
                  'active_hr': active_hr, 'inactive_ndcg': inactive_ndcg, 'inactive_hr': inactive_hr}

        return model, result
