import os
import pandas as pd
import warnings
from os.path import abspath, join, dirname, exists

import numpy as np
import torch
from scratch import Scratch
from utils import RecEraserAggregator, RecEraserTest_ensemble, SISATest_ensemble, saveObject

from read import RatingData, PairData
from read import loadData, readRating_full, readRating_group

from utils import NeuMF, seed_all, baseTrain, baseTest, train_receraser_aggregator
from utils import WMF, DMF, NeuMF, BPR

#/home/jiajie/Richard_He/CURE4Rec/data
DATA_DIR = abspath(join(dirname(__file__), 'data'))
#/home/jiajie/Richard_He/CURE4Rec/result
SAVE_DIR = abspath(join(dirname(__file__), 'result'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class InsParam(object):
    def __init__(self, dataset='ml-1m', model='wmf', epochs=50, n_worker=24, layers=[64, 32], n_group=10, del_per=5,
                 learn_type='retrain',
                 del_type='random'):
        
        # model param
        self.k = 32  # dimension of embedding
        self.lam = 0.1  # regularization coefficient
        self.layers = layers  # structure of FC layers in DMF   

        # training param
        self.seed = 42
        self.n_worker = n_worker
        self.batch = 256
        self.lr = 0.001
        self.epochs = epochs
        self.n_group = n_group
        self.learn_type = learn_type

        # dataset-varied param
        self.del_rating = []  # 2d array/list [[uid, iid], ...]
        self.dataset = dataset
        self.max_rating = 5
        self.del_per = del_per
        self.del_type = del_type
        self.model = model

        if dataset == 'ml-1m':
            self.train_dir = DATA_DIR + '/ml-1m/train.csv'
            self.test_dir = DATA_DIR + '/ml-1m/test.csv'
            self.pos_data = DATA_DIR + '/ml-1m/pos_dict.npy' #每个用户交互的正样本

            self.n_user = 6040
            self.n_item = 3706

        elif dataset == 'adm':
            self.train_dir = DATA_DIR + '/adm/train.csv'
            self.test_dir = DATA_DIR + '/adm/test.csv'
            self.pos_data = DATA_DIR + '/adm/pos_dict.npy'

            self.n_user = 1419
            self.n_item = 13583

        elif dataset == 'book':
            self.train_dir = DATA_DIR + '/book/train.csv'
            self.test_dir = DATA_DIR + '/book/test.csv'
            self.pos_data = DATA_DIR + '/book/pos_dict.npy'

            self.n_user = 1194174
            self.n_item = 314287
        else:
            raise NotImplementedError('Dataset not included!')


class Instance(object):
    def __init__(self, param):
        self.param = param
        prefix = '/test/' if self.param.del_type == 'test' else '/' + str(
            self.param.del_per) + '/' + self.param.del_type + '/'
        self.name = prefix + self.param.dataset + '_g_' + str(
            self.param.n_group)  # time.strftime("%Y%m%d_%H%M%S", time.localtime())
        param_dir = SAVE_DIR + self.name
        #/home/jiajie/Richard_He/CURE4Rec/result/0/random/ml-100k_g_0
  
        if exists(param_dir) == False:
            os.makedirs(param_dir)

        # save param
        saveObject(param_dir + '/param', self.param)  # loadObject(dir + '/param')

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            # np.save(param_dir + '/deletion', deletion)  # np.load('deletion.npy')

    def read(self,model_type='wmf'):
        learn_type = self.param.learn_type
        del_type = self.param.del_type
        del_per = self.param.del_per
        group = self.param.n_group
        if learn_type == 'retrain':
            train_rating, test_rating, active_rating, inactive_rating = readRating_full(self.param.train_dir,
                                                                                        self.param.test_dir, 
                                                                                        del_type, del_per)
            return train_rating, test_rating, active_rating, inactive_rating
        else:
            train_rating, test_rating, active_rating, inactive_rating, ensemble_test, ensemble_train = readRating_group(self.param.train_dir,
                                                                                         self.param.test_dir, del_type,
                                                                                         del_per, learn_type, group,
                                                                                         self.param.dataset, model_type)

            return train_rating, test_rating, active_rating, inactive_rating, ensemble_test, ensemble_train

    def runModel(self, model_type='wmf', verbose=2):
        print(self.name, 'begin:')
        # read raw data
        if self.param.learn_type == 'retrain':
            train_rating, test_rating, active_rating, inactive_rating = self.read(model_type)
        else:
            train_rating, test_rating, active_rating, inactive_rating, ensemble_test, ensemble_train = self.read(model_type)

        if self.param.learn_type == 'retrain':
            # load data
            if model_type in ['wmf', 'dmf', 'neumf']:
                #train_traing = [train_ratings['uid'], train_ratings['iid'], train_ratings['val']]
                train_data = loadData(RatingData(train_rating), self.param.batch,
                                      self.param.n_worker,
                                      True)

            elif model_type in ['bpr']:
                train_data = loadData(PairData(train_rating, self.param.pos_data), self.param.batch,
                                      self.param.n_worker,
                                      True)
            

            test_data = loadData(RatingData(test_rating), len(test_rating[0]), self.param.n_worker, False)
            
            active_test_data = loadData(RatingData(active_rating), len(active_rating[0]), self.param.n_worker,
                                        False)
            inactive_test_data = loadData(RatingData(inactive_rating), len(inactive_rating[0]), self.param.n_worker,
                                          False)

            model = Scratch(self.param, model_type)
            model, result = model.train(train_data, test_data, active_test_data, inactive_test_data, verbose,
                                        given_model='')
            result.update({'model': model_type, 'dataset': self.param.dataset, 'deltype': self.param.del_type,
                           'method': self.param.learn_type})
            

            save_dir = f'results/{self.param.learn_type}'
            os.makedirs(save_dir, exist_ok=True)   

            file_name = f'{model_type}_{self.param.dataset}_{self.param.del_type}_{self.param.del_per}.npy'
            save_path = os.path.join(save_dir, file_name)

            np.save(save_path, result)
            print('End of training', self.name)

        elif self.param.learn_type == 'sisa':

            group = self.param.n_group
            shard_model_paths = []

            for i in range(group):

                print(f"Training shard {i+1}/{group}")
                # load data
                if model_type in ['wmf', 'dmf', 'neumf']:
                    train_data = loadData(RatingData(train_rating[i]), self.param.batch,
                                          self.param.n_worker,
                                          True)
                elif model_type in ['bpr']:
                    train_data = loadData(PairData(train_rating[i], self.param.pos_data), self.param.batch,
                                          self.param.n_worker,
                                          True)

                test_data = loadData(RatingData(test_rating[i]), len(test_rating[i][0]), self.param.n_worker, False)
                if len(active_rating[i][0]) > 0:
                    active_test_data = loadData(RatingData(active_rating[i]), len(active_rating[i][0]),
                                                self.param.n_worker,
                                                False)
                else:
                    active_test_data = None
                inactive_test_data = loadData(RatingData(inactive_rating[i]), len(inactive_rating[i][0]),
                                              self.param.n_worker,
                                              False)
                
                model = Scratch(self.param, model_type)
                #model, result = model.train(train_data,test_data,None,None,verbose)
                model, result = model.train(train_data, test_data, active_test_data, inactive_test_data, verbose,
                                            given_model='')

                result.update({'model': model_type, 'dataset': self.param.dataset, 'deltype': self.param.del_type,
                               'method': self.param.learn_type, 'group': i + 1})
                
                save_dir = f'results/{self.param.learn_type}'
                os.makedirs(save_dir, exist_ok=True)   
                file_name = f'group{i + 1}_{self.param.n_group}_{model_type}_{self.param.dataset}_{self.param.del_type}_{self.param.del_per}.pth'
                model_path = os.path.join(save_dir, file_name)

                torch.save(model.state_dict(), model_path)
                shard_model_paths.append(model_path)

                #np.save(save_path, result)
                
                print(f'End of Group {str(i + 1)} / {group} training', self.name)

        

            # ==========================
            # SISA ensemble testing: load all shard models and test on the full test set (not split by group)
            # ==========================

            print("Start ensemble testing...")

            models = []

            for path in shard_model_paths:

                if model_type == 'neumf':
                    m = NeuMF(self.param.n_user,
                            self.param.n_item,
                            self.param.k,
                            self.param.layers)
                elif model_type == 'wmf':
                    m = WMF(
                        self.param.n_user,
                        self.param.n_item,
                        self.param.k
                    )
                else:
                    raise NotImplementedError
                state_dict = torch.load(path, map_location=device, weights_only=True)
                m.load_state_dict(state_dict)

                m.to(device)
                m.eval()

                models.append(m)

            #
            full_test_data = loadData(RatingData(ensemble_test),len(ensemble_test[0]),self.param.n_worker,False)

            pos_dict = np.load(self.param.pos_data,
                            allow_pickle=True).item()

            ndcg, hr = SISATest_ensemble(full_test_data,models,device,pos_dict,self.param.n_item,top_k=10)

            print("Final SISA HR:", hr)
            print("Final SISA NDCG:", ndcg)

        elif self.param.learn_type == 'receraser':

            group = self.param.n_group
            shard_model_paths = []

            for i in range(group):

                print(f"Training shard {i+1}/{group}")
                # load data
                if model_type in ['wmf', 'dmf', 'neumf']:
                    train_data = loadData(RatingData(train_rating[i]), self.param.batch,
                                          self.param.n_worker,
                                          True)
                elif model_type in ['bpr']:
                    train_data = loadData(PairData(train_rating[i], self.param.pos_data), self.param.batch,
                                          self.param.n_worker,
                                          True)

                test_data = loadData(RatingData(test_rating[i]), len(test_rating[i][0]), self.param.n_worker, False)

                if len(active_rating[i][0]) > 0:
                    active_test_data = loadData(RatingData(active_rating[i]), len(active_rating[i][0]),
                                                self.param.n_worker,
                                                False)
                else:
                    active_test_data = None
                
                if len(inactive_rating[i][0]) > 0:
                    inactive_test_data = loadData(RatingData(inactive_rating[i]), len(inactive_rating[i][0]),
                                                  self.param.n_worker,
                                                  False)
                else:
                    inactive_test_data = None
                
                model = Scratch(self.param, model_type)
                #model, result = model.train(train_data,test_data,None,None,verbose)
                model, result = model.train(train_data, test_data, active_test_data, inactive_test_data, verbose,
                                            given_model='')

                result.update({'model': model_type, 'dataset': self.param.dataset, 'deltype': self.param.del_type,
                               'method': self.param.learn_type, 'group': i + 1})
                
                save_dir = f'results/{self.param.learn_type}'
                os.makedirs(save_dir, exist_ok=True)   
                file_name = f'group{i + 1}_{self.param.n_group}_{model_type}_{self.param.dataset}_{self.param.del_type}_{self.param.del_per}.pth'
                model_path = os.path.join(save_dir, file_name)

                torch.save(model.state_dict(), model_path)
                shard_model_paths.append(model_path)

                #np.save(save_path, result)
                
                print(f'End of Group {str(i + 1)} / {group} training', self.name)

        

            # ==========================
            # RecEraser ensemble testing: load all shard models and test on the full test set (not split by group)
            # ==========================
            print("Start RecEraser aggregation...")

            models = []

            for path in shard_model_paths:

                if model_type == 'neumf':
                    m = NeuMF(
                        self.param.n_user,
                        self.param.n_item,
                        self.param.k,
                        self.param.layers
                    )
                elif model_type == 'wmf':
                    m = WMF(
                        self.param.n_user,
                        self.param.n_item,
                        self.param.k
                    )
                else:
                    raise NotImplementedError

                state_dict = torch.load(path, map_location=device, weights_only=True)
                m.load_state_dict(state_dict)

                m.to(device)
                m.eval()

                models.append(m)


            # ========= initialize aggregator =========
            num_shards = len(models)
            if model_type == "neumf":
                emb_dim = (
                    models[0].user_mat_mlp.weight.shape[1] +
                    models[0].user_mat_mf.weight.shape[1]
                )

            elif model_type in ["wmf", "bpr"]:
                emb_dim = models[0].user_mat.weight.shape[1]

            elif model_type == "dmf":
                emb_dim = models[0].layers[-1]   # DMF last layer output as embedding

            else:
                raise NotImplementedError

            aggregator = RecEraserAggregator(
                emb_dim=emb_dim,
                num_shards=num_shards,
                att_dim=64
            ).to(device)


            # ========= train aggregator =========
            print("Training RecEraser aggregator...")

            train_df = pd.DataFrame({
                "user": ensemble_train[0],
                "item": ensemble_train[1]
            })

            pos_dict = np.load(self.param.pos_data, allow_pickle=True).item()

            aggregator = train_receraser_aggregator(
                train_df=train_df,
                models=models,
                aggregator=aggregator,
                device=device,
                pos_dict=pos_dict,
                n_items=self.param.n_item,
                epochs_agg=5,
                batch_size=2048,
                num_neg=4,
                lr=1e-3
            )


            # ========= test =========
            print("Start RecEraser final testing...")

            full_test_data = loadData(
                RatingData(ensemble_test),
                len(ensemble_test[0]),
                self.param.n_worker,
                False
            )

            ndcg, hr = RecEraserTest_ensemble(
                full_test_data,
                models,
                aggregator,
                device,
                pos_dict,
                self.param.n_item,
                top_k=10
            )

            print("Final RecEraser HR:", hr)
            print("Final RecEraser NDCG:", ndcg)

        elif self.param.learn_type == 'ultrare':

            group = self.param.n_group
            shard_model_paths = []

            for i in range(group):

                print(f"Training shard {i+1}/{group}")
                # load data
                if model_type in ['wmf', 'dmf', 'neumf']:
                    train_data = loadData(RatingData(train_rating[i]), self.param.batch,
                                          self.param.n_worker,
                                          True)
                elif model_type in ['bpr']:
                    train_data = loadData(PairData(train_rating[i], self.param.pos_data), self.param.batch,
                                          self.param.n_worker,
                                          True)

                test_data = loadData(RatingData(test_rating[i]), len(test_rating[i][0]), self.param.n_worker, False)
                if len(active_rating[i][0]) > 0:
                    active_test_data = loadData(RatingData(active_rating[i]), len(active_rating[i][0]),
                                                self.param.n_worker,
                                                False)
                else:
                    active_test_data = None
                
                if len(inactive_rating[i][0]) > 0:
                    inactive_test_data = loadData(RatingData(inactive_rating[i]), len(inactive_rating[i][0]),
                                                  self.param.n_worker,
                                                  False)
                else:
                    inactive_test_data = None

                model = Scratch(self.param, model_type)
                #model, result = model.train(train_data,test_data,None,None,verbose)
                model, result = model.train(train_data, test_data, active_test_data, inactive_test_data, verbose,
                                            given_model='')

                result.update({'model': model_type, 'dataset': self.param.dataset, 'deltype': self.param.del_type,
                               'method': self.param.learn_type, 'group': i + 1})
                
                save_dir = f'results/{self.param.learn_type}'
                os.makedirs(save_dir, exist_ok=True)   
                file_name = f'group{i + 1}_{self.param.n_group}_{model_type}_{self.param.dataset}_{self.param.del_type}_{self.param.del_per}.pth'
                model_path = os.path.join(save_dir, file_name)

                torch.save(model.state_dict(), model_path)
                shard_model_paths.append(model_path)

                #np.save(save_path, result)
                
                print(f'End of Group {str(i + 1)} / {group} training', self.name)

        

            # ==========================
            # UltraRE ensemble testing: load all shard models and test on the full test set (not split by group)
            # InUltraRE, we use RecEraser aggregation because in the orginal paper, the autho said LR is only efficience than aggregation
            # ==========================
            print("Start UltraRE aggregation...")

            models = []

            for path in shard_model_paths:

                if model_type == 'neumf':
                    m = NeuMF(
                        self.param.n_user,
                        self.param.n_item,
                        self.param.k,
                        self.param.layers
                    )
                elif model_type == 'wmf':
                    m = WMF(
                        self.param.n_user,
                        self.param.n_item,
                        self.param.k
                    )
                else:
                    raise NotImplementedError

                state_dict = torch.load(path, map_location=device, weights_only=True)
                m.load_state_dict(state_dict)

                m.to(device)
                m.eval()

                models.append(m)


            # ========= initialize aggregator =========
            num_shards = len(models)
            if model_type == "neumf":
                emb_dim = (
                    models[0].user_mat_mlp.weight.shape[1] +
                    models[0].user_mat_mf.weight.shape[1]
                )

            elif model_type in ["wmf", "bpr"]:
                emb_dim = models[0].user_mat.weight.shape[1]

            elif model_type == "dmf":
                emb_dim = models[0].layers[-1]   # DMF last layer output as embedding

            else:
                raise NotImplementedError

            aggregator = RecEraserAggregator(
                emb_dim=emb_dim,
                num_shards=num_shards,
                att_dim=64
            ).to(device)


            # ========= train aggregator =========
            print("Training RecEraser aggregator...")

            train_df = pd.DataFrame({
                "user": ensemble_train[0],
                "item": ensemble_train[1]
            })

            pos_dict = np.load(self.param.pos_data, allow_pickle=True).item()

            aggregator = train_receraser_aggregator(
                train_df=train_df,
                models=models,
                aggregator=aggregator,
                device=device,
                pos_dict=pos_dict,
                n_items=self.param.n_item,
                epochs_agg=5,
                batch_size=2048,
                num_neg=4,
                lr=1e-3
            )


            # ========= test =========
            print("Start RecEraser final testing...")

            full_test_data = loadData(
                RatingData(ensemble_test),
                len(ensemble_test[0]),
                self.param.n_worker,
                False
            )

            ndcg, hr = RecEraserTest_ensemble(
                full_test_data,
                models,
                aggregator,
                device,
                pos_dict,
                self.param.n_item,
                top_k=10
            )

            print("Final UltraRE HR:", hr)
            print("Final UltraRE NDCG:", ndcg)
        else:
            raise NotImplementedError('Learning type not included!')
            
    def run(self, verbose=2):
        self.runModel(self.param.model, verbose)

'''
    group = self.param.n_group
    for i in range(group):
        # load data
        if model_type in ['wmf', 'dmf', 'neumf']:
            train_data = loadData(RatingData(train_rating[i]), self.param.batch,
                                self.param.n_worker,
                                True)
        elif model_type in ['bpr']:
            train_data = loadData(PairData(train_rating[i], self.param.pos_data), self.param.batch,
                                self.param.n_worker,
                                True)

        test_data = loadData(RatingData(test_rating[i]), len(test_rating[i][0]), self.param.n_worker, False)
        if len(active_rating[i][0]) > 0:
            active_test_data = loadData(RatingData(active_rating[i]), len(active_rating[i][0]),
                                        self.param.n_worker,
                                        False)
        else:
            active_test_data = None
        inactive_test_data = loadData(RatingData(inactive_rating[i]), len(inactive_rating[i][0]),
                                    self.param.n_worker,
                                    False)

        model = Scratch(self.param, model_type)
        model, result = model.train(train_data, test_data, active_test_data, inactive_test_data, verbose,
                                    given_model='')
        result.update({'model': model_type, 'dataset': self.param.dataset, 'deltype': self.param.del_type,
                    'method': self.param.learn_type, 'group': i + 1})
        
        save_dir = f'results/{self.param.learn_type}'
        os.makedirs(save_dir, exist_ok=True)   
        file_name = f'group{i + 1}_{self.param.n_group}_{model_type}_{self.param.dataset}_{self.param.del_type}_{self.param.del_per}.npy'
        save_path = os.path.join(save_dir, file_name)
        np.save(save_path, result)
        
        print(f'End of Group {str(i + 1)} / {group} training', self.name)
'''
