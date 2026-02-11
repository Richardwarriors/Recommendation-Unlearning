import os
from turtle import pos
import warnings
from os.path import abspath, join, dirname, exists

import numpy as np
from scratch import Scratch
from utils import saveObject

from read import RatingData, PairData
from read import loadData, readRating_full, readRating_group

#/home/jiajie/Richard_He/CURE4Rec/data
DATA_DIR = abspath(join(dirname(__file__), 'data'))
#/home/jiajie/Richard_He/CURE4Rec/result
SAVE_DIR = abspath(join(dirname(__file__), 'result'))


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

    def read(self):
        learn_type = self.param.learn_type
        del_type = self.param.del_type
        del_per = self.param.del_per
        group = self.param.n_group
        if learn_type == 'retrain':
            train_rating, test_rating, active_rating, inactive_rating = readRating_full(self.param.train_dir,
                                                                                        self.param.test_dir, 
                                                                                        del_type, del_per)
        else:
            train_rating, test_rating, active_rating, inactive_rating = readRating_group(self.param.train_dir,
                                                                                         self.param.test_dir, del_type,
                                                                                         del_per, learn_type, group,
                                                                                         self.param.dataset)

        return train_rating, test_rating, active_rating, inactive_rating

    def runModel(self, model_type='wmf', verbose=2):
        print(self.name, 'begin:')
        # read raw data
        train_rating, test_rating, active_rating, inactive_rating = self.read()

        if self.param.learn_type == 'retrain':
            # load data
            if model_type in ['wmf', 'dmf', 'neumf']:
                #train_traing = [train_ratings['uid'], train_ratings['iid'], train_ratings['val']]
                train_data = loadData(RatingData(train_rating), self.param.batch,
                                      self.param.n_worker,
                                      True)

            elif model_type in ['bpr', 'lightgcn']:
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
        else:
            group = self.param.n_group
            for i in range(group):
                # load data
                if model_type in ['wmf', 'dmf', 'neumf']:
                    train_data = loadData(RatingData(train_rating[i]), self.param.batch,
                                          self.param.n_worker,
                                          True)
                elif model_type in ['bpr', 'lightgcn']:
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
            
            
    def run(self, verbose=2):
        self.runModel(self.param.model, verbose)
