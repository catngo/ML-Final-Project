""" Latent Factor Model for Book Recommendation (ML Final Project)
    Name: Cat Ngo and Anna Serbent
    Date created: April 7th, 2019
"""

import os
import pickle
import pandas as pd
import numpy as np
from surprise import AlgoBase
from surprise import BaselineOnly
from surprise import NormalPredictor
from surprise import Reader
from surprise.accuracy import rmse
from surprise import Dataset
from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise.model_selection import train_test_split, cross_validate

class DumbBaseline(AlgoBase):
    def __init__(self):
        AlgoBase.__init__(self)

    def fit(self, trainset):
        # Here again: call base method before doing anything.
        AlgoBase.fit(self, trainset)

        # Compute the average rating. We might as well use the
        # trainset.global_mean attribute ;)
        self.the_mean = trainset.global_mean

        d_u = dict()
        for u in self.trainset.ur.keys():
            d_u[u] = np.mean([r for (_,r) in self.trainset.ur[u]]) - trainset.global_mean
        self.dict_u = d_u

        d_i = dict()
        for u in self.trainset.ir.keys():
            d_i[u] = np.mean([r for (_,r) in self.trainset.ir[u]]) - trainset.global_mean
        self.dict_i = d_i
        return self

    def estimate(self, u, i):
        if type(u) == int: #inner id
            if self.trainset.knows_user(u): b_u = self.dict_u[u]
            else: b_u = 0
            if self.trainset.knows_item(i): b_i = self.dict_i[i]
            else: b_i = 0
            return self.the_mean + b_i + b_u

        else: #raw id
            try: 
                uid = self.trainset.to_inner_uid(u)
                b_u = self.dict_u[uid]
            except: b_u = 0
            try:
                iid = self.trainset.to_inner_iid(i)
                b_i = self.dict_i(iid)
            except: b_i = 0
            return self.the_mean + b_i + b_u

def grid(trainset, trainset_test, validationset, n_factors, n_epochs, verbose = False):
    print "------------------- Results from Grid Search -----------------------------"

    min_train_rmse = np.inf
    min_train_rmse_params = []
    min_val_rmse = np.inf
    min_val_rmse_params = []

    for f in n_factors:
        for e in n_epochs:
            svd = SVD(n_factors=f, n_epochs=e)
            svd.fit(trainset)
            train_rmse = rmse(svd.test(trainset_test))
            val_rmse = rmse(svd.test(validationset))
            if verbose:
                print 'SVD with params ---  n_factors: %d, n_epochs: %d',  f, e
                print 'Training', train_rmse
                print 'Validating', val_rmse, '\n'
            if train_rmse < min_train_rmse:
                min_train_rmse = train_rmse
                min_train_rmse_params = [f,e]
            if val_rmse < min_val_rmse:
                min_val_rmse = val_rmse
                min_val_rmse_params = [f,e]

    print "##### SMALLEST TRAINING ERROR #####"
    print 'Training: ', min_train_rmse
    print 'SVD( n_factors = ', min_train_rmse_params[0], ', n_epochs = ', min_train_rmse_params[1], ')'
    print "#### SMALLEST VALIDATING ERROR ####"
    print 'Training: ', min_val_rmse
    print 'SVD( n_factors = ', min_val_rmse_params[0], ', n_epochs = ', min_val_rmse_params[1], ')'
    

    print "--------------------------------------------------------------------------"
            
def main():
    np.random.seed(1234)
    # load dataset into dataframe
    train = pd.read_csv('../data/train_update.csv', sep = ';')
    test = pd.read_csv('../data/test_update.csv', sep = ';')
    

    print train.head(5)
    print test.head(5)

    reader = Reader(rating_scale = (0,10))

    train_set = Dataset.load_from_df(train[['User-ID', 'ISBN', 'Book-Rating']], reader=reader)
    test_set = Dataset.load_from_df(test[['User-ID', 'ISBN', 'Book-Rating']], reader=reader)
    


    # to use when train on full train set
    trainset = train_set.build_full_trainset() 

    # validationset = trainset.build_testset()

    # intialize algo
    baseline = DumbBaseline()
    svd = SVD()

    # # fitting it to the training data
    # baseline.fit(trainset)
    # svd.fit(trainset)

    # # testing in the validation set
    # print 'Baseline \n'
    # print 'RMSE: ', rmse(baseline.test(validationset)), '\n' 2.655720355062527
    # print 'Matrix Factorization \n'
    # print 'RMSE: ', rmse(svd.test(validationset)) 1.0916621060800134

    #d_baseline = cross_validate(baseline, train_set, return_train_measures=True, verbose=True)
    #d_svd = cross_validate(svd, train_set, return_train_measures = True, verbose=True)

    # do 80-20 split on trainset
    # trainset,validationset = train_test_split(train_set,random_state= 1234)

    # baseline.fit(trainset)
    # svd.fit(trainset)
    # validationset = 
    # print 'Baseline \n'
    # print rmse(baseline.test(trainset.build_testset))
    # print 'SVD \n'
    # print rmse(svd.test(trainset.build_testset))
    
    
    # do 80-20 split on set with 2 ratings up data
    
    train_2 = pd.read_csv('../data/train_2_up.csv', sep = ';')
    train_2_set = Dataset.load_from_df(train_2[['User-ID', 'ISBN', 'Book-Rating']], reader=reader)
    trainset_2 = train_2_set.build_full_trainset()
    
    trainset,validationset = train_test_split(train_2_set,random_state= 1234)
    trainset_test = trainset.build_testset()

    print trainset.n_ratings
    print trainset_2.n_ratings


    baseline.fit(trainset)
    print 'Baseline \n'
    print 'Training', rmse(baseline.test(trainset_test)), '\n'
    print 'Testing', rmse(baseline.test(validationset))
    
    svd.fit(trainset)
    print 'SVD \n'
    print 'Training', rmse(svd.test(trainset_test)), '\n'
    print 'Testing', rmse(svd.test(validationset))

    # Calling grid search
    n_factors = [100, 200, 300]
    n_epochs = [3, 5, 7]
    grid(trainset, trainset_test, validationset, n_factors, n_epochs)
    

    bsl_options1 = {'method': 'als',
               'n_epochs': 5,
               'reg_u': 12,
               'reg_i': 5
               }

    bsl_options2 = {'method': 'sgd',
               'learning_rate': .00005,
               }

    #cross_validate(BaselineOnly(bsl_options = bsl_options2), data, verbose=True)
    #cross_validate(BaselineOnly(bsl_options = bsl_options1), data, verbose=True)
    #cross_validate(NormalPredictor(), data, verbose=True)

    # alg = SVD(n_factors = 25, verbose = True)
    # alg1 = SVD(n_factors = 50, verbose = True)
    # alg2 = SVD(lr_all = 0.05, verbose = True)
    # alg3 = SVD(biased = False, verbose = True)
    
    # cross_validate(alg, data, n_jobs = -1, verbose=True)
    # cross_validate(alg1, data, n_jobs = -1, verbose=True)
    # cross_validate(alg2, data, cv = 2, n_jobs = -1, verbose=True)
    # cross_validate(alg3, data, n_jobs = -1, verbose=True)


if __name__ == "__main__" :
    main()