""" Latent Factor Model for Book Recommendation (ML Final Project)
    Name: Cat Ngo and Anna Serbent
    Date created: April 7th, 2019
"""

import os
import pandas as pd
import numpy as np
from surprise import BaselineOnly
from surprise import NormalPredictor
from surprise import Reader
from surprise import Dataset
from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise.model_selection import cross_validate

class dumbBaseline(AlgoBase):
    def __init__(self):
        AlgoBase.__init__(self)

    def fit(self, trainset):
        # Here again: call base method before doing anything.
        AlgoBase.fit(self, trainset)

        # Compute the average rating. We might as well use the
        # trainset.global_mean attribute ;)
        self.the_mean = trainset.global_mean

        d_u = dict()
        for u in self.train_set.ur.keys():
            d_u[u] = np.mean([r for (_,r) in self.train_set.ur[u]]) - trainset.global_mean
        self.dict_u = d_u

        d_i = dict()
        for u in self.train_set.ir.keys():
            d_i[u] = np.mean([r for (_,r) in self.train_set.ir[u]]) - trainset.global_mean
        self.dict_i = d_i
        return self

    def estimate(self, u, i):

        return self.the_mean + self.dict_i[i] + self.dict_u[u]


def main():

    # load dataset into dataframe
    train = pd.read_csv('../data/train_update.csv', sep = ';')
    test = pd.read_csv('../data/test_update.csv', sep = ';')
    

    print train.head(5)
    print test.head(5)

    reader = Reader(rating_scale = (0,10))

    train_set = Dataset.load_from_df(train[['User-ID', 'ISBN', 'Book-Rating']], reader=reader)
    test_set = Dataset.load_from_df(test[['User-ID', 'ISBN', 'Book-Rating']], reader=reader)

    bsl_options1 = {'method': 'als',
               'n_epochs': 5,
               'reg_u': 12,
               'reg_i': 5
               }

    bsl_options2 = {'method': 'sgd',
               'learning_rate': .00005,
               }

    #cross_validate(BaselineOnly(bsl_options = bsl_options1), data, verbose=True)
    #cross_validate(BaselineOnly(bsl_options = bsl_options2), data, verbose=True)
    #cross_validate(NormalPredictor(), data, verbose=True)

    # alg = SVD(n_factors = 25, verbose = True)
    # alg1 = SVD(n_factors = 50, verbose = True)
    alg2 = SVD(lr_all = 0.05, verbose = True)
    #alg3 = SVD(biased = False, verbose = True)
    
    # cross_validate(alg, data, n_jobs = -1, verbose=True)
    # cross_validate(alg1, data, n_jobs = -1, verbose=True)
    cross_validate(alg2, data, cv = 2, n_jobs = -1, verbose=True)
    # cross_validate(alg3, data, n_jobs = -1, verbose=True)




if __name__ == "__main__" :
    main()