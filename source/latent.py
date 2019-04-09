""" Latent Factor Model for Book Recommendation (ML Final Project)
    Name: Cat Ngo and Anna Serbent
    Date created: April 7th, 2019
"""

import os
import pandas as pd
from surprise import BaselineOnly
from surprise import NormalPredictor
from surprise import Reader
from surprise import Dataset
from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise.model_selection import cross_validate

def main():

    # load dataset into dataframe
    df = pd.read_csv('../data/ratings.csv', sep = ';')
    
    # change ISBN into unique numerical value
    df['ISBN'] = pd.factorize(df['ISBN'])[0]

    print df.head(5)

    reader = Reader(rating_scale = (0,10))

    data = Dataset.load_from_df(df[['User-ID', 'ISBN', 'Book-Rating']], reader=reader)
    
    # We can now use this dataset as we please, e.g. calling cross_validate

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