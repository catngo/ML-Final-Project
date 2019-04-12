""" Neighbors Model for Book Recommendation (ML Final Project)
    Name: Jillian Cardamon and Teddy Dubno
    Date created: April 7th, 2019
"""

import os
import pandas as pd
import pickle
from surprise import KNNBasic
from surprise import BaselineOnly
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate

#import numpy as np
#from sklearn.cross_validation import train_test_split
#from sklearn.neighbors import KNeighborsClassifier


def main():
    # save path to training data csv
    # convert to panda Dataframe to bypass an error
    #file_path = os.path.expanduser('../data/train.csv')
    #df = pd.read_csv(path=file_path, sep = ';')

    # pickle_dict = pickle.load('../data/train_update.csv')
    # df = pd.DataFrame(ratings_dict)

    # load dataset into dataframe
    train = pd.read_csv('../data/train_update.csv', sep = ';')
    test = pd.read_csv('../data/test_update.csv', sep = ';')

    reader = Reader(rating_scale=(0,10))

    print

    train_set = Dataset.load_from_df(train[['User-ID', 'ISBN', 'Book-Rating']], reader=reader)
    test_set = Dataset.load_from_df(test[['User-ID', 'ISBN', 'Book-Rating']], reader=reader)
    # load data from file
    # data = Dataset.load_from_df(df[['User-ID', 'ISBN', 'Book-Rating']], reader=reader)

    # to use when train on full train set
    # trainset = train_set.build_full_trainset() 
    # validationset = trainset.build_testset()


    # create classifier (using a basic k nearest neighbors approach)
    algo = KNNBasic()

    trainset, testset = train_test_split(train_set, test_size=.9,random_state=1234)
    algo.fit(train_set)

    #cross_validate(algo, trainset, verbose=True)
    predictions = algo.test(testset)

    # compute MAE and RMSE
    accuracy.mae(predictions)
    accuracy.rmse(predictions)

    # SKLEARN SECTION





if __name__ == "__main__" :
    main()