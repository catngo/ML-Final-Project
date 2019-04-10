""" Neighbors Model for Book Recommendation (ML Final Project)
    Name: Jillian Cardamon and Teddy Dubno
    Date created: April 7th, 2019
"""

import os
import pandas as pd
from surprise import KNNBasic
from surprise import BaselineOnly
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate

# save path to training data csv
# convert to panda Dataframe to bypass an error
file_path = os.path.expanduser('../data/train.csv')
df = pd.read_csv(path=file_path, sep = ';')

# change ISBN into unique numerical value
df['ISBN'] = pd.factorize(df['ISBN'])[0]

reader = Reader(rating_scale=(0,10))

# load data from file
data = Dataset.load_from_df(df[['User-ID', 'ISBN', 'Book-Rating']], reader=reader)

# create classifier (using a basic k nearest neighbors approach)
# 
algo = KNNBasic()
trainset, testset = train_test_split(data, test_size=.9,random_state=1234)
algo.fit(trainset)

predictions = algo.test(testset)