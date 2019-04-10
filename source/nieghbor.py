from surprise import KNNBasic
from surprise import BaselineOnly
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate
import os
file_path = os.path.expanduser('~/repos/ML-Final-Project/data/clean_ratings.csv')
reader= Reader(line_format="user item rating",sep=';',skip_lines=1,rating_scale=(0,10))
data = Dataset.load_from_file(file_path, reader=reader)
print(type(data))
algo = KNNBasic()
trainset, testset = train_test_split(data, test_size=.9,random_state=1234)
algo.fit(trainset)
