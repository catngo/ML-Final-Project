""" Latent Factor Model for Book Recommendation (ML Final Project)
    Name: Cat Ngo and Anna Serbent
    Date created: April 7th, 2019
"""

import os
import pickle
import pandas as pd
import numpy as np
import csv
import pylab
from surprise import AlgoBase
from surprise import BaselineOnly
from surprise import NormalPredictor
from surprise import KNNBasic
from surprise import Reader
from surprise.accuracy import rmse
from surprise import Dataset
from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise.model_selection import train_test_split, cross_validate
from surprise.model_selection.search import GridSearchCV
import matplotlib.pyplot as plt

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

def get_results(setNum, num_factors, reg_term):
    reader = Reader(rating_scale = (0,10))
    train = pd.read_csv('../data/train_'+str(setNum)+'.csv', sep = ';')
    # test = pd.read_csv('../data/test_update.csv', sep = ';')
    train_set = Dataset.load_from_df(train[['User-ID', 'ISBN', 'Book-Rating']], reader=reader)
    # test_set = Dataset.load_from_df(test[['User-ID', 'ISBN', 'Book-Rating']], reader=reader)
    data = train_set.build_full_trainset()

    svd = SVD(n_factors = num_factors, reg_all=reg_term)
    svd_bias = SVD(n_factors = num_factors, biased=True, reg_all=reg_term)
    baseline = DumbBaseline()

    cv_svd = cross_validate(svd, train_set, n_jobs = -2, return_train_measures=True)
    cv_svd_bias = cross_validate(svd_bias, train_set, n_jobs = -2, return_train_measures=True)
    cv_baseline = cross_validate(baseline, train_set, n_jobs = -2, return_train_measures=True)

    # getting the results ready to plot
    val_res = [np.mean(cv_svd['test_rmse']), np.mean(cv_svd_bias['test_rmse']),np.mean(cv_baseline['test_rmse'])]
    train_res = [np.mean(cv_svd['train_rmse']), np.mean(cv_svd_bias['train_rmse']),np.mean(cv_baseline['train_rmse'])]
    val_err = [np.std(cv_svd['test_rmse']), np.std(cv_svd_bias['test_rmse']),np.std(cv_baseline['test_rmse'])]
    train_err = [np.std(cv_svd['train_rmse']), np.std(cv_svd_bias['train_rmse']),np.std(cv_baseline['train_rmse'])]
    algs = ['SVD', 'SVD With Bias', 'Baseline']

    return val_res,train_res,val_err,train_err,algs

def plot_from_results(val_res,train_res,val_err,train_err,algs):
    # First Bar Chart for Validation
    plt.bar(algs, val_res, yerr=val_err, color=['m', 'r', 'g', 'b'])
    plt.title('Validation RMSE for 3 Models on Training Set '+str(setNum))
    plt.xlabel('Model')
    plt.ylabel('RMSE')
    plt.savefig('/Users/annascomputer/Documents/GitHub/ML-Final-Project/results/ValidationBarChart'+str(setNum))
    plt.show()

    # Second Bar Chart for Training
    plt.bar(algs, train_res, yerr=train_err, color=['m', 'r', 'g', 'b'])
    plt.savefig('/Users/annascomputer/Documents/GitHub/ML-Final-Project/results/TrainingBarChart'+str(setNum))
    plt.title('Training RMSE for 3 Models on Training Set '+str(setNum))
    plt.xlabel('Model')
    plt.ylabel('RMSE')
    plt.show()

    # Save data from these charts
    with open('/Users/annascomputer/Documents/GitHub/ML-Final-Project/Testing RMSE for 3 Models on Training Set '+str(setNum)+'.csv', mode='w') as f:
        w = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        w.writerow(['Factors = ' + str(num_factors)+ ', Reg = '+ str(reg_term)])
        w.writerow(['Alg', 'Testing Result', 'Testing STD', 'Validation Result', 'Validation STD'])
        for i in range(len(val_err)):
            w.writerow([algs[i], test_res[i], test_err[i], train_res[i], train_err[i]])

def vary_factors(setNum, n_factors):
    reader = Reader(rating_scale = (0,10))
    train = pd.read_csv('../data/train_'+str(setNum)+'.csv', sep = ';')
    train_set = Dataset.load_from_df(train[['User-ID', 'ISBN', 'Book-Rating']], reader=reader)
    data = train_set.build_full_trainset()

    train_errors = []
    val_errors = []
    for f in n_factors:
        svd = SVD(n_factors = f)
        cv = cross_validate(svd, train_set, return_train_measures=True, n_jobs = -2, verbose=True)
        train_errors += [np.mean(cv['train_rmse'])]
        val_errors += [np.mean(cv['test_rmse'])]
    return train_errors, val_errors

def plot_factors(n_factors, train_errors, val_errors):
    # plotting 
    plt.figure(1)
    plt.plot(n_factors,train_errors, 'r--', label='Training Error')
    plt.plot(n_factors, val_errors, 'b--', label='Validation Error')
    pylab.legend(loc='upper right')
    plt.savefig("/Users/annascomputer/Documents/GitHub/ML-Final-Project/results/ErrorIncreadingFactors.png")

def grid_search(setNum):

    reader = Reader(rating_scale = (0,10))
    train = pd.read_csv('../data/train_'+str(setNum)+'.csv', sep = ';')
    train_set = Dataset.load_from_df(train[['User-ID', 'ISBN', 'Book-Rating']], reader=reader)
    data = train_set.build_full_trainset()

    param_grid = {'n_factors': [200,400,600,800,1000]}
    gs = GridSearchCV(SVD, param_grid)
    gs.fit(data)
    ## best RMSE score
    print(gs.best_score['rmse'])

    ## combination of parameters that gave the best RMSE score
    print(gs.best_params['rmse'])
            
def main():
    ''' Main function that calls other things depending on what we want to produce, comment in
    or out lines of code as needed '''
    np.random.seed(1234)

    #--------- JUST GET RESULTS FROM MODELS -------#
    ## get_results takes in: setNum, factors, reg term
    val_res,train_res,val_err,train_err,test_res,algs = get_results(10, 200, 0.2) 
    plot_from_results(val_res,train_res,val_err,train_err,test_res,algs)
    #--------------------------------------------------#

    #--------- VARYING FACTORS AND MAKING PLOT -------#
    ## Calling grid search, we have run and gotten plots for the following ranges
    ## FACTOR RANGE 1: [2,3,5,7,10,15,25,50,100,150,200,250,300,350,400]
    ## FACTOR RANGE 2: [100,200,300,400,500,600,700]
    n_factors = [100,200,300,400,500,600,700]
    train_errors, val_errors = vary_factors(15, n_factors)
    # Note: If you don't want to run above (costly) the results are below for Frange2
    # train_errors = [0.8161, 0.5196, 0.3847, 0.3041, 0.2519, 0.2154, 0.1891]
    # val_errors = [3.4995, 3.4517, 3.4271, 3.4170,  3.4123, 3.4110, 3.4113]
    
    
    plot_factors(n_factors, train_errors, val_errors)
    #--------------------------------------------------#

    #--------       Grid Search Call    ---------------#
    # grid_search(setNum)
    #--------------------------------------------------#

if __name__ == "__main__" :
    main()


###### ARCHIVED #######
# NOTE: below are lines of code from main that are not regularly being used but can be added back if needed
#######################

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

    '''
    baseline.fit(trainset)
    print 'Baseline \n'
    print 'Training', rmse(baseline.test(trainset_test)), '\n'
    print 'Testing', rmse(baseline.test(validationset))
    
    svd.fit(trainset)
    print 'SVD \n'
    print 'Training', rmse(svd.test(trainset_test)), '\n'
    print 'Testing', rmse(svd.test(validationset))
    '''
    '''

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

    '''