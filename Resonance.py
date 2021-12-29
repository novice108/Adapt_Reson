from functools import partial
import numpy as np
import sklearn.datasets as ds

from train_art import data_train
from test_art import data_test

#Regularization to penalize the parameters that are not important
l1_norm = partial(np.linalg.norm, ord=1, axis=-1)

if __name__ == '__main__': 
    #load the dataset in the python environment
    iris = ds.load_iris()
    #standardize the dataset
    data = iris['data'] / np.max(iris['data'], axis=0)
    print(data)

    r = 0.5 #Train on the data; nep - number of epochs
    Tmatrix = data_train(data, rho=r, beta=0.000001, alpha=0.5, nep=100)
    print(Tmatrix)
    
    T = data_test(data,Tmatrix,rho=r ,beta=0.000001, alpha=0.5, nep=100)
    print(T) #Print the Cluster Results
    print()
    print(iris['target']) #Match the cluster results
