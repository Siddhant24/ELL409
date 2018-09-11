import numpy as np
import pandas as pd
import matplotlib as plt
import scipy.linalg
from distance_metrics import *

import time
import random
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)



################################################################
		# PCA #
################################################################

def project(X, U, p = None):
    if p == None: p = X.shape[1]
    Z = np.matmul(X, U)
    Z[:, p:] = np.mean(Z[:, p:], axis = 0)
    X2 = np.matmul(Z, U.transpose())
    return (Z, X2)
def PCA(X, threshold = 0.9):
    X2 = X - np.mean(X, axis = 0)
    S = np.matmul(X2.transpose(), X2) #Covariance Matrix
    [W,U] = np.linalg.eigh(S) #eigen vectors in columns
    W = np.flip(W, axis = 0)
    U = np.flip(U, axis = 1)
    
    validity = np.cumsum(W)/np.sum(W)  #represents validity of choosing first i+1 eigenvalues
    p = np.argmax(validity>=threshold) + 1
    if p==1: p = X.shape[1]
    
    [Z, X3] = project(X, U, p)
    
    #Projection, P, Reconstruction, EigenVectors, EigenValues
    return [Z, p, X3, U, W]


################################################################
		# Whitening #
################################################################

def whiteningTransform(X, W, U):
	L = np.diag(W)
	Z = np.transpose(np.matmul(np.matmul(scipy.linalg.fractional_matrix_power(L, -0.5), U.transpose()), (X - np.mean(X, axis = 0)).transpose()))
	return Z
