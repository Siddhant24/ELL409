import numpy as np
import pandas as pd
import matplotlib as plt
import scipy as sc
from distance_metrics import *

import time
import random
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

################################################################
		#  Non parametric density estimators #
################################################################

def kNearestNeighboursEstimation(testX, trainX, funcN = lambda n: np.sqrt(n), distanceMetric = euclideanDistance) :
    [n,d] = trainX.shape
    k = min( max((int)(funcN(n)), 10) , n-1)
    indices, radius = getKNeighbours(testX, trainX, k, distanceMetric)
    predY = np.array([k/(n*(np.float_power(rad, d))) for rad in radius])
    return predY

    
def parzenWindowEstimation_gaussian(testX, trainX, h = 1, distanceMetric = euclideanDistance):
    d = trainX.shape[1]
    estimates = np.array([np.mean(np.exp(-np.square(distanceMetric(trainX, testx))/(2*(h*h)))/h) for testx in testX])
    return estimates