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
         # Evaluation Metrics #
################################################################

def accuracy(prediction, actual):
    return np.sum(prediction==actual)/prediction.shape[0]
    
def recall(prediction, actual, target):
	prediction = (prediction == target)
	actual = (actual==target)
	if(np.sum(actual) == 0): return 1.0
	return np.sum(prediction&actual>0)/np.sum(actual)

def precision(prediction, actual, target):
	prediction = (prediction == target)
	actual = (actual==target)
	if(np.sum(prediction) == 0): return 1.0
	return np.sum(prediction&actual>0)/np.sum(prediction)

def f1Score(prediction, actual, target):
	prec = precision(prediction, actual, target)
	rec = recall(prediction, actual, target)
	return 2*prec*rec/(prec + rec)
	