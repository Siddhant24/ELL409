import numpy as np
import pandas as pd
import matplotlib as plt
import scipy as sc

import time
import random
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)



################################################################
		# DISTANCES #
################################################################

def euclideanDistance (X,y):
    if len(X.shape)==1: return np.sqrt(np.sum(np.square(X-y)))
    else: return np.sqrt(np.sum((np.square(X - y)), axis = 1))
def manhattenDistance (X,y): 
    if len(X.shape)==1: return np.sum(np.abs(X-y))
    else: return np.sum((np.abs(X - y)), axis = 1)
def chebyschevDistance (X,y): 
    if len(X.shape)==1: return np.max(np.abs(X-y))
    else: return np.array(np.max((np.abs(X - y)), axis = 1))
