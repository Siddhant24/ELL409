import numpy as np
import pandas as pd
import matplotlib as plt

import time
import random

#DISTANCES
def euclideanDistance (x,y):
    if len(x.shape)==1: return np.sqrt(np.sum(np.square(x-y)))
    else: return np.sqrt(np.sum((np.square(x - y)), axis = 1))

def manhattenDistance (x,y): 
    if len(x.shape)==1: return np.sum(np.abs(x-y))
    else: return np.sum((np.abs(x - y)), axis = 1)

def chebyschevDistance (x,y): 
    if len(x.shape)==1: return np.max(np.abs(x-y))
    else: return np.array((np.abs(x - y)), axis = 1)
