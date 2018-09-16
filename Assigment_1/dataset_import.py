import numpy as np
import pandas as pd
import matplotlib as plt
import scipy.linalg
from PCA import *

import time
import random
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

################################################################
		# DATASET IMPORT FUNCTIONS #
################################################################

################################################################
		# FMNIST #
################################################################
from mnist import MNIST
def prepareFMNISTData(scale = 0, PCA_threshold = -1, Whitening = 0, PCA_p = None):
    mndata = MNIST('fashion_data')
    imagesTrain,labelsTrain = mndata.load_training()
    imagesTest, labelsTest = mndata.load_testing()

    X_test = np.array(imagesTest)
    y_test = np.array(labelsTest)

    
    n = len(imagesTrain)
    np.random.seed(RANDOM_SEED)
    indices = np.random.permutation(n)

    trainingIndex = indices[:int(4*n/5)]
    validationIndex = indices[int(4*n/5):]

    X_train = np.array(imagesTrain)[trainingIndex]
    y_train = np.array(labelsTrain)[trainingIndex]
    
    X_val = np.array(imagesTrain)[validationIndex]
    y_val = np.array(labelsTrain)[validationIndex]

    if(PCA_threshold != -1):

        [Z_train, p, Xr, U, W] = PCA(X_train, PCA_threshold)
        if PCA_p is not None: p = PCA_p
        [Z_test, Xr] = project(X_test, U, p)
        [Z_val, Xr] = project(X_val, U, p)
        X_train = Z_train[:, :p]
        X_val = Z_val[:, :p]
        X_test = Z_test[:, :p]
        print("PCA_Threshold = " + str(PCA_threshold) + ", P = " + str(p))

    if(scale == 1):
        mean = np.mean(X_train, axis = 0)
        X_train = X_train - mean
        X_test = X_test - mean
        X_val = X_val - mean
        
        variance = np.var(X_train, axis = 0)
        X_train = X_train/np.sqrt(variance)
        X_test = X_test/np.sqrt(variance)
        X_val = X_val/np.sqrt(variance)

    if(Whitening == 1):
    	[Z, p, X3, U, W] = PCA(X_train, 1.0)
    	X_train = whiteningTransform(X_train, W, U)
    	X_test = whiteningTransform(X_test, W, U)
    	X_val = whiteningTransform(X_val, W, U)

    return (X_train, y_train, X_val, y_val, X_test, y_test) 




################################################################
		# MEDICAL DATASET #
################################################################

def prepareMedicalData(scale = 0, PCA_threshold = -1, Whitening = 0):
    medicalData = pd.read_csv('Medical_data.csv')
    
    '''
    print("GROUPED Mean")
    print(medicalData[['Health', 'TEST1', 'TEST2', 'TEST3']].groupby('Health').mean())
    print("GROUPED Standard Deviation")
    print(medicalData[['Health', 'TEST1', 'TEST2', 'TEST3']].groupby('Health').std())
    '''
    
    medicalData['Health'] = medicalData['Health'].map({'HEALTHY': 0, 'MEDICATION': 1, 'SURGERY': 2}).astype(int)

    testData = pd.read_csv('test_medical.csv')
    testData['Health'] = testData['Health'].map({'HEALTHY': 0, 'MEDICATION': 1, 'SURGERY': 2}).astype(int)

    # Healthy == 0
    # Medication == 1
    # Surgery == 2
    X = medicalData.values[::, 1::]
    y = medicalData.values[::, 0].astype(int)

    X_test = testData.values[::, 1::]
    y_test = testData.values[::, 0].astype(int)


    n = X.shape[0]
    np.random.seed(RANDOM_SEED)
    indices = np.random.permutation(n)
    trainingIndex = indices[:int(5*n/6)]
    validationIndex = indices[int(5*n/6):]

    X_train = X[trainingIndex]
    y_train = y[trainingIndex]
    

    if(scale == 1):
        mean = np.mean(X_train, axis = 0)
        X = X - mean
        variance = np.var(X_train, axis = 0)
        X = X/np.sqrt(variance)
        
        X_test = X_test - mean
        X_test = X_test/np.sqrt(variance)

        X_train = X[trainingIndex]
        
    X_val = X[validationIndex]
    y_val = y[validationIndex]

    
    if(Whitening == 1):
    	[Z, p, X3, U, W] = PCA(X_train, 1.0)
    	X_train = whiteningTransform(X_train, W, U)
    	X_test = whiteningTransform(X_test, W, U)
    	X_val = whiteningTransform(X_val, W, U)
        
    if(PCA_threshold!=-1):
    	[Z_train, p, Xr, U, W] = PCA(X_train, PCA_threshold)
    	[Z_test, Xr] = project(X_test, U, p)
    	[Z_val, Xr] = project(X_val, U, p)
    	X_train = Z_train[:, :p]
    	X_val = Z_val[:, :p]
    	X_test = Z_test[:, :p]

    
    

    return (X_train, y_train, X_val, y_val, X_test, y_test)



################################################################
		# RAILWAY DATA #
################################################################

def prepareRailwayData(scale = 0, PCA_threshold = -1, Whitening = 0, OneHotEncoding = 0):
    ##RAILWAY BOOKING DATA
    #membercount from 0 to 10, add 1
    #preferredClass : FIRST_AC, NO_PREF, SECOND_AC, THIRD_AC
    #Age is age category 0 to 8
    railwayData = pd.read_csv('railwayBookingList.csv')

    railwayData['sex'] = railwayData['sex'].map({'female': 0, 'male': 1})
    railwayData.fillna(0, inplace = True)
    railwayData['memberCount'] = railwayData['memberCount'] + 1
    railwayData['preferredClass'] = railwayData['preferredClass'].map({'FIRST_AC': 1, 'SECOND_AC': 2, 'THIRD_AC': 3, 'NO_PREF': 0})


    if OneHotEncoding:
        railwayData['First_AC'] = (railwayData['preferredClass'] == 1).astype(int)
        railwayData['Second_AC'] = (railwayData['preferredClass'] == 2).astype(int)
        railwayData['Third_AC'] = (railwayData['preferredClass'] == 3).astype(int)



    X = railwayData.values[::, 2::]
    y = railwayData.values[::, 1].astype(int)
    

        
    n = X.shape[0]
    np.random.seed(RANDOM_SEED)
    indices = np.random.permutation(n)
    trainingIndex = indices[:int(4*n/6)]
    validationIndex = indices[int(4*n/6): int(5*n/6)]
    testIndex = indices[int(5*n/6):]

    X_train = X[trainingIndex]
    y_train = y[trainingIndex]

    if(scale == 1):
        mean = np.mean(X_train, axis = 0)
        X = X - mean
        variance = np.var(X_train, axis = 0)
        X = X/np.sqrt(variance)
        X_train = X[trainingIndex]

    X_val = X[validationIndex]
    y_val = y[validationIndex]
    
    X_test = X[testIndex]
    y_test = y[testIndex]

    if(Whitening == 1):
        [Z, p, X3, U, W] = PCA(X_train, 1.0)
        X_train = whiteningTransform(X_train, W, U)
        X_test = whiteningTransform(X_test, W, U)
        X_val = whiteningTransform(X_val, W, U)
        
    if(PCA_threshold!=-1):
        [Z_train, p, Xr, U, W] = PCA(X_train, PCA_threshold)
        print("EigenValues: ", W)
        print("Reduced Dimension: ", p)
        [Z_test, Xr] = project(X_test, U, p)
        [Z_val, Xr] = project(X_val, U, p)
        X_train = Z_train[:, :p]
        X_val = Z_val[:, :p]
        X_test = Z_test[:, :p]

    return (X_train, y_train, X_val, y_val, X_test, y_test)











