import numpy as np
import pandas as pd
import matplotlib as plt

import time
import random
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

#UTILITY FUNCTIONS
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

#DISTANCES
def euclideanDistance (X,y):
    if len(X.shape)==1: return np.sqrt(np.sum(np.square(X-y)))
    else: return np.sqrt(np.sum((np.square(X - y)), axis = 1))
def manhattenDistance (X,y): 
    if len(X.shape)==1: return np.sum(np.abs(X-y))
    else: return np.sum((np.abs(X - y)), axis = 1)
def chebyschevDistance (X,y): 
    if len(X.shape)==1: return np.max(np.abs(X-y))
    else: return np.array((np.abs(X - y)), axis = 1)

################################################################

from mnist import MNIST
def prepareFMNISTData(scale = 0, PCA_threshold = 1):
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

    [Z_train, p, Xr, U, W] = PCA(X_train, PCA_threshold)
    [Z_test, Xr] = project(X_test, U, p)
    [Z_val, Xr] = project(X_val, U, p)
    X_train = Z_train[:, :p]
    X_val = Z_val[:, :p]
    X_test = Z_test[:, :p]

    if(scale == 1):
        mean = np.mean(X_train, axis = 0)
        X_train = X_train - mean
        X_test = X_test - mean
        X_val = X_val - mean
        
        variance = np.var(X_train, axis = 0)
        X_train = X_train/np.sqrt(variance)
        X_test = X_test/np.sqrt(variance)
        X_val = X_val/np.sqrt(variance)


    return (X_train, y_train, X_val, y_val, X_test, y_test) 
def prepareMedicalData(scale = 0):
    medicalData = pd.read_csv('Medical_data.csv')
    
    '''
    print("GROUPED Mean")
    print(medicalData[['Health', 'TEST1', 'TEST2', 'TEST3']].groupby('Health').mean())
    print("GROUPED Standard Deviation")
    print(medicalData[['Health', 'TEST1', 'TEST2', 'TEST3']].groupby('Health').std())
    '''

    medicalData['Health'] = medicalData['Health'].map({'HEALTHY': 0, 'MEDICATION': 1, 'SURGERY': 2}).astype(int)
    # Healthy == 0
    # Medication == 1
    # Surgery == 2
    X = medicalData.values[::, 1::]
    y = medicalData.values[::, 0].astype(int)

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
        y_train = y[trainingIndex]
        
    
    X_val = X[validationIndex]
    y_val = y[validationIndex]

    X_test = X[testIndex]
    y_test = y[testIndex]

    return (X_train, y_train, X_val, y_val, X_test, y_test)
def prepareRailwayData(scale = 0):
    ##RAILWAY BOOKING DATA
    #membercount from 0 to 10, add 1
    #preferredClass : FIRST_AC, NO_PREF, SECOND_AC, THIRD_AC
    #Age is age category 0 to 8
    railwayData = pd.read_csv('railwayBookingList.csv')

    railwayData['sex'] = railwayData['sex'].map({'female': 1, 'male': 0})
    railwayData.fillna(0, inplace = True)
    railwayData['memberCount'] = railwayData['memberCount'] + 1
    railwayData['preferredClass'] = railwayData['preferredClass'].map({'FIRST_AC': 3, 'SECOND_AC': 2, 'THIRD_AC': 1, 'NO_PREF': 0})

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
        y_train = y[trainingIndex]

            
    X_val = X[validationIndex]
    y_val = y[validationIndex]
    
    X_test = X[testIndex]
    y_test = y[testIndex]

    return (X_train, y_train, X_val, y_val, X_test, y_test)

################################################################

def getKNeighbours(testX, trainX, K = 1, distanceMetric = euclideanDistance):
    dists = np.array([distanceMetric(trainX, testx) for testx in testX])
    ind = np.argpartition(dists, K, axis = 1)[:, 0:K]    
    radius = np.array( [dists[row, ind[row, K-1]] for row in range(testX.shape[0])])
    return [ind, radius ]
def knnClassifier(testX,trainX,trainY, K = 1, distanceMetric = euclideanDistance):    
    indices = getKNeighbours(testX, trainX, K, distanceMetric)[0]
    freqs = np.array([trainY[index] for index in indices])
    predY = [np.bincount(freq).argmax() for freq in freqs]
    return np.array(predY)
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
def bayesClassifier(testX, trainX, trainY, estimator, h = 1 , distanceMetric = euclideanDistance):
    A, priors = np.unique(trainY, return_counts = True)
    q = np.array([priors[idx]*estimator(testX, trainX[np.where(trainY == A[idx])], h, distanceMetric) for idx in range(len(A))])
    return np.array([A[idx] for idx in np.argmax(q, axis = 0)])
def naiveBayesClassifier(testX, trainX, trainY, estimator, h , distanceMetric = euclideanDistance):
    A, priors = np.unique(trainY, return_counts = True)
    d = trainX.shape[1]
    q = np.zeros([testX.shape[0], len(A)])
    for idx in range(len(A)):
        trainX_class_split = trainX[np.where(trainY == A[idx])]
        q[:,idx] = priors[idx]*np.prod([estimator(testX[:,i], np.array([trainX_class_split[:,i]]).transpose(), h, distanceMetric) for i in range(d)], axis=0)
    
    return np.array([A[idx] for idx in np.argmax(q, axis = 1)])

################################################################
#Evaluation Metrics
def accuracy(prediction, actual):
    return np.sum(prediction==actual)/prediction.shape[0]




