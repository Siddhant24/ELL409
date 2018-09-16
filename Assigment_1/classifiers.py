import numpy as np
import pandas as pd
import matplotlib as plt
import scipy.linalg
import time
import random
from distance_metrics import *
from evaluation_metrics import *

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

################################################################
		#  K N N  #
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


################################################################
        #  Non parametric density estimators #
################################################################

def kNearestNeighboursEstimation(testX, trainX, funcN = lambda n: np.sqrt(n), distanceMetric = euclideanDistance) :
    [n,d] = trainX.shape
    k = min( max((int)(funcN(n)), 10) , n-1)
    indices, radius = getKNeighbours(testX, trainX, k, distanceMetric)
    radius = [rad + 0.000001 for rad in radius]
    predY = np.array([k/(n*(np.float_power(rad, d))) for rad in radius])
    return predY
  
def kNearestNeighboursEstimationAuto(testX, trainX, valX, distanceMetric = euclideanDistance, kList = None):
    if kList is None: kList = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    estimates = np.array([kNearestNeighboursEstimation(valX, trainX, lambda n: k, distanceMetric) for k in kList])
    logLikelyhood = np.sum(np.log(estimates), axis = 1)
    k = kList[np.argmax(logLikelyhood)]
    return kNearestNeighboursEstimation(testX, trainX, lambda n: k, distanceMetric)

def parzenWindowEstimation_hypercube(testX, trainX, h = 1, distanceMetric = euclideanDistance):
    d = trainX.shape[1]
    n = trainX.shape[0]
    V = np.power(h,d)
    estimates = np.array([(np.sum(distanceMetric(trainX, testx) < h/2))/(n*V) for testx in testX])
    return estimates

def parzenWindowEstimation_gaussian(testX, trainX, h = 1, distanceMetric = euclideanDistance):
    d = trainX.shape[1]
    estimates = np.array([np.mean(np.exp(-np.square(distanceMetric(trainX, testx))/(2*(h*h)))/h) for testx in testX])
    return estimates

def parzenWindowEstimationAuto_gaussian(testX, trainX, valX, distanceMetric = euclideanDistance, hList = None, it = 2):
    if hList is None: hList = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000]

    estimates = np.array([parzenWindowEstimation_gaussian(valX, trainX, h, distanceMetric) for h in hList])
    logLikelyhood = np.sum(np.log(estimates), axis = 1)
    h = hList[np.argmax(logLikelyhood)]
    #print(h)
    if(it>1):
        hList2 = np.random.rand(25)*(1.5*h) + (h/2)
        return parzenWindowEstimationAuto_gaussian(testX, trainX, valX, distanceMetric, hList2, it-1)
    
    return parzenWindowEstimation_gaussian(testX, trainX, h , distanceMetric)
        
from scipy.stats import multivariate_normal
def maximumLikelyhoodEstimation(testX, trainX, a=0, b=0, c=0):
    #Find mu, sigma to fit to train data
    #Find estimates of test points
    n = trainX.shape[0]
    mu = np.mean(trainX, axis = 0, keepdims = True)
    X = (trainX - mu)
    sigma = (np.matmul((trainX - mu).T, (trainX - mu))*(1/n))
    return multivariate_normal.pdf(testX, mean = mu[0,:], cov = sigma) 

################################################################
		# B A Y E S #
################################################################

def bayesClassifier(testX, trainX, trainY, estimator, h = 1 , distanceMetric = euclideanDistance):
    A, priors = np.unique(trainY, return_counts = True)
    q = np.array([priors[idx]*estimator(testX, trainX[np.where(trainY == A[idx])], h, distanceMetric) for idx in range(len(A))])
    return np.array([A[idx] for idx in np.argmax(q, axis = 0)])

def bayesClassifierAuto(testX, trainX, trainY, valX, valY, estimatorAuto, distanceMetric):
    A, priors = np.unique(trainY, return_counts = True)
    q = np.array([priors[idx]*estimatorAuto(testX, trainX[np.where(trainY == A[idx])], valX[np.where(valY==A[idx])], distanceMetric) for idx in range(len(A))])
    return np.array([A[idx] for idx in np.argmax(q, axis = 0)])

def naiveBayesClassifier(testX, trainX, trainY, estimator, h=None, distanceMetric = euclideanDistance):
    A, priors = np.unique(trainY, return_counts = True)
    d = trainX.shape[1]
    q = np.zeros([testX.shape[0], len(A)])
    for idx in range(len(A)):
        trainX_class_split = trainX[np.where(trainY == A[idx])]
        q[:,idx] = np.log(priors[idx]) + np.sum(np.log([estimator(testX[:,i], np.array([trainX_class_split[:,i]]).transpose(), h, distanceMetric) for i in range(d)]), axis=0)
    
    return np.array([A[idx] for idx in np.argmax(q, axis = 1)])

################################################################
	# K - M E A N S #
################################################################


def initialize_centroids(X, k):
    """returns k random centroids from the data points"""
    centroids = X.copy()
    np.random.shuffle(centroids)
    return centroids[:k]

def closest_centroid(X, centroids, distanceMetric = euclideanDistance):
    """returns an array containing the index to the nearest centroid for each point"""
    distances = np.array([distanceMetric(X, centroid) for centroid in centroids])
    return np.argmin(distances, axis=0)

def reassign_centroids(X, closest, centroids):
    """returns the new centroids assigned from the points closest to them"""
    
    centrs = []
    for k in range(centroids.shape[0]):
        if np.sum(closest==k) == 0: centrs.append(centroids[k])
        else: centrs.append(X[closest==k].mean(axis=0))
    return np.array(centrs)
    

#     return np.array([X[closest==k].mean(axis=0) for k in range(centroids.shape[0])])

def KMeansClustering(X, K=3, distanceMetric = euclideanDistance):
    """returns K cluster centers"""
    centroids = initialize_centroids(X, K)
    prev_centroids = centroids
    epsilon = 1e-5
    err = 1
    numiter = 0
    while err > epsilon:
        prev_centroids = centroids
        centroids = reassign_centroids(X, closest_centroid(X, centroids, distanceMetric), centroids)
        numiter = numiter+1
        err = np.max(distanceMetric(prev_centroids, centroids))
    return centroids

def KMeansClassifier(testX,trainX,trainY, K = 3, distanceMetric = euclideanDistance):    
    """returns the label of the cluster to which a test point is assigned to"""
    centroids = KMeansClustering(trainX, K, distanceMetric)
    assigned_clusters = closest_centroid(trainX, centroids, distanceMetric)

    A = np.unique(trainY)
    clusterFreq = np.zeros([K, len(A)], dtype=int)
    for idx in range(assigned_clusters.shape[0]):
        clusterFreq[assigned_clusters[idx]][trainY[idx]] += 1
    clusterClass = [np.argmax(clusterFreq[idx]) for idx in range(K)]        
    predY = [clusterClass[x] for x in closest_centroid(testX, centroids, distanceMetric)]
    return [np.array(predY), centroids]

def kMeansClassifierAuto(testX, trainX, trainY, valX, valY, it = 10, distanceMetric = euclideanDistance):
    kList = [1,2,4,8,16,32,64,128, 256, 512]
    
    maxCentr = None
    maxAcc = 0
    maxK = 0

    for k in kList:
        for i in range(it):
            [predY, centroids] = KMeansClassifier(valX, trainX, trainY, k, distanceMetric)
            acc = accuracy(predY, valY);
            #print(k, i, acc)
            if(acc>maxAcc):
                maxK = k 
                maxAcc = acc 
                maxCentr = centroids

    assigned_clusters = closest_centroid(trainX, maxCentr, distanceMetric)

    A = np.unique(trainY)
    clusterFreq = np.zeros([maxK, len(A)], dtype=int)
    for idx in range(assigned_clusters.shape[0]):
        clusterFreq[assigned_clusters[idx]][trainY[idx]] += 1
    clusterClass = [np.argmax(clusterFreq[idx]) for idx in range(maxK)]        
    predY = [clusterClass[x] for x in closest_centroid(testX, maxCentr, distanceMetric)]
    
    return [np.array(predY), maxK, maxAcc]


        
