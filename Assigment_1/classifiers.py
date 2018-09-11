import numpy as np
import pandas as pd
import matplotlib as plt
import scipy as sc
import time
import random
from distance_metrics import *

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
		# B A Y E S #
################################################################

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
    return np.array([X[closest==k].mean(axis=0) for k in range(centroids.shape[0])])

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
    print(numiter)
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
    return np.array(predY)