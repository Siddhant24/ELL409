(X_train, y_train, X_val, y_val, X_test, y_test) = prepareMedicalData(scale = 0)

def parzenWindowEstimation_hypercube(testX, trainX, h = 1, distanceMetric = euclideanDistance):
    d = trainX.shape[1]
    n = trainX.shape[0]
    V = np.power(h,d)
    estimates = np.array([(np.sum(distanceMetric(trainX, testx) < h/2))/(n*V) for testx in testX])
    return estimates

def gaussianMixtureModelEstimation(testX, trainX, K=1, distanceMetric = euclideanDistance, it=5):
    #Alpha(S) is a scalar
    #Nu(S) is a vector
    #Sigma(S) is a scalar, SI gives covariance matrix
    
    #Wt(ij): Probability of i'th point being in j'th matrix
    #Wt: N*K, W: K*N
    #trainX: n*d

    [N,D] = trainX.shape
    W = np.random.rand(K, N)
    W = W/np.sum(W, axis = 0)     #Probabilities across columns add to one

    mu = np.random.rand(K,D)
    sigma = np.random.rand(K, 1)
    alpha = np.random.rand(K, 1)
    alpha = alpha / np.sum(alpha, 1)
    
    #E-Step
    #M-step
    print(N,D,K)
    print('W', W.shape)
    numiter = 0
    while numiter < it:
        mu = (np.dot(W, trainX).transpose()/np.sum(W, axis = 1)).transpose()
#     print('Mu', mu.shape)
        sigma = np.zeros([K, D])
        for idx in range(K):
            sigma[idx] = np.sum(np.dot(np.dot(W[idx], trainX-mu[idx]), (trainX-mu[idx]).transpose()), axis=0)/np.sum(W[idx])
        alpha = np.sum(W, axis = 0)/N
        print(sigma)
        for idx in range(K):
            W[idx] = alpha[idx] * scipy.stats.multivariate_normal.pdf(trainX, mu[idx], sigma[idx])
            W[idx] = W[idx]/np.sum(W[idx])
        
        numiter = numiter+1
                   
#     print('Sigma', sigma.shape)
#     sigma = np.array( [np.square(euclideanDistance(trainX, mean)) for mean in mu])
    
   
    
    
    
    
                   
(X_train, y_train, X_val, y_val, X_test, y_test) = prepareMedicalData(scale = 1) 
gaussianMixtureModelEstimation(X_test, X_train, 50, 5)   
    