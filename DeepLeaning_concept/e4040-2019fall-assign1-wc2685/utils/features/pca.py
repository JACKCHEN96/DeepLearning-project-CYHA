import time
import numpy as np

def pca_naive(X, K):
    """
    PCA -- naive version

    Inputs:
    - X: (float) A numpy array of shape (N, D) where N is the number of samples,
         D is the number of features
    - K: (int) indicates the number of features you are going to keep after
         dimensionality reduction

    Returns a tuple of:
    - P: (float) A numpy array of shape (K, D), representing the top K
         principal components
    - T: (float) A numpy vector of length K, showing the score of each
         component vector
    """

    ###############################################
    # TODO: Implement PCA by extracting        #
    # eigenvector.You may need to sort the      #
    # eigenvalues to get the top K of them.     #
    ###############################################
    ###############################################
    #          START OF YOUR CODE         #
    ###############################################

    n_samples,n_features=X.shape[0],X.shape[1]
    cov=(X.T.dot(X))/(n_samples-1)
    eig_val,eig_vec=np.linalg.eig(cov)
    eig_idx=eig_val.argsort()[::-1]
    eig_val=eig_val[eig_idx]
    eig_vec=eig_vec[:,eig_idx]
    eig_vec_K=eig_vec[:,0:K]
    P=eig_vec_K.T
    T=eig_val[0:K]
    

    ###############################################
    #           END OF YOUR CODE         #
    ###############################################
    
    return (P, T)
