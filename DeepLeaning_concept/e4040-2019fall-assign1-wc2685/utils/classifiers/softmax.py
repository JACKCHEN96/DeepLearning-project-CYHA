import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
      Softmax loss function, naive implementation (with loops)

      Inputs have dimension D, there are C classes, and we operate on minibatches
      of N examples.

      Inputs:
      - W: a numpy array of shape (D, C) containing weights.
      - X: a numpy array of shape (N, D) containing a minibatch of data.
      - y: a numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where 0 <= c < C.
      - reg: (float) regularization strength

      Returns a tuple of:
      - loss: (float) the mean value of loss functions over N examples in minibatch.
      - gradient: gradient wrt W, an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.    #
    # Store the loss in loss and the gradient in dW. If you are not careful    #
    # here, it is easy to run into numeric instability. Don't forget the      #
    # regularization!                                        #
    #############################################################################
    #############################################################################
    #                     START OF YOUR CODE                  #
    #############################################################################
    trainNUM=X.shape[0]
    classesNUM=W.shape[1]
    for i in range(trainNUM):
        score=X[i].dot(W)
        score_norm=score-np.max(score)
        loss-=np.log(np.exp(score_norm[y[i]])/np.sum(np.exp(score_norm)))
        for j in range(classesNUM):
            out=np.exp(score_norm[j])/sum(np.exp(score_norm))
            if j ==y[i]:
                dW[:,j]+=(-1+out)*X[i]
            else:
                dW[:,j]+=out*X[i]
    loss=loss/trainNUM+reg*np.sum(W*W)
    dW=dW/trainNUM+reg*2*W

    
    #############################################################################
    #                     END OF YOUR CODE                   #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful    #
    # here, it is easy to run into numeric instability. Don't forget the      #
    # regularization!                                        #
    #############################################################################
    #############################################################################
    #                     START OF YOUR CODE                  #
    #############################################################################
    trainNUM=X.shape[0]
    classesNUM=W.shape[1]
    score=X.dot(W)
    score_norm=score-np.max(score,axis=1).reshape(-1,1)
    out=np.exp(score_norm)/np.sum(np.exp(score_norm),axis=1).reshape(-1,1)
    loss=-np.sum(np.log(out[range(trainNUM),list(y)]))
    temp=out.copy()
    temp[range(trainNUM),list(y)]-=1
    dW=(X.T).dot(temp)
    loss=loss/trainNUM+reg*np.sum(W*W)
    dW=dW/trainNUM+reg*2*W
    
    #############################################################################
    #                     END OF YOUR CODE                   #
    #############################################################################

    return loss, dW
