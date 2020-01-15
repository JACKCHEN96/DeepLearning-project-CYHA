import numpy as np
from random import shuffle

def sigmoid(x):
    h = np.zeros_like(x)
    
    #############################################################################
    # TODO: Implement sigmoid function.                       #
    #############################################################################
    #############################################################################
    #                     START OF YOUR CODE                  #
    #############################################################################
    h=1/(1+np.exp(-x))

    
    #############################################################################
    #                     END OF YOUR CODE                   #
    #############################################################################
    return h 

def logistic_regression_loss_naive(W, X, y, reg):
    """
      Logistic regression loss function, naive implementation (with loops)

      Inputs have dimension D, there are C classes, and we operate on minibatches
      of N examples.

      Inputs:
      - W: a numpy array of shape (D, C) containing weights.
      - X: a numpy array of shape (N, D) containing a minibatch of data.
      - y: a numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where c can be either 0 or 1.
      - reg: (float) regularization strength

      Returns a tuple of:
      - loss: (float) the mean value of loss functions over N examples in minibatch.
      - gradient: gradient wrt W, an array of same shape as W
    """
    # Set the loss to a random number
    loss = 0.0
    # Initialize the gradient to zero
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
    yh=np.zeros((trainNUM,2))
    for i in range(trainNUM):
        if y[i]==1:
            yh[i,:]=np.array([[0,1]])
        else:
            yh[i,:] = np.array([[1, 0]])

    for i in range(trainNUM):
        score_norm=X[i,:].dot(W)
        loss+=np.sum(yh[i,:]*np.log(sigmoid(score_norm))+(1-yh[i,:])*np.log(1-sigmoid(score_norm)))

        # loss+=loss1
        dW[:,0]+=-X[i]*(yh[i,0]-sigmoid(score_norm[0]))
        dW[:,1]+=-X[i]*(yh[i,1]-sigmoid(score_norm[1]))

    loss=-loss/trainNUM+reg/2*np.sum(W*W)
    # loss+=reg*np.sum(W*W)
    dW=dW/trainNUM+reg*W

    #############################################################################
    #                     END OF YOUR CODE                   #
    #############################################################################
    return loss, dW



def logistic_regression_loss_vectorized(W, X, y, reg):
    """
    Logistic regression loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Set the loss to a random number
    loss = 0.0
    # Initialize the gradient to zero
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the logistic regression loss and its gradient using no    # 
    # explicit loops.                                       #
    # Store the loss in loss and the gradient in dW. If you are not careful   #
    # here, it is easy to run into numeric instability. Don't forget the     #
    # regularization!                                       #
    ############################################################################
    #############################################################################
    #                     START OF YOUR CODE                  #
    #############################################################################
    trainNUM=X.shape[0]
    classesNUM=W.shape[1]
    score_norm=X.dot(W)

    yh=np.zeros((trainNUM,2))
    for i in range(trainNUM):
        if y[i]==1:
            yh[i,:]=np.array([[0,1]])
        else:
            yh[i,:] = np.array([[1, 0]])

    out=sigmoid(score_norm)

    loss=np.sum(-(yh*np.log(out)+(1-yh)*np.log(1-out)))
    # loss=-(yh*np.log(out)+(1-yh)*np.log(1-out))


    for j in range(classesNUM):
        dW[:,j]+=np.dot(X.T,(out-yh)[:,j])

    # dW=(X.T).dot(out-y)

    dW=dW/trainNUM+reg*W

    loss/=trainNUM
    loss+=reg/2*np.sum(W*W)


    
    
    
    #############################################################################
    #                     END OF YOUR CODE                   #
    #############################################################################
    return loss, dW
