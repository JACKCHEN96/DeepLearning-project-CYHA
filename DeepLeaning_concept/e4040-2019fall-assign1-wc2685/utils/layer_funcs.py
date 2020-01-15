from builtins import range
import numpy as np

def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine function.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: a numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: a numpy array of weights, of shape (D, M)
    - b: a numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    """
    ############################################################################
    # TODO: Implement the affine forward pass. Store the result in 'out'. You  #
    # will need to reshape the input into rows.                      #
    ############################################################################
    ############################################################################
    #                   START OF YOUR CODE                   #
    ############################################################################

    out=np.dot(x,w)+b
    
    ############################################################################
    #                    END OF YOUR CODE                    #
    ############################################################################
    return out


def affine_backward(dout, x, w, b):
    """
    Computes the backward pass of an affine function.

    Inputs:
    - dout: upstream derivative, of shape (N, M)
    - x: input data, of shape (N, d_1, ... d_k)
    - w: weights, of shape (D, M)
    - b: bias, of shape (M,)

    Returns a tuple of:
    - dx: gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: gradient with respect to w, of shape (D, M)
    - db: gradient with respect to b, of shape (M,)
    """
    ############################################################################
    # TODO: Implement the affine backward pass.                      #
    ############################################################################
    ############################################################################
    #                   START OF YOUR CODE                   #
    ############################################################################
    # dout=db
    db=np.sum(dout,axis=0)
    # dw=dout/x
    dw=np.dot(np.reshape(x,(x.shape[0],-1)).T,dout)
    # dx=dout/w
    dx=np.reshape(np.dot(dout,w.T),(x.shape[0],-1))
    
    
    ############################################################################
    #                    END OF YOUR CODE                    #
    ############################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for rectified linear units (ReLUs).

    Input:
    - x: inputs, of any shape

    Returns a tuple of:
    - out: output, of the same shape as x
    """
    ############################################################################
    # TODO: Implement the ReLU forward pass.                        #
    ############################################################################
    ############################################################################
    #                   START OF YOUR CODE                   #
    ############################################################################
    
    out=np.maximum(x,0)
    
    ############################################################################
    #                    END OF YOUR CODE                    #
    ############################################################################
    return out


def relu_backward(dout, x):
    """
    Computes the backward pass for rectified linear units (ReLUs).

    Input:
    - dout: upstream derivatives, of any shape

    Returns:
    - dx: gradient with respect to x
    """
    ############################################################################
    # TODO: Implement the ReLU backward pass.                       #
    ############################################################################
    ############################################################################
    #                   START OF YOUR CODE                   #
    ############################################################################
    
    dx=dout
    dx[x<0]=0
    
    ############################################################################
    #                    END OF YOUR CODE                    #
    ############################################################################
    return dx


def softmax_loss(x, y):
    """
    Softmax loss function, vectorized version.
    y_prediction = argmax(softmax(x))
    
    Inputs:
    - x: (float) a tensor of shape (N, #classes)
    - y: (int) ground truth label, a array of length N

    Returns:
    - loss: the cross-entropy loss
    - dx: gradients wrt input x
    """
    ############################################################################
    # TODO: You can use the previous softmax loss function here.           # 
    # Hint: Be careful on overflow problem                         #
    ############################################################################
    ############################################################################
    #                   START OF YOUR CODE                   #
    ############################################################################
    
    N=x.shape[0]
    D=np.prod(x.shape[1:])
    x_reshape=np.reshape(x,(N,D))
    score=x_reshape-np.max(x_reshape,axis=1,keepdims=True)
    p=np.exp(score)
    p=p/np.sum(p,axis=1,keepdims=True)
    loss1=-np.log(p[np.arange(N),y])
    loss=np.sum(loss1)/N
    dx=p.copy()
    dx[np.arange(N),y]-=1
    dx=dx/N

    
    ############################################################################
    #                    END OF YOUR CODE                    #
    ############################################################################
    return loss, dx