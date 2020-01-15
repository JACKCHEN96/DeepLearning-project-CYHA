#!/usr/bin/env python
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell


class MyLSTMCell(RNNCell):
    """
    Your own basic LSTMCell implementation that is compatible with TensorFlow. To solve the compatibility issue, this
    class inherits TensorFlow RNNCell class.

    For reference, you can look at the TensorFlow LSTMCell source code. It's located at tensorflow/tensorflow/python/ops/rnn_cell_impl.py.
    If you're using Anaconda, it's located at
    anaconda_install_path/envs/your_virtual_environment_name/site-packages/tensorflow/python/ops/rnn_cell_impl.py

    So this is basically rewriting the TensorFlow LSTMCell, but with your own language.
    Also, you will find Colah's blog about LSTM to be very useful:
    http://colah.github.io/posts/2015-08-Understanding-LSTMs/
    """

    def __init__(self, num_units, num_proj, forget_bias=1.0, activation=None):
        """
        Initialize a class instance.

        In this function, you need to do the following:

        1. Store the input parameters and calculate other ones that you think necessary.

        2. Initialize some trainable variables which will be used during the calculation.

        :param num_units: The number of units in the LSTM cell.
        :param num_proj: The output dimensionality. For example, if you expect your output of the cell at each time step to be a 10-element vector, then num_proj = 10.
        :param forget_bias: The bias term used in the forget gate. By default we set it to 1.0.
        :param activation: The activation used in the inner states. By default we use tanh.

        There are biases used in other gates, but since TensorFlow doesn't have them, we don't implement them either.
        """
        super(MyLSTMCell, self).__init__(_reuse=True)
        #############################################
        #           TODO: YOUR CODE HERE            #
        #############################################
        #
        # Init, just copy the params from above.
        params = []
        params.append(num_units)
        params.append(num_proj)
        params.append(forget_bias)
        params.append(activation)
        self.params = params

        # W
        self.w={
            'Wh': tf.Variable(tf.random_normal([num_units, num_proj]), name='Wh', dtype=tf.float32),
            'Wf': tf.Variable(tf.random_normal([num_proj, 1]), name='Wf', dtype=tf.float32),
            'Wi': tf.Variable(tf.random_normal([num_proj, 1]), name='Wi', dtype=tf.float32),
            'Wc': tf.Variable(tf.random_normal([num_proj, 1]), name='Wc', dtype=tf.float32),
            'Wo': tf.Variable(tf.random_normal([num_proj, 1]), name='Wo', dtype=tf.float32),
            'Wf_i': tf.Variable(tf.random_normal([1, 1]), name='Wf_i', dtype=tf.float32),
            'Wi_i': tf.Variable(tf.random_normal([1, 1]), name='Wi_i', dtype=tf.float32),
            'Wc_i': tf.Variable(tf.random_normal([1, 1]), name='Wc_i', dtype=tf.float32),
            'Wo_i': tf.Variable(tf.random_normal([1, 1]), name='Wo_i', dtype=tf.float32)
        }
        # b
        self.b={
            'bf': tf.Variable(tf.random_normal([1, 1]), name='bf', dtype=tf.float32),
            'bi': tf.Variable(tf.random_normal([1, 1]), name='bi', dtype=tf.float32),
            'bc': tf.Variable(tf.random_normal([1, 1]), name='bc', dtype=tf.float32),
            'bo': tf.Variable(tf.random_normal([1, 1]), name='bo', dtype=tf.float32),
        }

    # The following 2 properties are required when defining a TensorFlow RNNCell.
    @property
    def state_size(self):
        """
        Overrides parent class method. Returns the state size of of the cell.

        state size = num_units + output_size

        :return: An integer.
        """
        #############################################
        #           TODO: YOUR CODE HERE            #
        #############################################
        return self.params[0] + self.params[1]

    @property
    def output_size(self):
        """
        Overrides parent class method. Returns the output size of the cell.

        :return: An integer.
        """
        #############################################
        #           TODO: YOUR CODE HERE            #
        #############################################
        #
        return self.params[1]

    def call(self, inputs, state):
        """
        Run one time step of the cell. That is, given the current inputs and the state from the last time step, calculate the current state and cell output.

        You will notice that TensorFlow LSTMCell has a lot of other features. But we will not try them. Focus on the very basic LSTM functionality.

        Hint 1: If you try to figure out the tensor shapes, use print(a.get_shape()) to see the shape.

        Hint 2: In LSTM there exist both matrix multiplication and element-wise multiplication. Try not to mix them.

        :param inputs: The input at the current time step. The last dimension of it should be 1.
        :param state:  The state value of the cell from the last time step. The state size can be found from function state_size(self).
        :return: A tuple containing (output, new_state). For details check TensorFlow LSTMCell class.
        """
        #############################################
        #           TODO: YOUR CODE HERE            #
        #############################################
        #
        params = self.params
        Wh=self.w['Wh']

        # pass W for h_p
        Wf = self.w['Wf']
        Wi = self.w['Wi']
        Wc = self.w['Wc']
        Wo = self.w['Wo']

        # pass W for input
        Wf_i = self.w['Wf_i']
        Wi_i = self.w['Wi_i']
        Wc_i = self.w['Wc_i']
        Wo_i = self.w['Wo_i']

        # pass b
        bf = self.b['bf']
        bi = self.b['bi']
        bc = self.b['bc']
        bo = self.b['bo']

        # previous
        c_p = tf.slice(state, [0, 0], [-1, params[0]])
        h_p = tf.slice(state, [0, params[0]], [-1, params[1]])
        # LSTM Formulas
        # cited from: the lecture and http://colah.github.io/posts/2015-08-Understanding-LSTMs/
        f = tf.sigmoid(tf.matmul(h_p, Wf) + tf.multiply(inputs, Wf_i) + bf)
        i = tf.sigmoid(tf.matmul(h_p, Wi) + tf.multiply(inputs, Wi_i) + bi)
        c_n = tf.tanh(tf.matmul(h_p, Wc) + tf.multiply(inputs, Wc_i) + bc)
        c = f * c_p + i * c_n
        o = tf.sigmoid(tf.matmul(h_p, Wo) + tf.multiply(inputs, Wo_i) + bo)
        h = o * tf.tanh(c)
        h = tf.matmul(h, Wh)

        new_state = (tf.concat([c, h], 1))
        output = h
        return output, new_state
