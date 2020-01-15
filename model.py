import tensorflow as tf


class Model(object):

    def inference(x, drop_rate):
        with tf.variable_scope('convolution1'):
            conv = tf.layers.conv2d(x, filters=48, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            convolution1 = dropout

        with tf.variable_scope('convolution2'):
            conv = tf.layers.conv2d(convolution1, filters=64, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=1, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            convolution2 = dropout

        with tf.variable_scope('convolution3'):
            conv = tf.layers.conv2d(convolution2, filters=128, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            convolution3 = dropout

        with tf.variable_scope('convolution4'):
            conv = tf.layers.conv2d(convolution3, filters=160, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=1, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            convolution4 = dropout

        with tf.variable_scope('convolution5'):
            conv = tf.layers.conv2d(convolution4, filters=192, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            convolution5 = dropout

        with tf.variable_scope('convolution6'):
            conv = tf.layers.conv2d(convolution5, filters=192, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=1, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            convolution6 = dropout

        with tf.variable_scope('convolution7'):
            conv = tf.layers.conv2d(convolution6, filters=192, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            convolution7 = dropout

        with tf.variable_scope('convolution8'):
            conv = tf.layers.conv2d(convolution7, filters=192, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=1, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            convolution8 = dropout

        flatten = tf.reshape(convolution8, [-1, 4 * 4 * 192])

        with tf.variable_scope('dense1'):
            dense = tf.layers.dense(flatten, units=3072, activation=tf.nn.relu)
            dense1 = dense

        with tf.variable_scope('dense2'):
            dense = tf.layers.dense(dense1, units=3072, activation=tf.nn.relu)
            dense2 = dense

        with tf.variable_scope('digit_length'):
            dense = tf.layers.dense(dense2, units=7)
            length = dense

        with tf.variable_scope('digit1'):
            dense = tf.layers.dense(dense2, units=11)
            digit1 = dense

        with tf.variable_scope('digit2'):
            dense = tf.layers.dense(dense2, units=11)
            digit2 = dense

        with tf.variable_scope('digit3'):
            dense = tf.layers.dense(dense2, units=11)
            digit3 = dense

        with tf.variable_scope('digit4'):
            dense = tf.layers.dense(dense2, units=11)
            digit4 = dense

        with tf.variable_scope('digit5'):
            dense = tf.layers.dense(dense2, units=11)
            digit5 = dense

        length_logits, digits_logits = length, tf.stack([digit1, digit2, digit3, digit4, digit5], axis=1)
        return length_logits, digits_logits

    def loss(length_logits, digits_logits, length_labels, digits_labels):
        length_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=length_labels, logits=length_logits))
        digit1_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 0], logits=digits_logits[:, 0, :]))
        digit2_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 1], logits=digits_logits[:, 1, :]))
        digit3_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 2], logits=digits_logits[:, 2, :]))
        digit4_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 3], logits=digits_logits[:, 3, :]))
        digit5_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 4], logits=digits_logits[:, 4, :]))
        loss = length_cross_entropy + digit1_cross_entropy + digit2_cross_entropy + digit3_cross_entropy + digit4_cross_entropy + digit5_cross_entropy
        return loss

