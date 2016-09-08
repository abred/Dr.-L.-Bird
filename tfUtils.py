import tensorflow as tf


def weight_variable(shape, name):
    return tf.get_variable(
        name, shape=shape,
        initializer=tf.contrib.layers.xavier_initializer())


def weight_variable_unit(shape, name, minV=-0.0003, maxV=0.0003):
    return tf.get_variable(
        name, shape=shape,
        initializer=tf.random_uniform_initializer(minval=minV, maxval=maxV))


def weight_variable_conv(shape, name):
    return tf.get_variable(
        name, shape=shape,
        initializer=tf.contrib.layers.xavier_initializer_conv2d())


def bias_variable(shape, name):
    return tf.Variable(tf.constant(0.1, shape=shape), name=name)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def convReluPoolLayer(inputs, inC, outC, fh=3, fw=3, scopeName=None):
    with tf.variable_scope(scopeName):

        weights = weight_variable_conv([fh, fw, inC, outC],
                                       'w')
        biases = bias_variable([outC], 'b')
        conv = tf.nn.relu(conv2d(inputs, weights) + biases)
        pool = max_pool_2x2(conv)
        return pool


def fullyConReluDrop(inputs, inC, outC, scopeName=None):
    with tf.variable_scope(scopeName):
        weights = weight_variable([inC, outC], 'w')
        biases = bias_variable([outC], 'b')
        fc = tf.nn.relu(tf.matmul(inputs, weights) + biases)
        # keepprob = tf.placeholder(tf.float32)
        drop = tf.nn.dropout(fc, 0.5)
        return drop


def mean_squared_diff(x, y):
    return tf.reduce_mean(tf.squared_difference(x, y))
