import math

import tensorflow as tf


def weight_variable(shape, name):
    var = tf.get_variable(
        name, shape=shape,
        initializer=tf.contrib.layers.xavier_initializer(uniform=True))
    summaries = variable_summaries(var, name)
    return var, summaries


def weight_variable_unit(shape, name, minV=-0.0003, maxV=0.0003):
    var = tf.get_variable(
        name, shape=shape,
        initializer=tf.random_uniform_initializer(minval=minV, maxval=maxV))
    summaries = variable_summaries(var, name)
    return var, summaries


def weight_variable_conv(shape, name):
    var = tf.get_variable(
        name, shape=shape,
        initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=True))
    summaries = variable_summaries(var, name)
    return var, summaries


def bias_variable(shape, name, minV=None, maxV=None):
    if minV is None:
        minV = - 1.0 / math.sqrt(shape[0])
    if maxV is None:
        maxV = 1.0 / math.sqrt(shape[0])
    var = tf.get_variable(
        name, shape=shape,
        initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=True))
        # initializer=tf.random_uniform_initializer(minval=minV,
                                                  # maxval=maxV))
    summaries = variable_summaries(var, name)
    return var, summaries


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    mean = tf.reduce_mean(var)
    s1 = tf.scalar_summary(
        tf.get_default_graph().unique_name(name + '/mean',
                                           mark_as_used=False), mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    s2 = tf.scalar_summary(
        tf.get_default_graph().unique_name(name + '/stddev',
                                           mark_as_used=False), stddev)
    s3 = tf.scalar_summary(
        tf.get_default_graph().unique_name(name + '/max',
                                           mark_as_used=False),
        tf.reduce_max(var))
    s4 = tf.scalar_summary(
        tf.get_default_graph().unique_name(name + '/min',
                                           mark_as_used=False),
        tf.reduce_min(var))
    s5 = tf.histogram_summary(
        tf.get_default_graph().unique_name(name + '/histo',
                                           mark_as_used=False), var)
    return [s1, s2, s3, s4, s5]


def conv2d(x, W, strh=1, strw=1):
    return tf.nn.conv2d(x, W, strides=[1, strh, strw, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='VALID')


def convReluPoolLayer(inputs, inC, outC, fh=3, fw=3,
                      scopeName=None, isTargetNN=False,
                      strh=1, strw=1,
                      is_training=None):
    with tf.variable_scope(scopeName):

        weights, sw = weight_variable_conv([fh, fw, inC, outC], 'w')
        biases, sb = bias_variable([outC], 'b')
        preactivate = conv2d(inputs, weights, strh, strw) + biases
        if isTargetNN:
            conv = tf.nn.relu(preactivate)
            pool = max_pool_2x2(conv)
            pool_n, _ = batch_norm(pool, is_training=is_training,
                                   scopeName=scopeName, isTargetNN=isTargetNN)

            return pool_n, sw + sb
        else:
            # preactivate = conv2d(inputs, weights)
            s1 = tf.histogram_summary(
                tf.get_default_graph().unique_name(
                    scopeName + '/pre_activation',
                    mark_as_used=False), preactivate)
            conv = tf.nn.relu(preactivate)
            s2 = tf.histogram_summary(
                tf.get_default_graph().unique_name(scopeName + '/activations',
                                                   mark_as_used=False), conv)
            pool = max_pool_2x2(conv)
            pool_n, s3 = batch_norm(pool, is_training=is_training,
                                    scopeName=scopeName, isTargetNN=isTargetNN)

            return pool_n, sw + sb + [s1] + [s2] + [s3]


def convReluLayer(inputs, inC, outC, fh=3, fw=3,
                  scopeName=None, isTargetNN=False,
                  strh=1, strw=1,
                  is_training=None):
    with tf.variable_scope(scopeName):

        weights, sw = weight_variable_conv([fh, fw, inC, outC], 'w')
        biases, sb = bias_variable([outC], 'b')
        preactivate = conv2d(inputs, weights, strh, strw) + biases
        # preactivate = conv2d(inputs, weights)
        if isTargetNN:
            conv = tf.nn.relu(preactivate)
            conv_n, _ = batch_norm(conv, is_training=is_training,
                                   scopeName=scopeName, isTargetNN=isTargetNN)
            return conv_n, sw + sb
        else:
            s1 = tf.histogram_summary(
                tf.get_default_graph().unique_name(
                    scopeName + '/pre_activation',
                    mark_as_used=False), preactivate)
            conv = tf.nn.relu(preactivate)
            s2 = tf.histogram_summary(
                tf.get_default_graph().unique_name(scopeName + '/activations',
                                                   mark_as_used=False), conv)
            conv_n, s3 = batch_norm(conv, is_training=is_training,
                                    scopeName=scopeName, isTargetNN=isTargetNN)
            return conv_n, sw + sb + [s1] + [s2] + [s3]


def fullyConReluDrop(inputs, inC, outC, keep_prob,
                     scopeName=None, isTargetNN=False,
                     is_training=None):
    with tf.variable_scope(scopeName):
        weights, sw = weight_variable([inC, outC], 'w')
        biases, sb = bias_variable([outC], 'b')
        preactivate = tf.matmul(inputs, weights) + biases
        # preactivate = tf.matmul(inputs, weights)
        if isTargetNN:
            fc = tf.nn.relu(preactivate)
            fc_n, _ = batch_norm(fc, is_training=is_training,
                                 scopeName=scopeName, isTargetNN=isTargetNN,
                                 outC=outC)

            drop = tf.nn.dropout(fc_n, keep_prob)
            return drop, sw + sb
        else:
            s1 = tf.histogram_summary(
                tf.get_default_graph().unique_name(
                    scopeName + '/pre_activation',
                    mark_as_used=False), preactivate)
            fc = tf.nn.relu(preactivate)
            s2 = tf.histogram_summary(
                tf.get_default_graph().unique_name(scopeName + '/activations',
                                                   mark_as_used=False), fc)
            # keepprob = tf.placeholder(tf.float32)
            fc_n, s3 = batch_norm(fc, is_training=is_training,
                                  scopeName=scopeName, isTargetNN=isTargetNN,
                                  outC=outC)

            drop = tf.nn.dropout(fc_n, keep_prob)
            return drop, sw + sb + [s1] + [s2] + [s3]


def fullyConRelu(inputs, inC, outC, scopeName=None, isTargetNN=False,
                 is_training=None):
    with tf.variable_scope(scopeName):
        weights, sw = weight_variable([inC, outC], 'w')
        biases, sb = bias_variable([outC], 'b')
        preactivate = tf.matmul(inputs, weights) + biases
        # preactivate = tf.matmul(inputs, weights)
        if isTargetNN:
            fc = tf.nn.relu(preactivate)
            fc_n, _ = batch_norm(fc, is_training=is_training,
                                 scopeName=scopeName, isTargetNN=isTargetNN,
                                 outC=outC)
            return fc_n, sw + sb
        else:
            s1 = tf.histogram_summary(
                tf.get_default_graph().unique_name(
                    scopeName + '/pre_activation',
                    mark_as_used=False), preactivate)
            fc = tf.nn.relu(preactivate)
            s2 = tf.histogram_summary(
                tf.get_default_graph().unique_name(scopeName + '/activations',
                                                   mark_as_used=False), fc)
            fc_n, s3 = batch_norm(fc, is_training=is_training,
                                  scopeName=scopeName, isTargetNN=isTargetNN,
                                  outC=outC)
            return fc_n, sw + sb + [s1] + [s2] + [s3]


def fullyCon(inputs, inC, outC, scopeName=None, isTargetNN=False,
             is_training=None):
    with tf.variable_scope(scopeName):
        weights, sw = weight_variable([inC, outC], 'w')
        biases, sb = bias_variable([outC], 'b')
        fc = tf.matmul(inputs, weights) + biases
        if isTargetNN:
            return fc, sw + sb
        else:
            s1 = tf.histogram_summary(
            tf.get_default_graph().unique_name(scopeName + '/outut',
                                               mark_as_used=False), fc)
            return fc, sw + sb + [s1]


def mean_squared_diff(x, y):
    return tf.reduce_mean(tf.squared_difference(x, y))


def batch_norm(inputs,
               scopeName=None,
               decay=0.999,
               center=True,
               scale=True,
               epsilon=0.001,
               updates_collections=None,
               is_training=True,
               isTargetNN=False,
               outC=None):
    original_shape = inputs.get_shape()
    original_rank = original_shape.ndims
    if original_rank == 2:
        inputs = tf.reshape(inputs, [-1, 1, 1, outC], name="bnreshapebugi")
    bn = tf.contrib.layers.batch_norm(inputs, decay=decay, center=center,
                                      scale=scale, scope=scopeName,
                                      epsilon=epsilon,
                                      updates_collections=updates_collections,
                                      variables_collections=['batchnorm'],
                                      is_training=is_training,
                                      fused=True)
    if original_rank == 2:
        bn = tf.reshape(bn, [-1, outC], name="bnreshapebugo")

    if not isTargetNN:
        s = tf.histogram_summary(
            tf.get_default_graph().unique_name(scopeName + '/normalized',
                                               mark_as_used=False), bn,
            name="batchnormtest")
        return bn, s
    else:
        return bn, []
