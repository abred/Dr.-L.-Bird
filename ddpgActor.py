from keras.applications.vgg16 import preprocess_input
import numpy as np
import os
import shutil
import sys
import time

import defineVGG

from tensorflow.python.framework import ops
import tensorflow as tf
import tensorflow.contrib.slim as slim

import math

import tfUtils as tfu


class Actor:
    O = 3
    H1 = 32
    H2 = 32
    H3 = 64
    H4 = 64
    H5 = 128
    H6 = 128
    H7 = 128
    H8 = 1024
    H9 = 1024
    Hconv1 = 32
    Hconv2 = 64
    Hconv3 = 128
    Hconv4 = 256
    HFC = [512, 512]
    HFCVGG = [2048, 2048]
    tau = 0.001
    train_dir = 'data'
    state_dim_x = 210
    state_dim_y = 120
    col_channels = 3
    actions_dim = 3
    vgg_state_dim = 224

    def __init__(self, sess, out_dir, params):
        self.sess = sess
        self.summaries = []
        self.useVGG = params['useVGG']
        self.batchnorm = params['batchnorm']
        self.dropout = params['dropout']
        self.top = params['top']
        self.learning_rate = params['learning-rateActor']
        self.weightDecay = params['weight-decayActor']
        self.momentum = params['momentumActor']
        self.opti = params['optimizerActor']
        self.stopGrad = params['stopGradActor']
        self.bnDecay = params['batchnorm-decay']
        if params['batchnorm']:
            # self.batchnorm = slim.batch_norm
            self.batchnorm = tfu.batch_normBUG
        else:
            self.batchnorm = None

        print("dropout actor", self.dropout)
        print("vgg actor", self.useVGG)
        print("top actor", self.top)

        if self.useVGG:
            varTmp = tf.get_variable("tmp", shape=[1,1], trainable=False)
            self.vggsaver = tf.train.import_meta_graph(
                '/home/s7550245/convNet/vgg-model.meta',
                import_scope="Actor/VGG",
                clear_devices=True)

            _VARSTORE_KEY = ("__variable_store",)
            varstore = ops.get_collection(_VARSTORE_KEY)[0]
            variables = tf.get_collection(
                ops.GraphKeys.GLOBAL_VARIABLES,
                scope="Actor/VGG")

            for v in variables:
                varstore._vars[v.name.split(":")[0]] = v

        with tf.variable_scope('Actor'):
            self.keep_prob = tf.placeholder(tf.float32)
            self.isTraining = tf.placeholder(tf.bool)

            # Actor Network
            if self.useVGG:
                self.setVGGNN()
            else:
                self.setNN()

            self.train_op = self.defineTraining()

        self.summary_op = tf.summary.merge(self.summaries)
        self.writer = tf.summary.FileWriter(out_dir, sess.graph)

    def setVGGNN(self):
        prevTrainVarCount = len(tf.trainable_variables())

        with tf.variable_scope("VGG") as scope:
            scope.reuse_variables()
            self.input_pl, self.nnB = self.defineNNVGG()
        with tf.variable_scope("VGG") as scope:
            self.nn = self.defineNNVGGTop(self.nnB, top=top)
        self.nn_params = tf.trainable_variables()[prevTrainVarCount:]

        # Target Network
        with tf.variable_scope('target'):
            prevTrainVarCount = len(tf.trainable_variables())
            self.target_input_pl, self.nnBT = self.defineNNVGG()
            self.target_nn = self.defineNNVGGTop(self.nnBT, top=top,
                                                 isTargetNN=True)
            self.target_nn_params = \
                tf.trainable_variables()[prevTrainVarCount:]
            with tf.variable_scope('init'):
                for i in range(len(self.target_nn_params)):
                    for j in range(len(self.nn_params)):
                        p1 = self.nn_params[j]
                        p2 = self.target_nn_params[i]
                        if p2.name.split("/")[-1] in p1.name:
                            tf.Variable.assign(
                                self.target_nn_params[i],
                                self.nn_params[j].initialized_value())
                            break
            self.target_nn_update_op = self.define_update_target_nn_op()

    def setNN(self):
        prevTrainVarCount = len(tf.trainable_variables())

        self.input_pl, self.nn = self.defineNN2()
        self.nn_params = tf.trainable_variables()[prevTrainVarCount:]

        # Target Network
        with tf.variable_scope('target'):
            prevTrainVarCount = len(tf.trainable_variables())
            self.target_input_pl, self.target_nn = \
                self.defineNN2(isTargetNN=True)
            self.target_nn_params = \
                tf.trainable_variables()[prevTrainVarCount:]
            with tf.variable_scope('init'):
                for i in range(len(self.nn_params)):
                    tf.Variable.assign(
                        self.target_nn_params[i],
                        self.nn_params[i].initialized_value())
            self.target_nn_update_op = self.define_update_target_nn_op()


    def defineNN2(self, isTargetNN=False):
        with tf.variable_scope('inf'):
            images = tf.placeholder(
                tf.float32,
                shape=[None,
                       self.state_dim_y,
                       self.state_dim_x,
                       self.col_channels],
                name='input')

            net = images
            with slim.arg_scope(
                [slim.fully_connected, slim.conv2d],
                activation_fn=tf.nn.relu,
                weights_initializer=tf.contrib.layers.xavier_initializer(
                    uniform=True),
                weights_regularizer=slim.l2_regularizer(self.weightDecay),
                biases_initializer=tf.contrib.layers.xavier_initializer(
                    uniform=True)):
                with slim.arg_scope([slim.conv2d], stride=1, padding='SAME'):
                    net = slim.repeat(net, 3, slim.conv2d, self.Hconv1,
                                      [3, 3], scope='conv1')
                    print(net)
                    net = slim.max_pool2d(net, [2, 2], scope='pool1')
                    print(net)
                    net = slim.repeat(net, 3, slim.conv2d, self.Hconv2,
                                      [3, 3], scope='conv2')
                    print(net)
                    net = slim.max_pool2d(net, [2, 2], scope='pool2')
                    print(net)

                    net = slim.repeat(net, 3, slim.conv2d, self.Hconv3,
                                      [3, 3], scope='conv3')
                    print(net)
                    net = slim.max_pool2d(net, [2, 2], scope='pool3')
                    print(net)
                    net = slim.repeat(net, 3, slim.conv2d, self.Hconv4,
                                      [3, 3], scope='conv4')
                    print(net)
                    net = slim.max_pool2d(net, [2, 2], scope='pool4')
                    print(net)


                remSzY = int(self.state_dim_y / 2**4)
                remSzX = int(self.state_dim_x / 2**4)
                net = tf.reshape(net, [-1, remSzY*remSzX*self.Hconv4],
                                 name='flatten')

                with slim.arg_scope([slim.fully_connected],
                                    normalizer_fn=self.batchnorm,
                                    normalizer_params={
                                        'fused': True,
                                        'is_training': self.isTraining,
                                        'updates_collections': None,
                                        'decay': self.bnDecay,
                                        'scale': True}):
                    for i in range(len(self.HFC)):
                        net = slim.fully_connected(net, self.HFC[i],
                                                   scope='fc' + str(i))
                        print(net)
                        if self.dropout:
                            net = slim.dropout(net,
                                               keep_prob=self.dropout,
                                               is_training=self.isTraining)
                            print(net)
                net = slim.fully_connected(net, self.O, activation_fn=None,
                                           scope='out')
                print(net)
                r, th, t = tf.split(1, 3, net)
                r_o = 50.0 * tf.sigmoid(r)
                th_o = 9000.0 * tf.sigmoid(th)
                t_o = 4000.0 * tf.sigmoid(t)
                if not isTargetNN:
                    self.summaries += [
                        tf.summary.histogram('out' + '/radius_action', r_o),
                        tf.summary.histogram('out' + '/theta_action', th_o),
                        tf.summary.histogram('out' +
                                             '/time_delay_action',t_o),
                        tf.summary.histogram('out' +
                                             '/radius_action_before_sig', r),
                        tf.summary.histogram('out' +
                                             '/theta_action_before_sig', th),
                        tf.summary.histogram('out' +
                                             '/time_delay_action_before_sig',t)
                    ]
                net = tf.sigmoid(net)
                # self.summaries += [tf.summary.histogram('output', net)]

        return images, net

    def defineNN(self, isTargetNN=False):
        images = tf.placeholder(
            tf.float32,
            shape=[None,
                   self.state_dim_y,
                   self.state_dim_x,
                   self.col_channels],
            name='input')

        h1, s = tfu.convReluLayer(images,
                                  self.col_channels, self.H1,
                                  scopeName='h1',
                                  isTargetNN=isTargetNN,
                                  is_training=self.isTraining)

        self.summaries += s
        h2, s = tfu.convReluPoolLayer(h1,
                                      self.H1, self.H2,
                                      scopeName='h2',
                                      isTargetNN=isTargetNN,
                                      is_training=self.isTraining)

        self.summaries += s
        h3, s = tfu.convReluLayer(h2,
                                  self.H2, self.H3,
                                  scopeName='h3',
                                  isTargetNN=isTargetNN,
                                  is_training=self.isTraining)

        self.summaries += s
        h4, s = tfu.convReluPoolLayer(h3,
                                      self.H3, self.H4,
                                      scopeName='h4',
                                      isTargetNN=isTargetNN,
                                      is_training=self.isTraining)

        self.summaries += s
        h5, s = tfu.convReluLayer(h4,
                                  self.H4, self.H5,
                                  scopeName='h5',
                                  isTargetNN=isTargetNN,
                                  is_training=self.isTraining)

        self.summaries += s
        h5b, s = tfu.convReluPoolLayer(h5,
                                      self.H5, self.H6,
                                      scopeName='h5b',
                                      isTargetNN=isTargetNN,
                                      is_training=self.isTraining)

        self.summaries += s
        h6, s = tfu.convReluLayer(h5b,
                                  self.H5, self.H6,
                                  scopeName='h6',
                                  isTargetNN=isTargetNN,
                                  is_training=self.isTraining)

        self.summaries += s
        h7, s = tfu.convReluPoolLayer(h6,
                                      self.H6, self.H7,
                                      scopeName='h7',
                                      isTargetNN=isTargetNN,
                                      is_training=self.isTraining)

        self.summaries += s
        h7_f = tf.reshape(h7, [-1, 13*7*self.H7], name='flatten')

        h8, s = tfu.fullyConRelu(h7_f,
                                 13*7*self.H7, self.H8,
                                 scopeName='h8', isTargetNN=isTargetNN,
                                 is_training=self.isTraining)
        self.summaries += s
        h9, s = tfu.fullyConRelu(h8,
                                 self.H8, self.H9,
                                 scopeName='h9', isTargetNN=isTargetNN,
                                 is_training=self.isTraining)
        self.summaries += s
        o, s = tfu.fullyCon(h9, self.H9, self.O,
                            scopeName='out', isTargetNN=isTargetNN,
                            is_training=self.isTraining)
        self.summaries += s
        r, th, t = tf.split(1, 3, o)
        r = tf.Print(r, [r], "r:")
        th = tf.Print(th, [th], "th:")
        t = tf.Print(t, [t], "t:")
        r_o = 50.0 * tf.sigmoid(r)
        th_o = 9000.0 * tf.sigmoid(th)
        t_o = 4000.0 * tf.sigmoid(t)
        if not isTargetNN:
            self.summaries += [
                tf.summary.histogram(tf.get_default_graph().unique_name(
                    'out' + '/radius_action',
                    mark_as_used=False), r_o),
                tf.summary.histogram(tf.get_default_graph().unique_name(
                    'out' + '/theta_action',
                    mark_as_used=False), th_o),
                tf.summary.histogram(tf.get_default_graph().unique_name(
                    'out' + '/time_delay_action',
                    mark_as_used=False), t_o),
                tf.summary.histogram(tf.get_default_graph().unique_name(
                    'out' + '/radius_action_before_sig',
                    mark_as_used=False), r),
                tf.summary.histogram(tf.get_default_graph().unique_name(
                    'out' + '/theta_action_before_sig',
                    mark_as_used=False), th),
                tf.summary.histogram(tf.get_default_graph().unique_name(
                    'out' + '/time_delay_action_before_sig',
                    mark_as_used=False), t)
            ]
        outputs = tf.sigmoid(o)
        return images, outputs

    def defineNNVGG(self):
        inputs = tf.placeholder(
            tf.float32,
            shape=[None,
                   self.vgg_state_dim,
                   self.vgg_state_dim,
                   self.col_channels],
            name='input')
        return inputs, defineVGG.defineNNVGG(inputs,
                                             self.vgg_state_dim, self.top)


    def defineNNVGGTop(self, bottom, top=13, isTargetNN=False):
        with tf.variable_scope('top') as scope:
            print(bottom)
            if top == 1:
                numPool = 0
                self.HFC1VGG = 256
                self.HFC2VGG = 256
            if top <= 2:
                numPool = 1
                H = 64
                self.HFC1VGG = 256
                self.HFC2VGG = 256
            elif top <= 4:
                numPool = 2
                H = 128
                self.HFC1VGG = 512
                self.HFC2VGG = 512
            elif top <= 7:
                numPool = 3
                H = 256
                self.HFC1VGG = 1024
                self.HFC2VGG = 1024
            elif top <= 10:
                numPool = 4
                H = 512
                self.HFC1VGG = 2048
                self.HFC2VGG = 2048
            elif top <= 13:
                numPool = 5
                H = 512
                self.HFC1VGG = 2048
                self.HFC2VGG = 2048
            remSz = int(self.vgg_state_dim / 2**numPool)
            h13_f = tf.reshape(bottom, [-1, remSz*remSz*H],
                               name='flatten')

            # print(h13_f)
            if self.dropout:
                h14, s = tfu.fullyConReluDrop(h13_f, remSz*remSz*H,
                                              self.HFC1VGG,
                                              self.keep_prob, scopeName='hfc1',
                                              norm=self.batchnorm,
                                              isTargetNN=isTargetNN,
                                              is_training=self.isTraining)
            else:
                h14, s = tfu.fullyConRelu(h13_f, remSz*remSz*H,
                                          self.HFC1VGG,
                                          scopeName='hfc1',
                                          norm=self.batchnorm,
                                          isTargetNN=isTargetNN,
                                          is_training=self.isTraining)
            self.summaries += s

            if self.dropout:
                h15, s = tfu.fullyConReluDrop(h14, self.HFC1VGG, self.HFC2VGG,
                                              self.keep_prob, scopeName='hfc2',
                                              norm=self.batchnorm,
                                              isTargetNN=isTargetNN,
                                              is_training=self.isTraining)
            else:
                h15, s = tfu.fullyConRelu(h14, self.HFC1VGG, self.HFC2VGG,
                                          scopeName='hfc2',
                                          norm=self.batchnorm,
                                          isTargetNN=isTargetNN,
                                          is_training=self.isTraining)
            self.summaries += s

            o, s = tfu.fullyCon(h15, self.HFC2VGG, self.O, scopeName='out',
                                isTargetNN=isTargetNN,
                                is_training=self.isTraining)
            self.summaries += s
            r, th, t = tf.split(1, 3, o)
            r_o = 50.0 * tf.sigmoid(r)
            th_o = 9000.0 * tf.sigmoid(th)
            t_o = 4000.0 * tf.sigmoid(t)
            if not isTargetNN:
                self.summaries += [
                    tf.summary.histogram(tf.get_default_graph().unique_name(
                        'out' + '/radius_action',
                        mark_as_used=False), r_o),
                    tf.summary.histogram(tf.get_default_graph().unique_name(
                        'out' + '/theta_action',
                        mark_as_used=False), th_o),
                    tf.summary.histogram(tf.get_default_graph().unique_name(
                        'out' + '/time_delay_action',
                        mark_as_used=False), t_o),
                    tf.summary.histogram(tf.get_default_graph().unique_name(
                        'out' + '/radius_action_before_sig',
                        mark_as_used=False), r),
                    tf.summary.histogram(tf.get_default_graph().unique_name(
                        'out' + '/theta_action_before_sig',
                        mark_as_used=False), th),
                    tf.summary.histogram(tf.get_default_graph().unique_name(
                        'out' + '/time_delay_action_before_sig',
                        mark_as_used=False), t)
                ]
            outputs = tf.sigmoid(o)
            return outputs

    def define_update_target_nn_op(self):
        with tf.variable_scope('update'):
            tau = tf.constant(self.tau, name='tau')
            invtau = tf.constant(1.0-self.tau, name='invtau')
            if self.useVGG:
                tmp = []
                for i in range(len(self.target_nn_params)):
                    for j in range(len(self.nn_params)):
                        p1 = self.nn_params[j]
                        p2 = self.target_nn_params[i]
                        if p2.name.split("/")[-1] in p1.name:
                            print(p1.name, p2.name)
                            tmp.append(self.target_nn_params[i].assign(
                                tf.mul(self.nn_params[j], tau) +
                                tf.mul(self.target_nn_params[i], invtau)))
                            break
                return tmp
            else:
                return \
                    [self.target_nn_params[i].assign(
                        tf.mul(self.nn_params[i], tau) +
                        tf.mul(self.target_nn_params[i], invtau))
                     for i in range(len(self.target_nn_params))]


    def defineTraining(self):
        with tf.variable_scope('train'):
            self.critic_actions_gradient_pl = tf.placeholder(
                tf.float32,
                [None, self.actions_dim],
                name='CriticActionsGradient')

            if self.useVGG and self.stopGrad is not None:
                # variables = tf.get_collection(
                #     ops.GraphKeys.TRAINABLE_VARIABLES,
                #     scope="Actor")
                var_list = []
                for v in self.nn_params:
                    if "block" not in v.name:
                        var_list.append(v)
                        continue

                    if self.stopGrad == 0:
                        var_list.append(v)
                    if self.stopGrad <= 1  and "block1_conv1" in v.name:
                        var_list.append(v)
                    if self.stopGrad <= 2  and "block1_conv2" in v.name:
                        var_list.append(v)
                    if self.stopGrad <= 3  and "block2_conv1" in v.name:
                        var_list.append(v)
                    if self.stopGrad <= 4  and "block2_conv2" in v.name:
                        var_list.append(v)
                    if self.stopGrad <= 5  and "block3_conv1" in v.name:
                        var_list.append(v)
                    if self.stopGrad <= 6  and "block3_conv2" in v.name:
                        var_list.append(v)
                    if self.stopGrad <= 7  and "block3_conv3" in v.name:
                        var_list.append(v)
                    if self.stopGrad <= 8  and "block4_conv1" in v.name:
                        var_list.append(v)
                    if self.stopGrad <= 9  and "block4_conv2" in v.name:
                        var_list.append(v)
                    if self.stopGrad <= 10 and "block4_conv3" in v.name:
                        var_list.append(v)
                    if self.stopGrad <= 11 and "block5_conv1" in v.name:
                        var_list.append(v)
                    if self.stopGrad <= 12 and "block5_conv2" in v.name:
                        var_list.append(v)
                    if self.stopGrad <= 13 and "block5_conv3" in v.name:
                        var_list.append(v)

                for v in var_list:
                    print(v.name)
            else:
                var_list = self.nn_params

            self.actor_gradients = tf.gradients(
                self.nn,
                self.nn_params,
                # critic grad descent
                # here ascent -> negative
                -self.critic_actions_gradient_pl)

        if self.opti == 'momentum':
            optimizer = tf.train.MomentumOptimizer(self.learning_rate,
                                                   self.momentum)
        elif self.opti == 'adam':
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
        elif self.opti == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(
                self.learning_rate)

        return optimizer.apply_gradients(zip(self.actor_gradients, var_list))

    def run_train(self, inputs, a_grad, step):
        if (step+1) % 1 == 0:
            _, summaries = self.sess.run([self.train_op,
                                          self.summary_op],
                                         feed_dict={
                self.input_pl: inputs,
                self.critic_actions_gradient_pl: a_grad,
                self.isTraining: True,
                self.keep_prob: self.dropout
            })
            self.writer.add_summary(summaries, step)
            self.writer.flush()
        else:
            self.sess.run([self.train_op],
                          feed_dict={
                self.input_pl: inputs,
                self.critic_actions_gradient_pl: a_grad,
                self.isTraining: True,
                self.keep_prob: self.dropout
            })

    def run_predict(self, inputs):
        return self.sess.run(self.nn, feed_dict={
            self.input_pl: inputs,
            self.isTraining: False,
            self.keep_prob: 1.0
        })

    def run_predict_target(self, inputs):
        return self.sess.run(self.target_nn, feed_dict={
            self.target_input_pl: inputs,
            self.isTraining: False,
            self.keep_prob: 1.0
        })

    def run_update_target_nn(self):
        self.sess.run(self.target_nn_update_op)
