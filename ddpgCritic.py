import numpy as np
import os
import shutil
import sys
import time

from tensorflow.python.framework import ops
import tensorflow as tf
import tensorflow.contrib.slim as slim

import tfUtils as tfu


class Critic:
    O = 1

    Hconv1 = 16
    Hconv2 = 32
    Hconv3 = 32
    Hconv4 = 48
    HFC = [96, 96]
    HFCVGG = [2048, 2048]

    tau = 0.001
    train_dir = 'data'
    state_dim_x = 210
    state_dim_y = 120
    col_channels = 3
    actions_dim = 3
    vgg_state_dim = 224

    def __init__(self, sess, out_dir, glStep, params,
                 inputs=None):
        self.sess = sess
        self.params = params
        self.summaries = []
        self.weight_summaries = []
        self.out_dir = out_dir
        if params['batchnorm']:
            self.batchnorm = slim.batch_norm
            # self.batchnorm = tfu.batch_normBUG
        else:
            self.batchnorm = None

        self.dropout = params['dropout']
        self.weightDecay = params['weight-decayCritic']
        self.learning_rate = params['learning-rateCritic']
        self.momentum = params['momentumCritic']
        self.opti = params['optimizerCritic']
        self.bnDecay = params['batchnorm-decay']
        self.useVGG = params['useVGG']

        if params['state_dim'] is not None:
            self.state_dim = params['state_dim']
        if params['col_channels'] is not None:
            self.col_channels = params['col_channels']

        if params['useVGG']:
            self.top = params['top']
            if self.top == 7:
                self.pretrained = sess.graph.get_tensor_by_name(
                    "VGG/MaxPool_2:0")
            elif self.top == 10:
                self.pretrained = sess.graph.get_tensor_by_name(
                    "VGG/MaxPool_3:0")
            elif self.top == 13:
                self.pretrained = sess.graph.get_tensor_by_name(
                    "VGG/MaxPool_4:0")
            print(self.pretrained)
            self.pretrained = tf.stop_gradient(self.pretrained)
            self.images = inputs

        print("dropout critic", self.dropout)

        self.global_step = glStep

        with tf.variable_scope('Critic'):
            self.keep_prob = tf.placeholder(tf.float32)
            self.isTraining = tf.placeholder(tf.bool, [])

            # Critic Network
            self.setNN()

            self.loss_op = self.define_loss()
            self.train_op = self.defineTraining()



        _VARSTORE_KEY = ("__variable_store",)
        varstore = ops.get_collection(_VARSTORE_KEY)[0]
        variables = tf.get_collection(
            ops.GraphKeys.GLOBAL_VARIABLES,
            scope="Critic")

        for v in variables:
            print(v.name)
            if v.name.startswith("Critic/inf/fc0/weights"):
                self.testw = v
            if v.name == "Critic/inf/fc0/BatchNorm/beta:0":
                self.fc0bnb = v
            if v.name == "Critic/inf/fc0/BatchNorm/gamma:0":
                self.fc0bng = v
            if v.name == "Critic/inf/fc0/BatchNorm/moving_mean:0":
                self.fc0bnmm = v
            if v.name == "Critic/inf/fc0/BatchNorm/moving_variance:0":
                self.fc0bnmv = v
            if v.name == "Critic/inf/fc1/BatchNorm/beta:0":
                self.fc1bnb = v
            if v.name == "Critic/inf/fc1/BatchNorm/gamma:0":
                self.fc1bng = v
            if v.name == "Critic/inf/fc1/BatchNorm/moving_mean:0":
                self.fc1bnmm = v
            if v.name == "Critic/inf/fc1/BatchNorm/moving_variance:0":
                self.fc1bnmv = v
            if v.name.endswith("weights:0") or \
               v.name.endswith("biases:0"):
                s = []
                var = v
                mean = tf.reduce_mean(var)
                s.append(tf.summary.scalar(v.name[:-2]+'mean', mean))
                with tf.name_scope('stddev'):
                    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                s.append(tf.summary.scalar(v.name[:-2]+'stddev', stddev))
                s.append(tf.summary.scalar(v.name[:-2]+'max',
                                           tf.reduce_max(var)))
                s.append(tf.summary.scalar(v.name[:-2]+'min',
                                           tf.reduce_min(var)))
                s.append(tf.summary.histogram(v.name[:-2]+'histogram', var))

                self.weight_summaries += s

        # self.action_grads = self.define_action_grad()
        self.grads = self.define_grads()

        self.summary_op = tf.summary.merge(self.summaries)
        self.weight_summary_op = tf.summary.merge(self.weight_summaries)
        self.writer = tf.summary.FileWriter(out_dir, sess.graph)

    def setNN(self):
        prevTrainVarCount = len(tf.trainable_variables())

        if self.useVGG:
            defNN = self.defineNNVGG
            self.input_pl = self.images
        else:
            defNN = self.defineNN
            self.input_pl = tf.placeholder(
                tf.float32,
                shape=[None,
                       self.state_dim_y,
                       self.state_dim_x,
                       self.col_channels],
                name='input')

        self.actions_pl = tf.placeholder(tf.float32,
                                         shape=[None, self.actions_dim],
                                         name='ActorActions')

        self.nn = defNN()
        self.nn_params = tf.trainable_variables()[prevTrainVarCount:]

        # Target Network
        with tf.variable_scope('target'):
            prevTrainVarCount = len(tf.trainable_variables())
            self.target_nn =\
                defNN(isTargetNN=True)
            self.target_nn_params = \
                tf.trainable_variables()[prevTrainVarCount:]
            with tf.variable_scope('init'):
                for i in range(len(self.nn_params)):
                    tf.Variable.assign(
                        self.target_nn_params[i],
                        self.nn_params[i].initialized_value())
            self.target_nn_update_op = self.define_update_target_nn_op()

    def defineNNVGG(self, isTargetNN=False):
        with tf.variable_scope('inf'):
            print("defining vgg net")
            net = self.pretrained

            with slim.arg_scope(
                [slim.fully_connected],
                activation_fn=tf.nn.relu,
                weights_initializer=\
                    tf.contrib.layers.variance_scaling_initializer(
                        factor=1.0, mode='FAN_AVG', uniform=True),
                weights_regularizer=slim.l2_regularizer(self.weightDecay),
                biases_initializer=\
                    tf.contrib.layers.variance_scaling_initializer(
                        factor=1.0, mode='FAN_AVG', uniform=True),
                normalizer_fn=self.batchnorm,
                normalizer_params={
                    'fused': True,
                    'is_training': self.isTraining,
                    'updates_collections': None,
                    'decay': self.bnDecay,
                    'scale': True}
            ):

                if self.top == 1:
                    numPool = 0
                    self.HFCVGG = [256, 256]
                    H = 64
                elif self.top <= 2:
                    numPool = 1
                    H = 64
                    self.HFCVGG = [256, 256]
                elif self.top <= 4:
                    numPool = 2
                    H = 128
                    self.HFCVGG = [512, 512]
                elif self.top <= 7:
                    numPool = 3
                    H = 256
                    self.HFCVGG = [1024, 1024]
                elif self.top <= 10:
                    numPool = 4
                    H = 512
                    self.HFCVGG = [2048, 2048]
                elif self.top <= 13:
                    numPool = 5
                    H = 512
                    self.HFCVGG = [2048, 2048]

                remSzY = int(self.vgg_state_dim / 2**numPool)
                remSzX = int(self.vgg_state_dim / 2**numPool)
                print(remSzY, remSzX, net, H)

                net = tf.concat(
                    [tf.reshape(net, [-1, remSzX*remSzY*H],
                                name='flatten'),
                     self.actions_pl], 1)
                # net = tf.concat(1,
                #     [tf.reshape(net, [-1, remSzX*remSzY*H],
                #                 name='flatten'),
                #      actions])

                for i in range(len(self.HFC)):
                    net = slim.fully_connected(net, 512,
                                               scope='fc' + str(i))
                    print(net)
                    if self.dropout:
                        net = slim.dropout(net,
                                           keep_prob=self.dropout,
                                           is_training=self.isTraining)
                        print(net)
                net = slim.fully_connected(net, self.O, activation_fn=None,
                                           scope='out')
                net = tf.Print(net, [net], "outputCritic:", summarize=1000,
                               first_n=10)
                if not isTargetNN:
                    self.weight_summaries += [tf.summary.histogram('outputCr', net)]

                print(net)
        return net

    def defineNN(self, isTargetNN=False):
        with tf.variable_scope('inf'):
            # actions = tf.placeholder(tf.float32,
            #                          shape=[None, self.actions_dim],
            #                          name='ActorActions')

            net = self.input_pl
            with slim.arg_scope(
                [slim.fully_connected, slim.conv2d],
                activation_fn=tf.nn.relu,
                weights_initializer=\
                    tf.contrib.layers.variance_scaling_initializer(
                        factor=1.0, mode='FAN_AVG', uniform=True),
                weights_regularizer=slim.l2_regularizer(self.weightDecay),
                biases_initializer=\
                    tf.contrib.layers.variance_scaling_initializer(
                        factor=1.0, mode='FAN_AVG', uniform=True)):
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
                net = tf.reshape(net, [-1, remSzX*remSzY*self.Hconv4],
                                 name='flatten')
                print(net)
                net = tf.concat(
                    [net,
                     self.actions_pl], 1)
                print(net)
                self.net2 = net
                # net = tf.concat(1,
                #     [tf.reshape(net, [-1, remSzX*remSzY*self.Hconv4],
                #                 name='flatten'),
                #      actions])

                with slim.arg_scope([slim.fully_connected],
                                    normalizer_fn=self.batchnorm,
                                    normalizer_params={
                                        'fused': True,
                                        'decay': self.bnDecay,
                                        'updates_collections': None,
                                        'is_training': self.isTraining,
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
                net = slim.fully_connected(net, 1, activation_fn=None,
                                           scope='out')
                net = tf.Print(net, [net], "outputCritic:", summarize=1000,
                               first_n=10)
                print(net)
                # net = tf.sigmoid(net)
                if not isTargetNN:
                    self.weight_summaries += [tf.summary.histogram('output', net)]

        return net

    def define_update_target_nn_op(self):
        with tf.variable_scope('update'):
            tau = tf.constant(self.tau, name='tau')
            invtau = tf.constant(1.0-self.tau, name='invtau')
            return \
                [self.target_nn_params[i].assign(
                    tf.multiply(self.nn_params[i], tau) +
                    tf.multiply(self.target_nn_params[i], invtau))
                 for i in range(len(self.target_nn_params))]
            # return \
            #     [self.target_nn_params[i].assign(
            #         tf.mul(self.nn_params[i], tau) +
            #         tf.mul(self.target_nn_params[i], invtau))
            #      for i in range(len(self.target_nn_params))]

    def define_loss(self):
        with tf.variable_scope('lossCritic'):
            self.td_targets_pl = tf.placeholder(tf.float32, [None, 1],
                                             name='tdTargets')
            in1 = tf.Print(self.td_targets_pl, [self.td_targets_pl],
                           "targetsCritic ", first_n=15, summarize=10)
            in2 = tf.Print(self.nn, [self.nn],
                           "nnCritic ", first_n=15, summarize=100)
            self.delta = in1 - in2
            # lossL2 = slim.losses.mean_squared_error(in1, in2)
            # lossL2 = tfu.mean_squared_diff(self.td_targets_pl, self.nn)
            # Huber loss
            lossL2 = tf.where(tf.abs(self.delta) < 1.0,
                              0.5 * tf.square(self.delta),
                              tf.abs(self.delta) - 0.5, name='clipped_error')
            # lossL2 = tf.select(tf.abs(self.delta) < 1.0,
            #                    0.5 * tf.square(self.delta),
            #                    tf.abs(self.delta) - 0.5, name='clipped_error')
            lossL2 = tf.reduce_mean(lossL2)
            # lossL2 = slim.losses.mean_squared_error(self.td_targets_pl,
                                                    # self.nn)
            lossL2 = tf.Print(lossL2, [lossL2], "lossL2Critic ", first_n=10)

            with tf.name_scope(''):
                self.summaries += [
                    tf.summary.scalar('mean_squared_diff_lossCritic',
                                      lossL2)]
            regs = []
            for v in self.nn_params:
                if "w" in v.name:
                    regs.append(tf.nn.l2_loss(v))
            lossReg = tf.add_n(regs) * self.weightDecay
            lossReg = tf.Print(lossReg, [lossReg], "regLossCritic ",
                               first_n=10)
            with tf.name_scope(''):
                self.summaries += [
                    tf.summary.scalar('mean_squared_diff_loss_regCritic',
                                      lossReg)]

            loss = lossL2 + lossReg
            with tf.name_scope(''):
                self.summaries += [
                    tf.summary.scalar('mean_squared_diff_loss_with_regCritic',
                                      loss)]

        return loss

    def defineTraining(self, conv=False):
        with tf.variable_scope('train'):
            if self.opti == 'momentum':
                optimizer = tf.train.MomentumOptimizer(self.learning_rate,
                                                       self.momentum)
            elif self.opti == 'adam':
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
            elif self.opti == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(
                    self.learning_rate)

            print(optimizer)
            grads_and_vars = optimizer.compute_gradients(self.loss_op)
            grads = []
            vars = []
            for (g,v) in grads_and_vars:
                vars.append(v)
                if g is None:
                    grads.append(g)
                else:
                    grads.append(tf.Print(g, [g], v.name+"critic grads ", first_n=10))
            return optimizer.apply_gradients(zip(grads, vars),
                                             global_step=self.global_step)
            # return optimizer.minimize(self.loss_op,
                                      # global_step=self.global_step)

    # def define_action_grad(self):
    #     with tf.variable_scope('getActionGradient'):
    #         print(self.nn)
    #         print(self.testw)
    #         if self.batchnorm is None:
    #             nn = tf.Print(self.nn, [self.nn],"getactiongradsOut", first_n=10, summarize=10)
    #         else:
    #             nn = tf.Print(self.nn, [self.fc0bnb,
    #                                 self.fc0bng,
    #                                 self.fc0bnmm,
    #                                 self.fc0bnmv,
    #                                 self.fc1bnb,
    #                                 self.fc1bng,
    #                                 self.fc1bnmm,
    #                                 self.fc1bnmv,
    #                                 self.nn], "getactiongradsOut", first_n=10, summarize=10)
    #         ac = tf.Print(self.testw, [self.testw], "getactiongradAct", first_n=10)
    #         gd = tf.gradients(nn, self.actions_pl)
    #         # print(tf.trainable_variables())
    #         grads = []
    #         for i in range(len(gd)):
    #             print(gd[i])
    #             grads.append(tf.Print(gd[i], [gd[i]], "actiongrads", first_n=10))
    #             # print(nn, ac, gd[i])
    #         return grads[0]
    #         # return tf.Print(gd, [gd], "actiongrads", first_n=10)
    #         # return tf.gradients(self.nn, self.actions_pl)[0]

    def define_grads(self):
        with tf.variable_scope('getGrads'):
            optimizer = tf.train.AdamOptimizer(self.learning_rate, epsilon=0.1)
            return optimizer.compute_gradients(self.loss_op,
                                               var_list=self.nn_params)

    def run_train(self, inputs, actions, targets):
        step = self.sess.run(self.global_step)
        wSum = 300
        lSum = 20
        if (step+1) % wSum == 0:
            out, delta, loss, _, summaries = self.sess.run(
                [self.nn,
                 self.delta,
                 self.loss_op,
                 self.train_op,
                 self.weight_summary_op],
                feed_dict={
                    self.input_pl: inputs,
                    self.actions_pl: actions,
                    self.td_targets_pl: targets,
                    self.isTraining: True,
                    self.keep_prob: self.dropout
                })
            self.writer.add_summary(summaries, step)
            self.writer.flush()
        elif (step+1) % lSum == 0:
            out, delta, loss, _, summaries = self.sess.run(
                [self.nn,
                 self.delta,
                 self.loss_op,
                 self.train_op,
                 self.summary_op],
                feed_dict={
                    self.input_pl: inputs,
                    self.actions_pl: actions,
                    self.td_targets_pl: targets,
                    self.isTraining: True,
                    self.keep_prob: self.dropout
                })
            self.writer.add_summary(summaries, step)
            self.writer.flush()
        else:
            out, delta, loss, _ = self.sess.run(
                [self.nn,
                 self.delta,
                 self.loss_op,
                 self.train_op],
                feed_dict={
                    self.input_pl: inputs,
                    self.actions_pl: actions,
                    self.td_targets_pl: targets,
                    self.isTraining: True,
                    self.keep_prob: self.dropout
                })
        print("step: {}, loss: {}".format(step, loss))
        return step, out, delta

    def run_predict(self, inputs, action):
        return self.sess.run(self.nn, feed_dict={
            self.input_pl: inputs,
            self.actions_pl: action,
            self.isTraining: False,
            self.keep_prob: 1.0
        })

    def run_predict_target(self, inputs, action):
        return self.sess.run(self.target_nn, feed_dict={
            self.input_pl: inputs,
            self.actions_pl: action,
            self.isTraining: False,
            self.keep_prob: 1.0
        })

    # def run_get_action_gradients2(self, inputs, actions):
    #     return self.sess.run(self.action_grads, feed_dict={
    #         self.input_pl: inputs,
    #         self.actions_pl: actions,
    #         self.isTraining: True,
    #         self.keep_prob: 1.0
    #     })

    # def run_get_action_gradients(self, inputs, actions):
    #     return self.sess.run(self.action_grads, feed_dict={
    #         self.input_pl: inputs,
    #         self.actions_pl: actions,
    #         self.isTraining: True,
    #         self.keep_prob: 1.0
    #     })

    def run_get_gradients(self, inputs, actions, targets):
        return zip(self.nn_params, self.sess.run(self.grads, feed_dict={
            self.input_pl: inputs,
            self.actions_pl: actions,
            self.td_targets_pl: targets,
            self.isTraining: False,
            self.keep_prob: 1.0
        }))

    def run_get_loss(self, inputs, actions, targets):
        return self.sess.run(self.loss_op, feed_dict={
            self.input_pl: inputs,
            self.actions_pl: actions,
            self.td_targets_pl: targets,
            self.isTraining: False,
            self.keep_prob: 1.0
        })

    def run_update_target_nn(self):
        self.sess.run(self.target_nn_update_op)
