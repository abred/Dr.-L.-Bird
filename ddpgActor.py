import numpy as np
import os
import shutil
import sys
import time

from tensorflow.python.framework import ops
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.distributions as tfd
import math

import tfUtils as tfu


class Actor:
    # O = 3

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
    actions_dim = 6
    vgg_state_dim = 224

    def __init__(self, sess, out_dir, params,
                 inputs=None):
        self.sess = sess
        self.params = params
        self.summaries = []
        self.weight_summaries = []
        if params['batchnorm']:
            self.batchnorm = slim.batch_norm
            # self.batchnorm = tfu.batch_normBUG
        else:
            self.batchnorm = None

        self.dropout = params['dropout']
        self.weightDecay = params['weight-decayActor']
        self.learning_rate = params['learning-rateActor']
        self.momentum = params['momentumActor']
        self.opti = params['optimizerActor']
        self.bnDecay = params['batchnorm-decay']
        self.useVGG = params['useVGG']
        self.tau = params['tau']
        self.miniBatchSize = params['miniBatchSize']

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

        with tf.variable_scope('Actor'):
            self.keep_prob = tf.placeholder(tf.float32)
            self.isTraining = tf.placeholder(tf.bool)

            # Actor Network
            self.setNN()

            self.loss_op = self.defineLoss()
            self.train_op = self.defineTraining()

        _VARSTORE_KEY = ("__variable_store",)
        varstore = ops.get_collection(_VARSTORE_KEY)[0]
        variables = tf.get_collection(
            ops.GraphKeys.GLOBAL_VARIABLES,
            scope="Actor")

        for v in variables:
            print(v.name)
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

        self.summary_op = tf.summary.merge(self.summaries)
        self.weight_summary_op = tf.summary.merge(self.weight_summaries)
        self.writer = tf.summary.FileWriter(out_dir, sess.graph)

    def setNN(self):
        prevTrainVarCount = len(tf.trainable_variables())

        if self.useVGG:
            defNN = self.defineNNVGG
        else:
            defNN = self.defineNN

        self.input_pl, self.nn = defNN()
        self.nn_params = tf.trainable_variables()[prevTrainVarCount:]
        mu, var = tf.split(self.nn, 2, 1)

        var = tf.nn.softplus(var)
        self.mu = tf.Print(mu, [mu], "muActor:", summarize=1000,
                      first_n=20)
        self.var = tf.Print(var, [var], "varActor:", summarize=1000,
                       first_n=20)
        mu1, mu2, mu3 = tf.split(self.mu, 3, 1)
        self.sigma = tf.sqrt(self.var)
        s1, s2, s3 = tf.split(self.sigma, 3, 1)

        self.normal = tfd.Normal(loc=0.0, scale=1.0)


        o1n = self.normal.sample(1)
        o1 = o1n * s1 + mu1
        o2n = self.normal.sample(1)
        o2 = o2n * s2 + mu2
        o3n = self.normal.sample(1)
        o3 = o3n * s3 + mu3
        # o1 = tf.random_normal((1,), mu1, s1)
        # o2 = tf.random_normal((1,), mu2, s2)
        # o3 = tf.random_normal((1,), mu3, s3)
        out = tf.concat([o1n, o2n, o3n], 0)
        self.outN = tf.Print(out, [out], "outActorNorm:",
                            summarize=1000,
                            first_n=20)
        out = tf.concat([o1, o2, o3], 0)
        self.out = tf.Print(out, [out], "outActor:",
                            summarize=1000,
                            first_n=20)
        # self.nn = net

        # Target Network
        with tf.variable_scope('target'):
            prevTrainVarCount = len(tf.trainable_variables())
            self.target_input_pl, self.target_nn = \
                defNN(isTargetNN=True)
            self.target_nn_params = \
                tf.trainable_variables()[prevTrainVarCount:]
            with tf.variable_scope('init'):
                for i in range(len(self.nn_params)):
                    tf.Variable.assign(
                        self.target_nn_params[i],
                        self.nn_params[i].initialized_value())
            self.target_nn_update_op = self.define_update_target_nn_op()

            mu, var = tf.split(self.nn, 2, 1)

            var = tf.nn.softplus(var)
            mu = tf.Print(mu, [mu], "muTargetActor:", summarize=1000,
                               first_n=20)
            var = tf.Print(var, [var], "varTargetActor:", summarize=1000,
                               first_n=20)
            mu1, mu2, mu3 = tf.split(mu, 3, 1)
            s1, s2, s3 = tf.split(tf.sqrt(var), 3, 1)

            o1 = tf.random_normal((1,), mu1, s1)
            o2 = tf.random_normal((1,), mu2, s2)
            o3 = tf.random_normal((1,), mu3, s3)
            out = tf.concat([o1, o2, o3], 1)
            self.target_out = tf.Print(out, [out], "outTargetActor:",
                                       summarize=1000,
                                       first_n=20)

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
                net = tf.reshape(net, [self.miniBatchSize,
                                       -1,
                                       remSzY*remSzX*H],
                                 name='flatten')
                net = net / 255.0
                print(remSzY, remSzX, net, H)

                for i in range(len(self.HFC)):
                    net = slim.fully_connected(net, 256,
                                               scope='fc' + str(i))
                    if not isTargetNN:
                        self.weight_summaries += [tf.summary.histogram(
                            'fc' + str(i),
                            net)]
                    print(net)
                    if self.dropout:
                        net = slim.dropout(net,
                                           keep_prob=self.dropout,
                                           is_training=self.isTraining)
                        print(net)
                net = slim.fully_connected(net, self.actions_dim,
                                           activation_fn=None,
                                           scope='out')
                net = tf.Print(net, [net], "outputActor:", summarize=1000,
                               first_n=10)
                if not isTargetNN:
                    r, th, t = tf.split(net, 3, 1)
                    # r, th, t = tf.split(1, 3, net)
                    # r_o = -50.0 * tf.sigmoid(r)
                    # th_o = 50.0 * tf.sigmoid(th)
                    # t_o = 4000.0 * tf.sigmoid(t)
                    self.weight_summaries += [
                        # tf.summary.histogram('out' + '/radius_action', r_o),
                        # tf.summary.histogram('out' + '/theta_action', th_o),
                        # tf.summary.histogram('out' +
                        #                      '/time_delay_action',t_o),
                        tf.summary.histogram('out' +
                                             '/radius_action_before_sig', r),
                        tf.summary.histogram('out' +
                                             '/theta_action_before_sig', th),
                        tf.summary.histogram('out' +
                                             '/time_delay_action_before_sig',t)
                    ]
                print(net)
                # net = tf.sigmoid(net)
                # net = tf.tanh(net)

        return self.images, net

    def defineNN(self, isTargetNN=False):
        with tf.variable_scope('inf'):
            images = tf.placeholder(
                tf.float32,
                shape=[self.miniBatchSize, None,
                       self.state_dim_y,
                       self.state_dim_x,
                       self.col_channels],
                name='input')

            net = images
            net = tf.reshape(net, [-1,
                                   self.state_dim_y,
                                   self.state_dim_x,
                                   self.col_channels],
                             name='rnn_input_flatten')
            net = tf.Print(net, [net], "input", first_n=2, summarize=10000)
            # net = (net - 127.0) / 255.0
            with slim.arg_scope(
                [slim.fully_connected, slim.conv2d],
                activation_fn=tf.nn.relu,
                weights_initializer=
                    tf.contrib.layers.variance_scaling_initializer(
                        factor=1.0, mode='FAN_AVG', uniform=True),
                weights_regularizer=slim.l2_regularizer(self.weightDecay),
                biases_initializer=
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
                net = tf.reshape(net, [self.miniBatchSize,
                                       -1,
                                       remSzY*remSzX*self.Hconv4],
                                 name='flatten')

                with slim.arg_scope([slim.fully_connected],
                                    normalizer_fn=self.batchnorm,
                                    normalizer_params={
                                        'fused': True,
                                        'is_training': self.isTraining,
                                        'updates_collections': None,
                                        'decay': self.bnDecay,
                                        'scale': True}):
                    self.seqlen = tf.placeholder(tf.int32,
                                                 [self.miniBatchSize])
                    cell = tf.contrib.rnn.LSTMCell(self.HFC[0])
                    rnn_outputs, final_state = \
                        tf.nn.dynamic_rnn(cell, net,
                                          sequence_length=self.seqlen,
                                          dtype=tf.float32)
                    if self.dropout:
                        rnn_outputs = slim.dropout(rnn_outputs,
                                                   keep_prob=self.dropout,
                                                   is_training=self.isTraining)
                    last_rnn_output = \
                        tf.gather_nd(rnn_outputs,
                                     tf.stack([tf.range(self.miniBatchSize),
                                               self.seqlen-1], axis=1))

                    # for i in range(len(self.HFC)):
                    #     net = slim.fully_connected(net, self.HFC[i],
                    #                                scope='fc' + str(i))
                    #     if not isTargetNN:
                    #         self.weight_summaries += [tf.summary.histogram(
                    #             'fc' + str(i),
                    #             net)]
                    #     print(net)
                    #     if self.dropout:
                    #         net = slim.dropout(net,
                    #                            keep_prob=self.dropout,
                    #                            is_training=self.isTraining)
                    #         print(net)

                net = last_rnn_output
                net = slim.fully_connected(net, self.actions_dim,
                                           activation_fn=None,
                                           scope='out')
                net = tf.Print(net, [net], "outputActor:", summarize=1000,
                               first_n=10)
                # print(net)
                # net = tf.nn.relu(net)
                r, th, t = tf.split(net, 3, 1)
                # r, th, t = tf.split(1, 3, net)
                # r_o = 50.0 * tf.sigmoid(r)
                # th_o = 9000.0 * tf.sigmoid(th)
                # t_o = 4000.0 * tf.sigmoid(t)
                if not isTargetNN:
                    self.weight_summaries += [
                        # tf.summary.histogram('out' + '/radius_action', r_o),
                        # tf.summary.histogram('out' + '/theta_action', th_o),
                        # tf.summary.histogram('out' +
                        #                      '/time_delay_action',t_o),
                        tf.summary.histogram('out' +
                                             '/radius_action_before_sig', r),
                        tf.summary.histogram('out' +
                                             '/theta_action_before_sig', th),
                        tf.summary.histogram('out' +
                                             '/time_delay_action_before_sig',t)
                    ]
                # net = tf.sigmoid(net)
                # net = tf.tanh(net)
                if not isTargetNN:
                    self.weight_summaries += [tf.summary.histogram('output', net)]

        return images, net

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

    def defineLoss(self):
        with tf.variable_scope("lossActor"):
            # self.td_targets_pl = tf.placeholder(tf.float32, [None, 1],
            #                                     name='tdTargets')
            # self.critic_val_pl = tf.placeholder(tf.float32, [None, 1],
            #                                     name='critic_val_pl')
            # in1 = tf.Print(self.td_targets_pl,
            #                [self.td_targets_pl],
            #                "targets ", first_n=15, summarize=10)
            # in2 = tf.Print(self.critic_val_pl, [self.critic_val_pl],
            #                "critic ", first_n=15, summarize=100)
            # self.delta = in1 - in2


            self.delta_pl = tf.placeholder(tf.float32, [None, 1],
                                           name='delta')
            delta = tf.Print(self.delta_pl,
                           [self.delta_pl],
                           "deltaActor ", first_n=15, summarize=10)
            # inT1 = self.nn
            inT2 = tf.Print(self.outN,
                            [self.outN],
                            "outNActor ", first_n=15, summarize=10)
            # inT2 = self.out
            # in2 = tf.log(inT2)
            # advantage = loss

            # advantage = array_ops.stop_gradient(advantage)
            # return stochastic_tensor.distribution.log_prob(value) * advantage
            # self._log_unnormalized_prob(x) - self._log_normalization()
            logUnNormProb = -0.5 * tf.square(inT2)
            logNorm = 0.5 * tf.log(2. * np.pi) + tf.log(self.sigma)
            # return (x - self.loc) / self.scale

            logProb = logUnNormProb - logNorm
            lossL2 = tf.reduce_mean(logProb * tf.stop_gradient(delta))

            # lossL2 = tf.reduce_mean(in1 * in2)
            # lossL2 = slim.losses.mean_squared_error(self.td_targets_pl,
                                                    # self.nn)
            lossL2 = tf.Print(lossL2, [lossL2], "lossL2Actor ", first_n=25)

            with tf.name_scope(''):
                self.summaries += [
                    tf.summary.scalar('mean_squared_diff_lossActor',
                                      lossL2)]
            regs = []
            for v in self.nn_params:
                if "w" in v.name:
                    regs.append(tf.nn.l2_loss(v))
            lossReg = tf.add_n(regs) * self.weightDecay
            lossReg = tf.Print(lossReg, [lossReg], "regLossActor ", first_n=10)
            with tf.name_scope(''):
                self.summaries += [
                    tf.summary.scalar('mean_squared_diff_loss_regActor',
                                      lossReg)]

            # mu, var = tf.split(inT1, 2, 1)
            # var = tf.nn.softplus(var)
            lossEnt = tf.reduce_sum(-0.5 * (tf.log(2 * np.pi * self.var) + 1)) * 0.001
            lossEnt = tf.Print(lossEnt, [lossEnt], "entLossActor ", first_n=25)
            with tf.name_scope(''):
                self.summaries += [
                    tf.summary.scalar('mean_loss_entropyActor',
                                      lossEnt)]
            loss = lossL2 + lossReg + lossEnt
            with tf.name_scope(''):
                self.summaries += [
                    tf.summary.scalar('mean_squared_diff_loss_with_regActor',
                                      loss)]

        return loss

    def defineTraining(self):
        with tf.variable_scope('train'):
            # self.critic_actions_gradient_pl = tf.placeholder(
        #         tf.float32,
        #         [None, self.actions_dim],
        #         name='CriticActionsGradient')

        #     cag = tf.Print(self.critic_actions_gradient_pl, [self.critic_actions_gradient_pl], "cag", first_n=10)
        #     gd = tf.gradients(
        #         self.nn,
        #         self.nn_params,
        #         # critic grad descent
        #         # here ascent -> negative
        #         -cag)

        # grads = []
        # for i in range(len(gd)):
        #     if gd[i] is None:
        #         grads.append(gd[i])
        #     else:
        #         grads.append(tf.Print(gd[i], [gd[i]], self.nn_params[i].name+"actor grads ", first_n = 10))
        #         # grads.append(tf.clip_by_value(g, -1, 1))
        # self.actor_gradients = grads

            if self.opti == 'momentum':
                optimizer = tf.train.MomentumOptimizer(self.learning_rate,
                                                       self.momentum)
            elif self.opti == 'adam':
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
            elif self.opti == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(
                    self.learning_rate)

        return optimizer.minimize(self.loss_op)

        # return optimizer.apply_gradients(zip(self.actor_gradients,
        #                                      self.nn_params))

    def run_train(self, inputs, delta, step, lens):
        wSum = 300
        lSum = 20
        if (step+1) % wSum == 0:
            _, summaries = self.sess.run([self.train_op,
                                          self.weight_summary_op],
                                         feed_dict={
                self.input_pl: inputs,
                self.delta_pl: delta,
                self.isTraining: True,
                self.keep_prob: self.dropout,
                self.seqlen: lens
            })
            self.writer.add_summary(summaries, step)
            self.writer.flush()
        elif (step+1) % lSum == 0:
            _, summaries = self.sess.run([self.train_op,
                                          self.summary_op],
                                         feed_dict={
                self.input_pl: inputs,
                self.delta_pl: delta,
                self.isTraining: True,
                self.keep_prob: self.dropout,
                self.seqlen: lens
            })
            self.writer.add_summary(summaries, step)
            self.writer.flush()
        else:
            self.sess.run([self.train_op],
                          feed_dict={
                self.input_pl: inputs,
                self.delta_pl: delta,
                self.isTraining: True,
                self.keep_prob: self.dropout,
                self.seqlen: lens
            })

    def run_predict(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.input_pl: inputs,
            self.isTraining: False,
            self.keep_prob: 1.0,
            self.seqlen: [1]
        })

    def run_predict_target(self, inputs):
        return self.sess.run(self.target_out, feed_dict={
            self.target_input_pl: inputs,
            self.isTraining: False,
            self.keep_prob: 1.0,
            self.seqlen: [1]
        })

    def run_update_target_nn(self):
        self.sess.run(self.target_nn_update_op)
