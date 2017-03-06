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

import tfUtils as tfu


class Critic:
    O = 1
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

    def __init__(self, sess, out_dir, glStep, params):
        self.sess = sess
        self.summaries = []
        self.out_dir = out_dir
        self.useVGG = params['useVGG']
        self.batchnorm = params['batchnorm']
        self.dropout = params['dropout']
        self.top = params['top']
        self.weightDecay = params['weight-decayCritic']
        self.learning_rate = params['learning-rateCritic']
        self.momentum = params['momentumCritic']
        self.opti = params['optimizerCritic']
        self.stopGrad = params['stopGradCritic']
        self.bnDecay = params['batchnorm-decay']
        if params['batchnorm']:
            # self.batchnorm = slim.batch_norm
            self.batchnorm = tfu.batch_normBUG
        else:
            self.batchnorm = None

        print("dropout critic", self.dropout)
        print("vgg critic", self.useVGG)
        print("top critic", self.top)

        self.global_step = glStep
        if self.useVGG:
            self.vggsaver = tf.train.import_meta_graph(
                '/home/s7550245/convNet/vgg-model.meta',
                import_scope="Critic/VGG",
                clear_devices=True)

            _VARSTORE_KEY = ("__variable_store",)
            varstore = ops.get_collection(_VARSTORE_KEY)[0]
            variables = tf.get_collection(
                ops.GraphKeys.GLOBAL_VARIABLES,
                scope="Critic/VGG")

            for v in variables:
                varstore._vars[v.name.split(":")[0]] = v
                # print(v.name)

        with tf.variable_scope('Critic'):
            self.keep_prob = tf.placeholder(tf.float32)
            self.isTraining = tf.placeholder(tf.bool)

            # Critic Network
            if self.useVGG:
                self.setVGGNN()
            else:
                self.setNN()

            self.loss_op = self.define_loss()
            self.train_op = self.defineTraining()

            self.action_grads = self.define_action_grad()
            self.grads = self.define_grads()

            self.summary_op = tf.summary.merge(self.summaries)
            self.writer = tf.summary.FileWriter(out_dir, sess.graph)

    def setVGGNN(self):
        prevTrainVarCount = len(tf.trainable_variables())

        with tf.variable_scope("VGG") as scope:
            scope.reuse_variables()
            self.input_pl, self.nnB = self.defineNNVGG(top=top)
        with tf.variable_scope("VGG") as scope:
            self.actions_pl, self.nn = self.defineNNVGGTop(self.nnB,
                                                           top=top)
        print(self.nn)
        self.nn_params = tf.trainable_variables()[prevTrainVarCount:]

        # Target Network
        with tf.variable_scope('target'):
            prevTrainVarCount = len(tf.trainable_variables())
            self.target_input_pl, self.nnBT = \
                self.defineNNVGG(top=top, isTargetNN=True)
            self.target_actions_pl, self.target_nn = \
                self.defineNNVGGTop(self.nnBT, top=top,
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

        self.input_pl, self.actions_pl, self.nn = self.defineNN2()
        self.nn_params = tf.trainable_variables()[prevTrainVarCount:]

        # Target Network
        with tf.variable_scope('target'):
            prevTrainVarCount = len(tf.trainable_variables())
            self.target_input_pl, self.target_actions_pl, self.target_nn =\
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
            actions = tf.placeholder(tf.float32,
                                     shape=[None, self.actions_dim],
                                     name='ActorActions')

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
                net = tf.concat(
                    1, [tf.reshape(net, [-1, remSzY*remSzX*self.Hconv4],
                                   name='flatten'),
                        actions])

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
                net = slim.fully_connected(net, 1, activation_fn=None,
                                           scope='out')
                print(net)
                # net = tf.sigmoid(net)
                if not isTargetNN:
                    self.summaries += [tf.summary.histogram('output', net)]

        return images, actions, net

    def defineNN(self, isTargetNN=False):
        images = tf.placeholder(
            tf.float32,
            shape=[None,
                   self.state_dim_y,
                   self.state_dim_x,
                   self.col_channels],
            name='input')
        actions = tf.placeholder(tf.float32,
                                 shape=[None, self.actions_dim],
                                 name='ActorActions')

        # x = tf.reshape(images, [-1,
        #                         self.state_dim_y,
        #                         self.state_dim_x,
        #                         self.col_channels],
        #                name='deflatten')

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
        h7_a = tf.concat(1, [
            tf.reshape(h7, [-1, 13*7*self.H7], name='flatten'),
            actions])

        # h7_a = tf.Print(h7_a, [h7_a], message="h7_a ", summarize=9999)

        h8, s = tfu.fullyConRelu(h7_a,
                                 13*7*self.H7+self.actions_dim, self.H8,
                                 scopeName='h8', isTargetNN=isTargetNN,
                                 is_training=self.isTraining)
        # h8 = tf.Print(h8, [h8], message="h8 ", summarize=9999)
        self.summaries += s
        h9, s = tfu.fullyConRelu(h8,
                                 self.H8, self.H9,
                                 scopeName='h9', isTargetNN=isTargetNN,
                                 is_training=self.isTraining)
        # h9 = tf.Print(h9, [h9], message="h9 ", summarize=9999)

        self.summaries += s
        o, s = tfu.fullyCon(h9, self.H9, self.O,
                            scopeName='out', isTargetNN=isTargetNN,
                            is_training=self.isTraining)
        self.summaries += s
        return images, actions, o

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
            actions = tf.placeholder(tf.float32,
                                     shape=[None, self.actions_dim],
                                     name='ActorActions')

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

            bottom_a = tf.concat(1, [
                tf.reshape(bottom, [-1, remSz*remSz*H], name='flatten'),
                actions])

            if self.dropout:
                h14, s = tfu.fullyConReluDrop(bottom_a,
                                              remSz*remSz*H+self.actions_dim,
                                              self.HFC1VGG,
                                              self.keep_prob, scopeName='hfc1',
                                              norm=self.batchnorm,
                                              isTargetNN=isTargetNN,
                                              is_training=self.isTraining)
            else:
                h14, s = tfu.fullyConRelu(bottom_a,
                                          remSz*remSz*H+self.actions_dim,
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
            return actions, o

    def define_update_target_nn_op(self):
        with tf.variable_scope('update'):
            tau = tf.constant(self.tau, name='tau')
            invtau = tf.constant(1.0-self.tau, name='invtau')
            if self.useVGG:
                a=1
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


    def define_loss(self):
        with tf.variable_scope('loss2'):
            self.td_targets_pl = tf.placeholder(tf.float32, [None, 1],
                                             name='tdTargets')

            lossL2 = tfu.mean_squared_diff(self.td_targets_pl, self.nn)
            lossL2 = tf.Print(lossL2, [lossL2], "lossL2 ", first_n=10)

            with tf.name_scope(''):
                self.summaries += [
                    tf.summary.scalar('mean_squared_diff_loss',
                                      lossL2)]
            regs = []
            for v in self.nn_params:
                if "w" in v.name:
                    regs.append(tf.nn.l2_loss(v))
            lossReg = tf.add_n(regs) * self.weightDecay
            lossReg = tf.Print(lossReg, [lossReg], "regLoss ", first_n=10)
            with tf.name_scope(''):
                self.summaries += [
                    tf.summary.scalar('mean_squared_diff_loss_reg',
                                      lossReg)]

            loss = lossL2 + lossReg
            with tf.name_scope(''):
                self.summaries += [
                    tf.summary.scalar('mean_squared_diff_loss_with_reg',
                                      loss)]

        return loss

    def defineTraining(self, conv=False):
        with tf.variable_scope('train'):
            var_list = None
            if self.useVGG and self.stopGrad is not None:
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

            if self.opti == 'momentum':
                optimizer = tf.train.MomentumOptimizer(self.learning_rate,
                                                       self.momentum)
            elif self.opti == 'adam':
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
            elif self.opti == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(
                    self.learning_rate)

            print(optimizer)
            return optimizer.minimize(self.loss_op, var_list=var_list,
                                      global_step=self.global_step)

    def define_action_grad(self):
        with tf.variable_scope('getActionGradient'):
            return tf.gradients(self.nn, self.actions_pl)

    def define_grads(self):
        with tf.variable_scope('getGrads'):
            optimizer = tf.train.AdamOptimizer(self.learning_rate, epsilon=0.1)
            return optimizer.compute_gradients(self.loss_op,
                                               var_list=self.nn_params)

    def run_train(self, inputs, actions, targets):
        step = self.sess.run(self.global_step)
        if (step+1) % 10 == 0:
            out, loss, _, summaries = self.sess.run([self.nn,
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
            out, loss, _ = self.sess.run(
                [self.nn, self.loss_op, self.train_op],
                feed_dict={
                    self.input_pl: inputs,
                    self.actions_pl: actions,
                    self.td_targets_pl: targets,
                    self.isTraining: True,
                    self.keep_prob: self.dropout
                })
        # print("loss: {}".format(loss))
        return step, out, loss


    def run_predict(self, inputs, action):
        return self.sess.run(self.nn, feed_dict={
            self.input_pl: inputs,
            self.actions_pl: action,
            self.isTraining: False,
            self.keep_prob: 1.0
        })

    def run_predict_target(self, inputs, action):
        return self.sess.run(self.target_nn, feed_dict={
            self.target_input_pl: inputs,
            self.target_actions_pl: action,
            self.isTraining: False,
            self.keep_prob: 1.0
        })

    def run_get_action_gradients2(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.input_pl: inputs,
            self.actions_pl: actions,
            self.isTraining: True,
            self.keep_prob: 1.0
        })

    def run_get_action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.input_pl: inputs,
            self.actions_pl: actions,
            self.isTraining: False,
            self.keep_prob: 1.0
        })

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
