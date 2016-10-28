import math

import numpy as np

import tensorflow as tf

import tfUtils as tfu


class Actor:
    I = 832 * 448
    O = 6
    H1 = 32
    H2 = 64
    H3 = 128
    H4 = 256
    H5 = 512
    H6 = 512
    H7 = 1024
    learning_rate = 0.0001
    mini_batch_size = 8
    tau = 0.001
    train_dir = 'data'
    actions_dim = 3

    def __init__(self, sess, out_dir):
        self.sess = sess
        self.summaries = []

        with tf.variable_scope('Actor'):
            self.isTraining = tf.placeholder(tf.bool)

            # Actor Network
            prevTrainVarCount = len(tf.trainable_variables())
            # print("actor 1: {}".format(prevTrainVarCount))
            self.input_pl, self.nn = self.defineNN()
            self.nn_params = tf.trainable_variables()[prevTrainVarCount:]

            # Target Network
            with tf.variable_scope('target'):
                prevTrainVarCount = len(tf.trainable_variables())
                # print("actor 2: {}".format(prevTrainVarCount))
                self.target_input_pl, self.target_nn = \
                    self.defineNN(isTargetNN=True)
                self.target_nn_params = \
                    tf.trainable_variables()[prevTrainVarCount:]
                with tf.variable_scope('init'):
                    for i in range(len(self.nn_params)):
                        tf.Variable.assign(
                            self.target_nn_params[i],
                            self.nn_params[i].initialized_value())
                self.target_nn_update_op = self.define_update_target_nn_op()

            # Optimization Op
            self.loss_op = self.define_loss()
            self.train_op = self.define_training(self.loss_op)
            self.summary_op = tf.merge_summary(self.summaries)
            self.writer = tf.train.SummaryWriter(out_dir, sess.graph)
            # print("actor 3: {}".format(len(tf.trainable_variables())))
            # print("actor params: {}".format(self.nn_params))
            # print("actortarget params: {}".format(self.target_nn_params))


    def defineNN(self, isTargetNN=False):
        images = tf.placeholder(tf.float32,
                                shape=[None, 448*832],
                                name='input')
        x = tf.reshape(images, [-1, 448, 832, 1], name='deflatten')
        h1, s = tfu.convReluPoolLayer(x, 1, self.H1, fh=5, fw=5,
                                      scopeName='h1',
                                      isTargetNN=isTargetNN,
                                      is_training=self.isTraining)
        self.summaries += s
        h2, s = tfu.convReluPoolLayer(h1, self.H1, self.H2, scopeName='h2',
                                      isTargetNN=isTargetNN,
                                      is_training=self.isTraining)
        self.summaries += s
        h3, s = tfu.convReluPoolLayer(h2, self.H2, self.H3, scopeName='h3',
                                      isTargetNN=isTargetNN,
                                      is_training=self.isTraining)
        self.summaries += s
        h4, s = tfu.convReluPoolLayer(h3, self.H3, self.H4, scopeName='h4',
                                      isTargetNN=isTargetNN,
                                      is_training=self.isTraining)
        self.summaries += s
        h5, s = tfu.convReluPoolLayer(h4, self.H4, self.H5, scopeName='h5',
                                      isTargetNN=isTargetNN,
                                      is_training=self.isTraining)
        self.summaries += s
        h6, s = tfu.convReluPoolLayer(h5, self.H5, self.H6, scopeName='h6',
                                      isTargetNN=isTargetNN,
                                      is_training=self.isTraining)
        self.summaries += s
        h6_f = tf.reshape(h6, [-1, 7*13*self.H6], name='flatten')
        h7, s= tfu.fullyConReluDrop(h6_f, 7*13*self.H6, self.H7,
                                    scopeName='h7', isTargetNN=isTargetNN,
                                    is_training=self.isTraining)
        self.summaries += s

        with tf.variable_scope('out') as scope:
            wmu, sw = tfu.weight_variable([self.H7, self.O/2], 'wMu')
            self.summaries += sw
            wsd, sw = tfu.weight_variable([self.H7, self.O/2], 'wSd')
            self.summaries += sw
            weights = tf.concat(1, [wmu, wsd])
            bmu, sb = tfu.bias_variable([self.O/2], 'bmu')
            self.summaries += sb
            bsd, sb = tfu.bias_variable([self.O/2], 'bsd')
                                        # , minV=-2.5, maxV=1.0)
            self.summaries += sb
            biases = tf.concat(0, [bmu, bsd])
            o_fc = tf.matmul(h7, weights) + biases
            if not isTargetNN:
                self.summaries += [
                    tf.histogram_summary(
                        tf.get_default_graph().unique_name(
                            'out' + '/pre_activation',
                            mark_as_used=False), o_fc)
                ]

            # outputs = tf.sigmoid(o_fc)
            r, th, t, rsd, thsd, tsd = tf.split(1, 6, o_fc)
            r_o = 50.0 * tf.sigmoid(r)
            th_o = 9000.0 * tf.sigmoid(th)
            t_o = 4000.0 * tf.sigmoid(t)
            rsd_o = 25.0 * tf.sigmoid(rsd)
            thsd_o = 9000.0 * tf.sigmoid(thsd)
            tsd_o = 4000.0 * tf.sigmoid(tsd)
            if not isTargetNN:
                self.summaries += [
                    tf.histogram_summary(tf.get_default_graph().unique_name(
                        'out' + '/radius_action',
                        mark_as_used=False), r_o),
                    tf.histogram_summary(tf.get_default_graph().unique_name(
                        'out' + '/theta_action',
                        mark_as_used=False), th_o),
                    tf.histogram_summary(tf.get_default_graph().unique_name(
                        'out' + '/time_delay_action',
                        mark_as_used=False), t_o),
                    tf.histogram_summary(tf.get_default_graph().unique_name(
                        'out' + '/sd_radius_action',
                        mark_as_used=False), rsd),
                    tf.histogram_summary(tf.get_default_graph().unique_name(
                        'out' + '/sd_theta_action',
                        mark_as_used=False), thsd),
                    tf.histogram_summary(tf.get_default_graph().unique_name(
                        'out' + '/sd_time_delay_action',
                        mark_as_used=False), tsd)
                ]
            outputs = tf.concat(1, [r_o, th_o, t_o, rsd_o, thsd_o, tsd_o])
        return images, outputs

    def define_update_target_nn_op(self):
        with tf.variable_scope('update'):
            tau = tf.constant(self.tau, name='tau')
            invtau = tf.constant(1.0-self.tau, name='invtau')
            return \
                [self.target_nn_params[i].assign(
                    tf.mul(self.nn_params[i], tau) +
                    tf.mul(self.target_nn_params[i], invtau))
                 for i in range(len(self.target_nn_params))]

    def define_loss(self):
        with tf.variable_scope('loss'):
            self.x = tf.placeholder(tf.float32, [None, 3],
                                    name='noisedActions')

            mu, sigma = tf.split(1, 2, self.nn)
            factor = tf.div(1.0, tf.sqrt(tf.mul(2*math.pi, sigma)))
            # expp1 = -tf.square(tf.div(self.x, [50.0, 9000.0, 4000.0])-mu)
            expp1 = -tf.square(self.x-mu)
            expp2 = 2.0 * tf.square(sigma)
            expp3 = tf.div(expp1, expp2)
            exp = tf.exp(expp3)
            mul = tf.mul(factor, exp)
            E = tf.log(mul)

            muTag = []
            sigmaTag = []
            eTag = []
            xTag = []
            fTag = []
            expTag = []
            exp1Tag = []
            exp2Tag = []
            exp3Tag = []
            mulTag = []
            for i in range(self.mini_batch_size):
                muTag.append([])
                sigmaTag.append([])
                eTag.append([])
                xTag.append([])
                fTag.append([])
                expTag.append([])
                exp1Tag.append([])
                exp2Tag.append([])
                exp3Tag.append([])
                mulTag.append([])
                for j in range(3):
                    muTag[i].append("mu" + str(i) + "_" + str(j))
                    sigmaTag[i].append("sigma" + str(i) + "_" + str(j))
                    eTag[i].append("e" + str(i) + "_" + str(j))
                    xTag[i].append("x" + str(i) + "_" + str(j))
                    fTag[i].append("f" + str(i) + "_" + str(j))
                    expTag[i].append("exp" + str(i) + "_" + str(j))
                    exp1Tag[i].append("exp1_" + str(i) + "_" + str(j))
                    exp2Tag[i].append("exp2_" + str(i) + "_" + str(j))
                    exp3Tag[i].append("exp3_" + str(i) + "_" + str(j))
                    mulTag[i].append("mul" + str(i) + "_" + str(j))

            lossL2 = tf.reduce_mean(E, 0)
            self.summaries += [tf.scalar_summary('gaussianLoss1', lossL2[0]),
                               tf.scalar_summary('gaussianLoss2', lossL2[1]),
                               tf.scalar_summary('gaussianLoss3', lossL2[2]),
                               tf.scalar_summary(muTag, mu),
                               tf.scalar_summary(sigmaTag, sigma),
                               tf.scalar_summary(eTag, E),
                               tf.scalar_summary(xTag, self.x),
                               tf.scalar_summary(fTag, factor),
                               tf.scalar_summary(expTag, exp),
                               tf.scalar_summary(exp1Tag, expp1),
                               tf.scalar_summary(exp2Tag, expp2),
                               tf.scalar_summary(exp3Tag, expp3),
                               tf.scalar_summary(mulTag, mul)
            ]
            return E

    def define_training(self, loss):
        with tf.variable_scope('train'):
            self.critic_actions_gradient_pl = tf.placeholder(
                tf.float32,
                [None, self.actions_dim],
                name='CriticActionsGradient')
            self.actor_gradients = tf.gradients(
                loss,
                self.nn_params,
                # critic grad descent
                # here ascent -> negative
                -self.critic_actions_gradient_pl)

            # self.actor_gradients = tf.gradients(
            #     tf.reduce_mean(tf.mul(-self.critic_actions_gradient_pl,
            #                           loss),
                               # 0),
                # self.nn_params)

            return tf.train.AdamOptimizer(self.learning_rate).\
                apply_gradients(zip(self.actor_gradients, self.nn_params))

    def run_train(self, inputs, a_grad, action, step):
        _, _, summaries = self.sess.run([self.nn,
                                         self.train_op,
                                         self.summary_op],
                                        feed_dict={
            self.input_pl: inputs,
            self.x: action,
            self.critic_actions_gradient_pl: a_grad,
            self.isTraining: True
        })
        self.writer.add_summary(summaries, step)
        self.writer.flush()

    def run_predict(self, inputs):
        return self.sess.run(self.nn, feed_dict={
            self.input_pl: inputs,
            self.isTraining: False
        })

    def run_predict_target(self, inputs):
        return self.sess.run(self.target_nn, feed_dict={
            self.target_input_pl: inputs,
            self.isTraining: False
        })

    def run_update_target_nn(self):
        self.sess.run(self.target_nn_update_op)
