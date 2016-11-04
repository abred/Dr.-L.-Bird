import math

import tensorflow as tf

import tfUtils as tfu


class Actor:
    O = 6
    H1 = 16
    H2 = 32
    H3 = 256
    learning_rate = 0.000001
    mini_batch_size = 16
    tau = 0.001
    train_dir = 'data'
    state_dim_x = 105
    state_dim_y = 60
    actions_dim = 3

    def __init__(self, sess, out_dir):
        self.sess = sess
        self.summaries = []

        with tf.variable_scope('Actor'):
            self.isTraining = tf.placeholder(tf.bool)

            # Actor Network
            prevTrainVarCount = len(tf.trainable_variables())
            self.input_pl, self.nn = self.defineNN()
            self.nn_params = tf.trainable_variables()[prevTrainVarCount:]

            # Target Network
            with tf.variable_scope('target'):
                prevTrainVarCount = len(tf.trainable_variables())
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

    def defineNN(self, isTargetNN=False):
        images = tf.placeholder(
            tf.float32,
            shape=[None, self.state_dim_x*self.state_dim_y],
            name='input')
        x = tf.reshape(images, [-1,
                                self.state_dim_y,
                                self.state_dim_x,
                                1],
                       name='deflatten')

        h1, s = tfu.convReluLayer(x,
                                  1, self.H1,
                                  fh=8, fw=8,
                                  strh=4, strw=4,
                                  scopeName='h1',
                                  isTargetNN=isTargetNN,
                                  is_training=self.isTraining)
        self.summaries += s
        h2, s = tfu.convReluLayer(h1,
                                  self.H1, self.H2,
                                  fh=4, fw=4,
                                  strh=2, strw=2,
                                  scopeName='h2',
                                  isTargetNN=isTargetNN,
                                  is_training=self.isTraining)
        self.summaries += s
        h2_f = tf.reshape(h2, [-1, 14*8*self.H2], name='flatten')
        h3, s = tfu.fullyConRelu(h2_f,
                                 8*14*self.H2, self.H3,
                                 scopeName='h3', isTargetNN=isTargetNN,
                                 is_training=self.isTraining)
        self.summaries += s

        o, s = tfu.fullyCon(h3, self.H3, self.O,
                            scopeName='out', isTargetNN=isTargetNN,
                            is_training=self.isTraining)
        self.summaries += s
        mu, var = tf.split(1, 2, o, name='split_mu_var')
        var = tf.nn.softplus(var, name='softplus_var')
        var = tf.maximum(var, 1.0)
        outputs = tf.concat(1, [mu, var])
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

            mu, var = tf.split(1, 2, self.nn)
            factor = tf.div(1.0, tf.sqrt(tf.mul(2*math.pi, var)))
            numerator = -tf.square(self.x-mu)
            denominator = 2.0 * var
            fraction = tf.div(numerator, denominator)
            exp = tf.exp(fraction)
            normal = tf.mul(factor, exp)
            unterschied jetztiges mean damalige action zu gross ->
            prob gauss gegen 0 ->
            log 0 -> inf
            E = tf.log(normal)

            muTag = []
            varTag = []
            eTag = []
            xTag = []
            factorTag = []
            numeratorTag = []
            denominatorTag = []
            fractionTag = []
            normalTag = []
            expTag = []
            for i in range(self.mini_batch_size):
                muTag.append([])
                varTag.append([])
                eTag.append([])
                xTag.append([])
                factorTag.append([])
                numeratorTag.append([])
                denominatorTag.append([])
                fractionTag.append([])
                normalTag.append([])
                expTag.append([])
                for j in range(3):
                    muTag[i].append("mu" + str(i) + "_" + str(j))
                    varTag[i].append("var" + str(i) + "_" + str(j))
                    eTag[i].append("e" + str(i) + "_" + str(j))
                    xTag[i].append("x" + str(i) + "_" + str(j))
                    factorTag[i].append("factor" + str(i) + "_" + str(j))
                    numeratorTag[i].append("numerator" + str(i) + "_" + str(j))
                    denominatorTag[i].append(
                        "denominator" + str(i) + "_" + str(j))
                    fractionTag[i].append("fraction" + str(i) + "_" + str(j))
                    normalTag[i].append("normal" + str(i) + "_" + str(j))
                    expTag[i].append("exp" + str(i) + "_" + str(j))

            lossL2 = tf.reduce_mean(E, 0)
            self.summaries += [tf.scalar_summary('gaussianLoss1', lossL2[0]),
                               tf.scalar_summary('gaussianLoss2', lossL2[1]),
                               tf.scalar_summary('gaussianLoss3', lossL2[2]),
                               tf.scalar_summary(muTag, mu),
                               tf.scalar_summary(varTag, var),
                               tf.scalar_summary(eTag, E),
                               tf.scalar_summary(xTag, self.x),
                               tf.scalar_summary(factorTag, factor),
                               tf.scalar_summary(numeratorTag, numerator),
                               tf.scalar_summary(denominatorTag, denominator),
                               tf.scalar_summary(fractionTag, fraction),
                               tf.scalar_summary(normalTag, normal),
                               tf.scalar_summary(expTag, exp)]
            return E

    def define_training(self, loss):
        with tf.variable_scope('train'):
            self.critic_advantage_pl = tf.placeholder(
                tf.float32,
                [None, 1],
                name='CriticAdvantage')
            self.actor_gradients = tf.gradients(
                loss,
                self.nn_params,
                self.critic_advantage_pl)

            # for g,v in zip(self.actor_gradients, self.nn_params):
            #     sz = tf.size(g)
            #     gradTag = []
            #     for i in range(sz):
            #         gradTag.append('grad'+v.name+str(i))
            #     gradTagTens = tf.convert_to_tensor(gradTag)
            #     tf.reshape(gradTagTens, tf.shape(g))
            #     self.summaries.append(tf.scalar_summary(gradTagTens, g))

            return self.actor_gradients
            # return tf.train.AdamOptimizer(self.learning_rate).\
                # apply_gradients(zip(self.actor_gradients, self.nn_params))

    def run_train(self, inputs, advantage, action, step):
        _, _, summaries = self.sess.run([self.nn,
                                         self.train_op,
                                         self.summary_op],
                                        feed_dict={
            self.input_pl: inputs,
            self.x: action,
            self.critic_advantage_pl: advantage,
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
