import math

import tensorflow as tf

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
    learning_rate = 0.0001
    # mini_batch_size = 16
    tau = 0.001
    train_dir = 'data'
    state_dim_x = 105
    state_dim_y = 60
    actions_dim = 3

    def __init__(self, sess, out_dir):
        self.sess = sess
        self.summaries = []

        with tf.variable_scope('Actor'):
            self.keep_prob = tf.placeholder(tf.float32)
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
            self.train_op = self.define_training()
            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope="Actor")
            self.summary_op = tf.merge_summary(summaries)
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
        h6, s = tfu.convReluLayer(h5,
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
        r_o = 50.0 * tf.sigmoid(r)
        th_o = 9000.0 * tf.sigmoid(th)
        t_o = 4000.0 * tf.sigmoid(t)
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
                    'out' + '/radius_action_before_sig',
                    mark_as_used=False), r),
                tf.histogram_summary(tf.get_default_graph().unique_name(
                    'out' + '/theta_action_before_sig',
                    mark_as_used=False), th),
                tf.histogram_summary(tf.get_default_graph().unique_name(
                    'out' + '/time_delay_action_before_sig',
                    mark_as_used=False), t)
            ]
        # outputs = tf.concat(1, [r_o, th_o, t_o])
        outputs = tf.sigmoid(o)
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

    def define_training(self):
        with tf.variable_scope('train'):
            self.critic_actions_gradient_pl = tf.placeholder(
                tf.float32,
                [None, self.actions_dim],
                name='CriticActionsGradient')
            self.actor_gradients = tf.gradients(
                self.nn,
                self.nn_params,
                # critic grad descent
                # here ascent -> negative
                -self.critic_actions_gradient_pl
                )

            # ag = []
            # for g in self.actor_gradients:
            #     # gp = tf.Print(g, [g], 'grad: ', summarize=100000)
            #     gp = tf.check_numerics(g, "grad numerics")
            #     ag.append(gp)
            # self.actor_gradients = ag

            return tf.train.AdamOptimizer(self.learning_rate, epsilon=0.1).\
                apply_gradients(zip(self.actor_gradients, self.nn_params))

    def run_train(self, inputs, a_grad, step):
        if (step+1) % 10 == 0:
            _, summaries = self.sess.run([self.train_op,
                                          self.summary_op],
                                         feed_dict={
                self.input_pl: inputs,
                self.critic_actions_gradient_pl: a_grad,
                self.isTraining: True,
                self.keep_prob: 0.5
            })
            self.writer.add_summary(summaries, step)
            self.writer.flush()
        else:
            self.sess.run([self.train_op],
                          feed_dict={
                self.input_pl: inputs,
                self.critic_actions_gradient_pl: a_grad,
                self.isTraining: True,
                self.keep_prob: 0.5
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
