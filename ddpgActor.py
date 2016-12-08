import math

import tensorflow as tf

import tfUtils as tfu


class Actor:
    O = 3
    H1 = 16
    H2 = 32
    H3 = 256
    learning_rate = 0.0001
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
        h3, s, _, _ = tfu.fullyConRelu(h2_f,
                                       8*14*self.H2, self.H3,
                                       scopeName='h3', isTargetNN=isTargetNN,
                                       is_training=self.isTraining)
        self.summaries += s

        o, s = tfu.fullyCon(h3, self.H3, self.O,
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
        outputs = tf.concat(1, [r_o, th_o, t_o])
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

            return tf.train.AdamOptimizer(self.learning_rate, epsilon=0.1).\
                apply_gradients(zip(self.actor_gradients, self.nn_params))

    def run_train(self, inputs, a_grad, step):
        if (step+1) % 10 == 0:
            _, summaries = self.sess.run([self.train_op,
                                          self.summary_op],
                                         feed_dict={
                self.input_pl: inputs,
                self.critic_actions_gradient_pl: a_grad,
                self.isTraining: True
            })
            self.writer.add_summary(summaries, step)
            self.writer.flush()
        else:
            self.sess.run([self.train_op],
                          feed_dict={
                self.input_pl: inputs,
                self.critic_actions_gradient_pl: a_grad,
                self.isTraining: True
            })

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
