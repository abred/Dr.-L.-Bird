import tensorflow as tf

import tfUtils as tfu


class Actor:
    I = 832 * 448
    O = 3
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
            self.train_op = self.define_training()
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
            weights, sw = tfu.weight_variable_unit([self.H7, self.O], 'w')
            self.summaries += sw
            biases, sb = tfu.bias_variable([self.O], 'b')
            self.summaries += sb
            o_fc = tf.matmul(h7, weights) + biases
            if not isTargetNN:
                self.summaries += [
                    tf.histogram_summary(
                        tf.get_default_graph().unique_name(
                            'out' + '/pre_activation',
                            mark_as_used=False), o_fc)
                ]

            outputs = tf.sigmoid(o_fc)
            # x, y, t = tf.split(1, 3, o_fc)
            # x_o = -50.0 * tf.sigmoid(x)
            # y_o = 50.0 * tf.sigmoid(y)
            # t_o = 4000.0 * tf.sigmoid(t)
            # if not isTargetNN:
            #     self.summaries += [
            #         tf.histogram_summary(tf.get_default_graph().unique_name(
            #             'out' + '/x_coord_action',
            #             mark_as_used=False), x_o),
            #         tf.histogram_summary(tf.get_default_graph().unique_name(
            #             'out' + '/y_coord_action',
            #             mark_as_used=False), y_o),
            #         tf.histogram_summary(tf.get_default_graph().unique_name(
            #             'out' + '/time_delay_action',
            #             mark_as_used=False), t_o)
            #     ]
            r, th, t = tf.split(1, 3, o_fc)
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
                tf.reduce_mean(tf.mul(-self.critic_actions_gradient_pl,
                                      self.nn),
                               0),
                self.nn_params
                # critic grad descent
                # here ascent -> negative
                )

            # print(self.nn_params)
            # for i in range(len(self.nn_params)):
            #     print(self.nn_params[i].name)
            #     print(self.actor_gradients[i])
            #     print(tf.constant(10.0, dtype=tf.float32))
            #     temp = tf.div(self.actor_gradients[i], tf.constant(10.0, dtype=tf.float32))

            # self.actor_gradients = \
            #     tf.truediv(
            #         tf.gradients(
            #             self.nn,
            #             self.nn_params,
            #             # critic grad descent
            #             # here ascent -> negative
            #             -self.critic_actions_gradient_pl),
            #         tf.constant(self.mini_batch_size, dtype=tf.float32))

            return tf.train.AdamOptimizer(self.learning_rate).\
                apply_gradients(zip(self.actor_gradients, self.nn_params))

    def run_train(self, inputs, a_grad, step):
        _, _, summaries = self.sess.run([self.nn,
                                         self.train_op,
                                         self.summary_op],
                                        feed_dict={
            self.input_pl: inputs,
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
