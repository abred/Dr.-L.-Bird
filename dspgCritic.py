import tensorflow as tf

import tfUtils as tfu


class Critic:
    I = 832 * 448
    O = 1
    H1 = 32
    H2 = 64
    H3 = 128
    H4 = 256
    H5 = 512
    H6 = 512
    H7 = 1024
    learning_rate = 0.001
    mini_batch_size = 8
    tau = 0.001
    train_dir = 'data'
    actions_dim = 3
    weight_decay = 0.001

    def __init__(self, sess, out_dir):
        self.sess = sess
        self.summaries = []

        with tf.variable_scope('Critic'):
            self.isTraining = tf.placeholder(tf.bool)

            # Critic Network
            prevTrainVarCount = len(tf.trainable_variables())
            # print("critic 1: {}".format(prevTrainVarCount))
            self.input_pl, self.actions_pl, self.nn = self.defineNN()
            self.nn_params = tf.trainable_variables()[prevTrainVarCount:]

            # Target Network
            with tf.variable_scope('target'):
                prevTrainVarCount = len(tf.trainable_variables())
                # print("critic 2: {}".format(prevTrainVarCount))
                self.target_input_pl, self.target_actions_pl, self.target_nn =\
                    self.defineNN(isTargetNN=True)
                self.target_nn_params = \
                    tf.trainable_variables()[prevTrainVarCount:]
                with tf.variable_scope('init'):
                    for i in range(len(self.nn_params)):
                        tf.Variable.assign(
                            self.target_nn_params[i],
                            self.nn_params[i].initialized_value())
                self.target_nn_update_op = self.define_update_target_nn_op()

            self.loss_op = self.define_loss()
            self.train_op = self.define_training(self.loss_op)

            # Get the gradient of the net w.r.t. the action
            self.action_grads = self.define_action_grad()
            self.summary_op = tf.merge_summary(self.summaries)
            self.writer = tf.train.SummaryWriter(out_dir, sess.graph)
            # print("critic 3: {}".format(prevTrainVarCount))
            # print("critic params: {}".format(self.nn_params))
            # print("critictarget params: {}".format(self.target_nn_params))

    def defineNN(self, isTargetNN=False):
        images = tf.placeholder(tf.float32,
                                shape=[None, 448*832],
                                name='input')
        actions = tf.placeholder(tf.float32,
                                 shape=[None, self.actions_dim],
                                 name='ActorActions')
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

        h6_a = tf.concat(1, [tf.reshape(h6, [-1, 7*13*self.H6], name='flatten'),
                             actions])
        h7, s = tfu.fullyConReluDrop(h6_a, 7*13*self.H6+self.actions_dim,
                                     self.H7,
                                     scopeName='h7', isTargetNN=isTargetNN,
                                     is_training=self.isTraining)
        self.summaries += s

        with tf.variable_scope('out') as scope:
            weights, sw = tfu.weight_variable_unit([self.H7, self.O], 'w')
            self.summaries += sw
            biases, sb = tfu.bias_variable([self.O], 'b')
            self.summaries += sb
            o = tf.matmul(h7, weights) + biases
            if not isTargetNN:
                self.summaries += [
                    tf.histogram_summary(
                        tf.get_default_graph().unique_name(
                            'out' + '/output',
                            mark_as_used=False), o),
                ]
        return images, actions, o

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
            self.td_targets_pl = tf.placeholder(tf.float32, [None, 1],
                                             name='tdTargets')

            lossL2 = tfu.mean_squared_diff(self.td_targets_pl, self.nn)
            self.summaries += [tf.scalar_summary('mean_squared_diff_loss',
                                                 lossL2)]
            lossReg = tf.add_n([tf.nn.l2_loss(v) for v in self.nn_params])
            loss = lossL2 + lossReg * self.weight_decay
            self.summaries += [
                tf.scalar_summary('mean_squared_diff_loss_with_reg',
                                  loss)]

        return loss

    def define_training(self, loss):
        with tf.variable_scope('train'):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.global_step = tf.Variable(0, name='global_step',
                                           trainable=False)
            train_op = optimizer.minimize(loss,
                                          global_step=self.global_step)
            return train_op

    def define_action_grad(self):
        with tf.variable_scope('getActionGradient'):
            return tf.gradients(self.nn, self.actions_pl)

    def run_train(self, inputs, actions, targets):
        _, _, step, summaries = self.sess.run([self.nn,
                                               self.train_op,
                                               self.global_step,
                                               self.summary_op],
                                              feed_dict={
            self.input_pl: inputs,
            self.actions_pl: actions,
            self.td_targets_pl: targets,
            self.isTraining: True
        })
        self.writer.add_summary(summaries, step)
        self.writer.flush()

        return step


    def run_predict(self, inputs, action):
        return self.sess.run(self.nn, feed_dict={
            self.input_pl: inputs,
            self.actions_pl: action,
            self.isTraining: False
        })

    def run_predict_target(self, inputs, action):
        return self.sess.run(self.target_nn, feed_dict={
            self.target_input_pl: inputs,
            self.target_actions_pl: action,
            self.isTraining: False
        })

    def run_get_action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.input_pl: inputs,
            self.actions_pl: actions
        })

    def run_update_target_nn(self):
        self.sess.run(self.target_nn_update_op)
