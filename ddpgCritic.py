import tensorflow as tf

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
    learning_rate = 0.0001
    # mini_batch_size = 16
    tau = 0.001
    train_dir = 'data'
    state_dim_x = 210
    state_dim_y = 120
    col_channels = 3
    weight_decay = 0.001
    actions_dim = 3

    def __init__(self, sess, out_dir, glStep):
        self.sess = sess
        self.summaries = []
        self.out_dir = out_dir

        self.global_step = glStep
        with tf.variable_scope('Critic'):
            self.keep_prob = tf.placeholder(tf.float32)
            self.isTraining = tf.placeholder(tf.bool)

            # Critic Network
            prevTrainVarCount = len(tf.trainable_variables())
            self.input_pl, self.actions_pl, self.nn = self.defineNN()
            self.nn_params = tf.trainable_variables()[prevTrainVarCount:]

            # Target Network
            with tf.variable_scope('target'):
                prevTrainVarCount = len(tf.trainable_variables())
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
            self.action_grads = self.define_action_grad()
            self.grads = self.define_grads()
            self.train_op = self.define_training(self.loss_op)

            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope="Critic")
            self.summary_op = tf.summary.merge(summaries)
            self.writer = tf.summary.FileWriter(out_dir, sess.graph)

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

        h8, s = tfu.fullyConRelu(h7_a,
                                 13*7*self.H7+self.actions_dim, self.H8,
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
        with tf.variable_scope('loss2'):
            self.td_targets_pl = tf.placeholder(tf.float32, [None, 1],
                                             name='tdTargets')

            lossL2 = tfu.mean_squared_diff(self.td_targets_pl, self.nn)
            with tf.name_scope(''):
                self.summaries += [
                    tf.summary.scalar('mean_squared_diff_loss',
                                      lossL2)]
            regs = []
            for v in self.nn_params:
                if "w" in v.name:
                    regs.append(tf.nn.l2_loss(v))
            lossReg = tf.add_n(regs)
            with tf.name_scope(''):
                self.summaries += [
                    tf.summary.scalar('mean_squared_diff_loss_reg',
                                      lossReg)]

            loss = lossL2 + lossReg * self.weight_decay
            with tf.name_scope(''):
                self.summaries += [
                    tf.summary.scalar('mean_squared_diff_loss_with_reg',
                                      loss)]

        return loss

    def define_training(self, loss):
        with tf.variable_scope('train'):
            optimizer = tf.train.AdamOptimizer(self.learning_rate) #, epsilon=0.1)
            # optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
            train_op = optimizer.minimize(loss,
                                          global_step=self.global_step)

            # grad_vars = optimizer.compute_gradients(loss,
            #                                         var_list=self.nn_params)
            # gp_vars = []
            # for (g,v) in grad_vars:
            #     # gp = tf.Print(g, [g], 'grad: ', summarize=100000)
            #     gp = tf.check_numerics(g, "grad numerics")
            #     gp_vars.append((gp, v))
            # train_op = optimizer.apply_gradients(gp_vars)
            return train_op

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
            out, _, summaries = self.sess.run([self.nn,
                                               self.train_op,
                                               self.summary_op],
                                              feed_dict={
                self.input_pl: inputs,
                self.actions_pl: actions,
                self.td_targets_pl: targets,
                self.isTraining: True,
                self.keep_prob: 0.5
            })
            self.writer.add_summary(summaries, step)
            self.writer.flush()
        else:
            out, _ = self.sess.run([self.nn, self.train_op],
                                   feed_dict={
                                       self.input_pl: inputs,
                                       self.actions_pl: actions,
                                       self.td_targets_pl: targets,
                                       self.isTraining: True,
                                       self.keep_prob: 0.5
                                   })

        return step, out


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
