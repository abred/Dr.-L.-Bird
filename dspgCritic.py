import tensorflow as tf

import tfUtils as tfu


class Critic:
    O = 1
    H1 = 16
    H2 = 32
    H3 = 256
    learning_rate = 0.001
    mini_batch_size = 8
    tau = 0.001
    train_dir = 'data'
    state_dim_x = 105
    state_dim_y = 60
    weight_decay = 0.001

    def __init__(self, sess, out_dir, glStep):
        self.sess = sess
        self.summaries = []

        self.global_step = glStep
        with tf.variable_scope('Critic'):
            self.isTraining = tf.placeholder(tf.bool)

            # Critic Network
            prevTrainVarCount = len(tf.trainable_variables())
            self.input_pl, self.nn = self.defineNN()
            self.nn_params = tf.trainable_variables()[prevTrainVarCount:]

            # Target Network
            with tf.variable_scope('target'):
                prevTrainVarCount = len(tf.trainable_variables())
                self.target_input_pl, self.target_nn =\
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
        return images, o

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
            train_op = optimizer.minimize(loss,
                                          global_step=self.global_step)
            return train_op

    def run_train(self, inputs, targets):
        _, step, summaries = self.sess.run([self.train_op,
                                            self.global_step,
                                            self.summary_op],
                                           feed_dict={
            self.input_pl: inputs,
            self.td_targets_pl: targets,
            self.isTraining: True
        })
        self.writer.add_summary(summaries, step)
        self.writer.flush()

        return step

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
