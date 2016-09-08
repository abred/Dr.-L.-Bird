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
    learning_rate = 0.0001
    mini_batch_size = 4
    tau = 0.001
    train_dir = 'data'
    actions_dim = 3
    weight_decay = 0.01

    def __init__(self, sess):
        self.sess = sess

        # Critic Network
        prevTrainVarCount = len(tf.trainable_variables())
        self.input_pl, self.actions_pl, self.nn = self.defineNN('Critic')
        self.nn_params = tf.trainable_variables()[prevTrainVarCount:]

        # Target Network
        prevTrainVarCount = len(tf.trainable_variables())
        self.target_input_pl, self.target_actions_pl, \
            self.target_nn = self.defineNN('Critic/target')
        self.target_nn_params = tf.trainable_variables()[prevTrainVarCount:]
        for i in range(len(self.nn_params)):
            tf.Variable.assign(self.target_nn_params[i],
                               self.nn_params[i].initialized_value())
        self.target_nn_update_op = self.define_update_target_nn_op()

        # Network target (y_i)
        self.td_targets = tf.placeholder(tf.float32, [None, 1])

        self.loss_op = self.define_loss(self.td_targets, self.nn)
        self.train_op = self.define_training(self.loss_op)

        # Get the gradient of the net w.r.t. the action
        self.action_grads = tf.gradients(self.nn, self.actions_pl)


    def defineNN(self, scope):
        with tf.variable_scope(scope):
            images = tf.placeholder(tf.float32,
                                    shape=[None, 448*832])
            actions = tf.placeholder(tf.float32,
                                     shape=[None, self.actions_dim])
            x = tf.reshape(images, [-1, 448, 832, 1])
            h1 = tfu.convReluPoolLayer(x, 1, self.H1, fh=5, fw=5, scopeName='h1')
            h2 = tfu.convReluPoolLayer(h1, self.H1, self.H2, scopeName='h2')
            h3 = tfu.convReluPoolLayer(h2, self.H2, self.H3, scopeName='h3')
            h4 = tfu.convReluPoolLayer(h3, self.H3, self.H4, scopeName='h4')
            h5 = tfu.convReluPoolLayer(h4, self.H4, self.H5, scopeName='h5')
            h6 = tfu.convReluPoolLayer(h5, self.H5, self.H6, scopeName='h6')

            h6_a = tf.concat(1, [tf.reshape(h6, [-1, 7*13*self.H6]), actions])
            h7 = tfu.fullyConReluDrop(h6_a, 7*13*self.H6+3, self.H7,
                                      scopeName='h7')

            with tf.variable_scope('out'):
                weights = tfu.weight_variable_unit([self.H7, self.O], 'w')
                biases = tfu.bias_variable([self.O], 'b')
                o = tf.matmul(h7, weights) + biases
            return images, actions, o

    def define_update_target_nn_op(self):
        return \
            [self.target_nn_params[i].assign(
                tf.mul(self.nn_params[i], self.tau) +
                tf.mul(self.target_nn_params[i], 1.0 - self.tau))
             for i in range(len(self.target_nn_params))]

    def define_loss(self, targets, qVals):
        lossL2 = tfu.mean_squared_diff(targets, qVals)
        lossReg = tf.add_n([tf.nn.l2_loss(v) for v in self.nn_params])
        return lossL2 + lossReg * self.weight_decay

    def define_training(self, loss):
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss,
                                      global_step=self.global_step)
        return train_op

    def run_train(self, inputs, actions, targets):
        return self.sess.run([self.nn, self.train_op], feed_dict={
            self.input_pl: inputs,
            self.actions_pl: actions,
            self.td_targets: targets
        })

    def run_predict(self, inputs, action):
        return self.sess.run(self.nn, feed_dict={
            self.input_pl: inputs,
            self.actions_pl: action
        })

    def run_predict_target(self, inputs, action):
        return self.sess.run(self.target_nn, feed_dict={
            self.target_input_pl: inputs,
            self.target_actions_pl: action
        })

    def run_get_action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.input_pl: inputs,
            self.actions_pl: actions
        })

    def run_update_target_nn(self):
        self.sess.run(self.target_nn_update_op)
