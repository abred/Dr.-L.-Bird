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
    mini_batch_size = 4
    tau = 0.001
    train_dir = 'data'
    actions_dim = 3

    def __init__(self, sess):
        self.sess = sess

        # Actor Network
        prevTrainVarCount = len(tf.trainable_variables())
        self.input_pl, self.nn = self.defineNN('Actor')
        self.nn_params = tf.trainable_variables()[prevTrainVarCount:]

        # Target Network
        prevTrainVarCount = len(tf.trainable_variables())
        self.target_input_pl, self.target_nn = self.defineNN('Actor/target')
        self.target_nn_params = tf.trainable_variables()[prevTrainVarCount:]
        for i in range(len(self.nn_params)):
            tf.Variable.assign(self.target_nn_params[i],
                               self.nn_params[i].initialized_value())
        self.target_nn_update_op = self.define_update_target_nn_op()

        self.critic_actions_gradient_pl = tf.placeholder(
            tf.float32,
            [None, self.actions_dim])
        self.actor_gradients = tf.gradients(self.nn,
                                            self.nn_params,
                                            # critic grad descent
                                            # here ascent -> negative
                                            -self.critic_actions_gradient_pl)

        # Optimization Op
        self.train_op = self.define_training()

    def defineNN(self, scope):
        with tf.variable_scope(scope):
            images = tf.placeholder(tf.float32,
                                    shape=[None, 448*832])
            x = tf.reshape(images, [-1, 448, 832, 1])
            h1 = tfu.convReluPoolLayer(x, 1, self.H1, fh=5, fw=5, scopeName='h1')
            h2 = tfu.convReluPoolLayer(h1, self.H1, self.H2, scopeName='h2')
            h3 = tfu.convReluPoolLayer(h2, self.H2, self.H3, scopeName='h3')
            h4 = tfu.convReluPoolLayer(h3, self.H3, self.H4, scopeName='h4')
            h5 = tfu.convReluPoolLayer(h4, self.H4, self.H5, scopeName='h5')
            h6 = tfu.convReluPoolLayer(h5, self.H5, self.H6, scopeName='h6')
            h6_f = tf.reshape(h6, [-1, 7*13*self.H6])
            h7 = tfu.fullyConReluDrop(h6_f, 7*13*self.H6, self.H7,
                                      scopeName='h7')

            with tf.variable_scope('out'):
                weights = tfu.weight_variable_unit([self.H7, self.O], 'w')
                biases = tfu.bias_variable([self.O], 'b')
                o_fc = tf.matmul(h7, weights) + biases
                x, y, t = tf.split(1, 3, o_fc)
                x_o = -50.0 * tf.sigmoid(0.05 * x)
                y_o = 50.0 * tf.sigmoid(0.02 * x)
                t_o = 1000.0 * tf.tanh(0.001 * t) + 1000.0
                outputs = tf.concat(1, [x_o, y_o, t_o])
            return images, outputs

    def define_update_target_nn_op(self):
        return \
            [self.target_nn_params[i].assign(
                tf.mul(self.nn_params[i], self.tau) +
                tf.mul(self.target_nn_params[i], 1.0 - self.tau))
             for i in range(len(self.target_nn_params))]

    def define_training(self):
        return tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.nn_params))

    def run_train(self, inputs, a_grad):
        return self.sess.run([self.nn, self.train_op], feed_dict={
            self.input_pl: inputs,
            self.critic_actions_gradient_pl: a_grad
        })

    def run_predict(self, inputs):
        return self.sess.run(self.nn, feed_dict={
            self.input_pl: inputs,
        })

    def run_predict_target(self, inputs):
        return self.sess.run(self.target_nn, feed_dict={
            self.target_input_pl: inputs
        })

    def run_update_target_nn(self):
        self.sess.run(self.target_nn_update_op)
