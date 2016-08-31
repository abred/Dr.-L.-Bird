import os.path
import time

import tensorflow as tf


def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


class ActorNetwork:
    I = 832 * 448
    O = 6
    H1_F = 32
    H2_F = 64
    H3_F = 128
    H4_F = 256
    H5_F = 512
    H6_F = 512
    H7_F = 1024

    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
    flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')
    # flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
    # flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
    flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')
    flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                         'for unit testing.')

    def __init__(self):
        self.images_placeholder = tf.placeholder(tf.float32,
                                                 shape=[None, 448*832])
        self.actions_placeholder = tf.placeholder(tf.float32,
                                                  shape=[None, 3])
        self.returns_placeholder = tf.placeholder(tf.float32,
                                                  shape=[None])

        self.forward_op = self.inference(self.images_placeholder)

        self.loss_op = self.loss(self.forward_op,
                                 self.actions_placeholder,
                                 self.returns_placeholder)

        self.train_op = self.training(self.loss_op, self.FLAGS.learning_rate)
        self.summary_op = tf.merge_all_summaries()

        init = tf.initialize_all_variables()
        self.saver = tf.train.Saver()

        self.sess = tf.Session()
        self.summary_writer = tf.train.SummaryWriter(self.FLAGS.train_dir,
                                                     self.sess.graph)

        self.sess.run(init)

        # self.h = self.sess.partial_run_setup([self.forward_op,
        #                                       self.train_op,
        #                                       self.loss_op],
        #                                      [self.images_placeholder,
        #                                       self.h7_keepprob,
        #                                       self.actions_placeholder,
        #                                       self.returns_placeholder])

    def run_inference(self, inputs):
        feed_dict = {
            self.images_placeholder: inputs,
            self.h7_keepprob: 0.5
        }
        # return self.sess.partial_run(self.h, self.forward_op,
        #                              feed_dict=feed_dict)
        return self.sess.run(self.forward_op, feed_dict=feed_dict)

    def inference(self, images):
        """Build model up to where it may be used for inference.

      Args:
        images: Images placeholder, from inputs().

      Returns:
        softmax_linear: Output tensor with the computed logits.
        """
        x = tf.reshape(images, [-1, 448, 832, 1])
        # hidden 1
        with tf.name_scope('h1'):
            weights = weight_variable([5, 5, 1, self.H1_F], 'w01')
            biases = bias_variable([self.H1_F], 'b01')
            h1_conv = tf.nn.relu(conv2d(x, weights) + biases)
            h1_pool = max_pool_2x2(h1_conv)
        # hidden 2
        with tf.name_scope('h2'):
            weights = weight_variable([3, 3, self.H1_F, self.H2_F], 'w12')
            biases = bias_variable([self.H2_F], 'b12')
            h2_conv = tf.nn.relu(conv2d(h1_pool, weights) + biases)
            h2_pool = max_pool_2x2(h2_conv)
        # hidden 3
        with tf.name_scope('h3'):
            weights = weight_variable([3, 3, self.H2_F, self.H3_F], 'w23')
            biases = bias_variable([self.H3_F], 'b23')
            h3_conv = tf.nn.relu(conv2d(h2_pool, weights) + biases)
            h3_pool = max_pool_2x2(h3_conv)
        # hidden 4
        with tf.name_scope('h4'):
            weights = weight_variable([3, 3, self.H3_F, self.H4_F], 'w34')
            biases = bias_variable([self.H4_F], 'b34')
            h4_conv = tf.nn.relu(conv2d(h3_pool, weights) + biases)
            h4_pool = max_pool_2x2(h4_conv)
        # hidden 5
        with tf.name_scope('h5'):
            weights = weight_variable([3, 3, self.H4_F, self.H5_F], 'w45')
            biases = bias_variable([self.H5_F], 'b45')
            h5_conv = tf.nn.relu(conv2d(h4_pool, weights) + biases)
            h5_pool = max_pool_2x2(h5_conv)
        # hidden 6
        with tf.name_scope('h6'):
            weights = weight_variable([3, 3, self.H5_F, self.H6_F], 'w56')
            biases = bias_variable([self.H6_F], 'b56')
            h6_conv = tf.nn.relu(conv2d(h5_pool, weights) + biases)
            h6_pool = max_pool_2x2(h6_conv)
        # hidden 7
        with tf.name_scope('h7'):
            weights = weight_variable([7*13*512, self.H7_F], 'w67')
            biases = bias_variable([self.H7_F], 'b67')
            h6_flat = tf.reshape(h6_pool, [-1, 7*13*512])
            h7_fc = tf.nn.relu(tf.matmul(h6_flat, weights) + biases)
            self.h7_keepprob = tf.placeholder(tf.float32)
            h7_drop = tf.nn.dropout(h7_fc, self.h7_keepprob)
        # output
        with tf.name_scope('out'):
            weights = weight_variable([self.H7_F, self.O], 'w78')
            biases = bias_variable([self.O], 'b78')
            o_fc = tf.matmul(h7_drop, weights) + biases

        return o_fc

    def run_loss(self, outputs, actions, returns):
        feed_dict = {
            self.actions_placeholder: actions,
            self.returns_placeholder: returns,
}
        return self.sess.run(self.loss_op, feed_dict=feed_dict)

    def loss(self, outputs, actions, returns):
        """Calculates the loss from the output and the actions.

        Args:
        outputs: output tensor, float - [batch_size, 6 (3*mu + 3*sigma)].
        actions: action tensor, float - [batch_size, 3 ].

        Returns:
        loss: Loss tensor of type float.
        """
        mu, sigma = tf.split(1, 2, outputs)
        E = tf.mul(-tf.truediv(tf.square(tf.sub(actions, mu)),
                               tf.mul(2.0, tf.square(sigma))),
                   returns)
        loss = tf.reduce_mean(E, name='log_gauss_loss_mean')
        return loss

    def training(self, loss, learning_rate):
        """Sets up the training Ops.

        Creates a summarizer to track the loss over time in TensorBoard.

        Creates an optimizer and applies the gradients to all trainable
        variables.

        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train.

        Args:
        loss: Loss tensor, from loss().
        learning_rate: The learning rate to use for gradient descent.

        Returns:
        train_op: The Op for training.
        """
        # Add a scalar summary for the snapshot loss.
        tf.scalar_summary(loss.op.name, loss)
        # Create the gradient descent optimizer with the given learning rate.
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        # Create a variable to track the global step.
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        # Use the optimizer to apply the gradients that minimize the loss
        # (and also increment the global step counter) as a single training
        # step.
        train_op = optimizer.minimize(loss, global_step=self.global_step)
        return train_op

    def run_training(self, states, actions, returns):
        start_time = time.time()
        feed_dict = {
            self.images_placeholder: states,
            self.h7_keepprob: 0.5,
            self.actions_placeholder: actions,
            self.returns_placeholder: returns,
        }

        # Run one step of the model.  The return values are the activations
        # from the `train_op` (which is discarded) and the `loss` Op.  To
        # inspect the values of your Ops or variables, you may include them
        # in the list passed to sess.run() and the value tensors will be
        # returned in the tuple from the call.
        _, loss_value = self.sess.run([self.train_op, self.loss_op],
                                      feed_dict=feed_dict)
        # _, loss_value = self.sess.partial_run(self.h,
                                              # [self.train_op, self.loss_op],
                                              # feed_dict=feed_dict)

        duration = time.time() - start_time

        currStep = tf.train.global_step(self.sess, self.global_step)
        print('Step {}: loss = {:.2f} ({:.3f} sec)'.format(
            currStep,
            loss_value,
            duration))
        # Update the events file.
        summary_str = self.sess.run(self.summary_op, feed_dict=feed_dict)
        self.summary_writer.add_summary(summary_str, currStep)
        self.summary_writer.flush()

        # Save a checkpoint and evaluate the model periodically.
        if ((currStep + 1) % 100 == 0 or
            (currStep + 1) == self.FLAGS.max_steps):
            checkpoint_file = os.path.join(self.FLAGS.train_dir,
                                           'checkpoint')
            self.saver.save(self.sess, checkpoint_file, global_step=currStep)
