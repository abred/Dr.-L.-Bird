import tensorflow as tf

import tfUtils as tfu


class Critic:
    O = 1
    H1 = 16
    H2 = 32
    H3 = 256
    learning_rate = 0.0001
    mini_batch_size = 16
    tau = 0.001
    train_dir = 'data'
    state_dim_x = 105
    state_dim_y = 60
    weight_decay = 0.001
    actions_dim = 3

    def __init__(self, sess, out_dir, glStep):
        self.sess = sess
        self.summaries = []
        self.out_dir = out_dir

        self.global_step = glStep
        with tf.variable_scope('Critic'):
            self.isTraining = tf.placeholder(tf.bool)

            # Critic Network
            prevTrainVarCount = len(tf.trainable_variables())
            self.input_pl, self.actions_pl, self.nn, self.middle = self.defineNN()
            self.nn_params = tf.trainable_variables()[prevTrainVarCount:]

            # Target Network
            with tf.variable_scope('target'):
                prevTrainVarCount = len(tf.trainable_variables())
                self.target_input_pl, self.target_actions_pl, self.target_nn, _ =\
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
            self.summary_op = tf.merge_summary(summaries)
            self.writer = tf.train.SummaryWriter(out_dir, sess.graph)

    def defineNN(self, isTargetNN=False):
        images = tf.placeholder(
            tf.float32,
            shape=[None, self.state_dim_x*self.state_dim_y],
            name='input')
        actions = tf.placeholder(tf.float32,
                                 shape=[None, self.actions_dim],
                                 name='ActorActions')

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
        # h1 = tf.Print(h1, [h1], message=str(isTargetNN)+"critic h1: ", summarize=1000)
        h1 = tf.check_numerics(h1, str(isTargetNN)+"critic h1: ")

        self.summaries += s
        h2, s = tfu.convReluLayer(h1,
                                  self.H1, self.H2,
                                  fh=4, fw=4,
                                  strh=2, strw=2,
                                  scopeName='h2',
                                  isTargetNN=isTargetNN,
                                  is_training=self.isTraining)
        # h2 = tf.Print(h2, [h2], message=str(isTargetNN)+"critic h2: ", summarize=1000)
        h2 = tf.check_numerics(h2, str(isTargetNN)+"critic h2: ")
        self.summaries += s
        h2_f = tf.reshape(h2, [-1, 14*8*self.H2], name='flatten')
        h2_a = tf.concat(1, [
            tf.reshape(h2, [-1, 14*8*self.H2], name='flatten'),
            actions])

        h3, s, w, b = tfu.fullyConRelu(h2_a,
                                    8*14*self.H2+self.actions_dim, self.H3,
                                    scopeName='h3', isTargetNN=isTargetNN,
                                    is_training=self.isTraining)
        if not isTargetNN:
            self.w = w
            self.h3 = h3
            self.h2 = h2_a
            self.h1 = h1
            self.b = b
        h3 = tf.check_numerics(h3, str(isTargetNN)+"critic h3: ")
        self.summaries += s

        o, s = tfu.fullyCon(h3, self.H3, self.O,
                            scopeName='out', isTargetNN=isTargetNN,
                            is_training=self.isTraining)
        self.summaries += s
        return images, actions, o, h2_a

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
            # lossL2 = tf.Print(lossL2, [lossL2], message="lossl2: ")
            self.summaries += [tf.scalar_summary('mean_squared_diff_loss',
                                                 lossL2)]

            regs = []
            for v in self.nn_params:
                if "w" in v.name:
                    regs.append(tf.nn.l2_loss(v))
            lossReg = tf.add_n(regs)
            # lossReg = tf.Print(lossReg, [lossReg], message="lossreg: ")
            self.summaries += [
                tf.scalar_summary('mean_squared_diff_loss_reg',
                                  lossReg)]
            loss = lossL2 + lossReg * self.weight_decay
            self.summaries += [
                tf.scalar_summary('mean_squared_diff_loss_with_reg',
                                  loss)]

        # return loss
        return loss

    def define_training(self, loss):
        with tf.variable_scope('train'):
            # loss = tf.Print(loss, [loss], "opt loss: ")
            optimizer = tf.train.AdamOptimizer(self.learning_rate, epsilon=0.1)
            # optimizer = tf.train.RMSPropOptimizer(self.learning_rate)

            # grads_vars = optimizer.compute_gradients(loss,
            #                                          var_list=self.nn_params)
            # tmp = tf.Print(loss, [g for (g,v) in grads_vars],
            #                "grads: ")
            # train_op = optimizer.apply_gradients(grads_vars,
            #                                      global_step=self.global_step)
            train_op = optimizer.minimize(loss,
                                          global_step=self.global_step)
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
            _, summaries = self.sess.run([self.train_op,
                                          self.summary_op],
                                         feed_dict={
                self.input_pl: inputs,
                self.actions_pl: actions,
                self.td_targets_pl: targets,
                self.isTraining: True
            })
            self.writer.add_summary(summaries, step)
            self.writer.flush()
        else:
            self.sess.run([self.train_op],
                          feed_dict={
                self.input_pl: inputs,
                self.actions_pl: actions,
                self.td_targets_pl: targets,
                self.isTraining: True
            })

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
        return self.sess.run([self.middle,self.nn,self.action_grads], feed_dict={
            self.input_pl: inputs,
            self.actions_pl: actions,
            self.isTraining: False
        })

    def run_get_gradients(self, inputs, actions, targets):
        return zip(self.nn_params, self.sess.run(self.grads, feed_dict={
            self.input_pl: inputs,
            self.actions_pl: actions,
            self.td_targets_pl: targets,
            self.isTraining: False
        }))

    def run_get_loss(self, inputs, actions, targets):
        return self.sess.run(self.loss_op, feed_dict={
            self.input_pl: inputs,
            self.actions_pl: actions,
            self.td_targets_pl: targets,
            self.isTraining: False
        })
    def run_update_target_nn(self):
        self.sess.run(self.target_nn_update_op)
