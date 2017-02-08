from tensorflow.python.framework import ops
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
    learning_rate = 0.00005
    # mini_batch_size = 16
    tau = 0.001
    train_dir = 'data'
    state_dim_x = 210
    state_dim_y = 120
    col_channels = 3
    weight_decay = 0.001
    actions_dim = 3
    vgg_state_dim = 224

    def __init__(self, sess, out_dir, glStep, useVGG=False, top=None,
                 useDrop=False, batchnorm=False):
        self.sess = sess
        self.summaries = []
        self.out_dir = out_dir
        self.useVGG = useVGG
        self.batchnorm = batchnorm
        self.useDrop = useDrop
        self.top = top
        print("useDrop critic", self.useDrop)
        print("top critic", self.top)

        self.global_step = glStep
        if useVGG:
                self.vggsaver = tf.train.import_meta_graph(
                    '/home/s7550245/convNet/vgg-model.meta',
                    import_scope="Critic/VGG",
                    clear_devices=True)

                _VARSTORE_KEY = ("__variable_store",)
                varstore = ops.get_collection(_VARSTORE_KEY)[0]
                variables = tf.get_collection(
                    ops.GraphKeys.GLOBAL_VARIABLES,
                    scope="Critic/VGG")

                for v in variables:
                    varstore._vars[v.name.split(":")[0]] = v
                    # print(v.name)

        with tf.variable_scope('Critic'):
            self.keep_prob = tf.placeholder(tf.float32)
            self.isTraining = tf.placeholder(tf.bool)

            prevTrainVarCount = len(tf.trainable_variables())

            # Critic Network
            if useVGG:
                with tf.variable_scope("VGG") as scope:
                    scope.reuse_variables()
                    self.input_pl, self.nnB = self.defineNNVGG(top=top)
                with tf.variable_scope("VGG") as scope:
                    self.actions_pl, self.nn = self.defineNNVGGTop(self.nnB,
                                                                   top=top)
                print(self.nn)
                self.nn_params = tf.trainable_variables()[prevTrainVarCount:]

                # Target Network
                with tf.variable_scope('target'):
                    prevTrainVarCount = len(tf.trainable_variables())
                    self.target_input_pl, self.nnBT = \
                        self.defineNNVGG(top=top, isTargetNN=True)
                    self.target_actions_pl, self.target_nn = \
                        self.defineNNVGGTop(self.nnBT, top=top,
                                            isTargetNN=True)
                    self.target_nn_params = \
                        tf.trainable_variables()[prevTrainVarCount:]
                    with tf.variable_scope('init'):
                        for i in range(len(self.target_nn_params)):
                            for j in range(len(self.nn_params)):
                                p1 = self.nn_params[j]
                                p2 = self.target_nn_params[i]
                                if p2.name.split("/")[-1] in p1.name:
                                    tf.Variable.assign(
                                        self.target_nn_params[i],
                                        self.nn_params[j].initialized_value())
                                    break
                    self.target_nn_update_op = \
                        self.define_update_target_nn_op()
                self.loss_op = self.define_loss()
                self.train_op = self.defineTrainingVGG(self.loss_op)
            else:
                self.input_pl, self.actions_pl, self.nn = self.defineNN()
                self.nn_params = tf.trainable_variables()[prevTrainVarCount:]

                # Target Network
                with tf.variable_scope('target'):
                    prevTrainVarCount = len(tf.trainable_variables())
                    self.target_input_pl, self.target_actions_pl, \
                        self.target_nn = self.defineNN(isTargetNN=True)
                    self.target_nn_params = \
                        tf.trainable_variables()[prevTrainVarCount:]
                    with tf.variable_scope('init'):
                        for i in range(len(self.nn_params)):
                            tf.Variable.assign(
                                self.target_nn_params[i],
                                self.nn_params[i].initialized_value())
                    self.target_nn_update_op =self.define_update_target_nn_op()
                self.loss_op = self.define_loss()
                self.train_op = self.define_training(self.loss_op)

            self.action_grads = self.define_action_grad()
            self.grads = self.define_grads()

            # summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope="Critic")
            self.summary_op = tf.summary.merge(self.summaries)
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

    def defineNNVGG(self, top=None, isTargetNN=False):
        # with tf.variable_scope('inf') as scope:
            inputs = tf.placeholder(
                tf.float32,
                shape=[None,
                       self.vgg_state_dim,
                       self.vgg_state_dim,
                       self.col_channels],
                name='input')

            h1, s = tfu.convReluLayer(inputs,
                                      3, 64,
                                      scopeName='block1_conv1',
                                      isTargetNN=isTargetNN,
                                      is_training=self.isTraining,
                                      norm=self.batchnorm)
            self.summaries += s
            if top == 1:
                return inputs, h1

            h2, s = tfu.convReluPoolLayer(h1,
                                          64, 64,
                                          scopeName='block1_conv2',
                                          isTargetNN=isTargetNN,
                                          is_training=self.isTraining,
                                          norm=self.batchnorm)
            self.summaries += s
            if top == 2:
                return inputs, h2

            h3, s = tfu.convReluLayer(h2,
                                      64, 128,
                                      scopeName='block2_conv1',
                                      isTargetNN=isTargetNN,
                                      is_training=self.isTraining,
                                      norm=self.batchnorm)
            self.summaries += s
            if top == 3:
                return inputs, h3

            h4, s = tfu.convReluPoolLayer(h3,
                                          128, 128,
                                          scopeName='block2_conv2',
                                          isTargetNN=isTargetNN,
                                          is_training=self.isTraining,
                                          norm=self.batchnorm)
            self.summaries += s
            if top == 4:
                return inputs, h4

            h5, s = tfu.convReluLayer(h4,
                                      128, 256,
                                      scopeName='block3_conv1',
                                      isTargetNN=isTargetNN,
                                      is_training=self.isTraining,
                                      norm=self.batchnorm)
            self.summaries += s
            if top == 5:
                return inputs, h5

            h6, s = tfu.convReluLayer(h5,
                                      256, 256,
                                      scopeName='block3_conv2',
                                      isTargetNN=isTargetNN,
                                      is_training=self.isTraining,
                                      norm=self.batchnorm)
            self.summaries += s
            if top == 6:
                return inputs, h6

            h7, s = tfu.convReluPoolLayer(h6,
                                          256, 256,
                                          scopeName='block3_conv3',
                                          isTargetNN=isTargetNN,
                                          is_training=self.isTraining,
                                          norm=self.batchnorm)
            self.summaries += s
            if top == 7:
                return inputs, h7

            h8, s = tfu.convReluLayer(h7,
                                      256, 512,
                                      scopeName='block4_conv1',
                                      isTargetNN=isTargetNN,
                                      is_training=self.isTraining,
                                      norm=self.batchnorm)
            self.summaries += s
            if top == 8:
                return inputs, h8

            h9, s = tfu.convReluLayer(h8,
                                      512, 512,
                                      scopeName='block4_conv2',
                                      isTargetNN=isTargetNN,
                                      is_training=self.isTraining,
                                      norm=self.batchnorm)
            self.summaries += s
            if top == 9:
                return inputs, h9

            h10, s = tfu.convReluPoolLayer(h9,
                                           512, 512,
                                           scopeName='block4_conv3',
                                           isTargetNN=isTargetNN,
                                           is_training=self.isTraining,
                                           norm=self.batchnorm)
            self.summaries += s
            if top == 10:
                return inputs, h10

            h11, s = tfu.convReluLayer(h10,
                                       512, 512,
                                       scopeName='block5_conv1',
                                       isTargetNN=isTargetNN,
                                       is_training=self.isTraining,
                                       norm=self.batchnorm)
            self.summaries += s
            if top == 11:
                return inputs, h11

            h12, s = tfu.convReluLayer(h11,
                                       512, 512,
                                       scopeName='block5_conv2',
                                       isTargetNN=isTargetNN,
                                       is_training=self.isTraining,
                                       norm=self.batchnorm)
            self.summaries += s
            if top == 12:
                return inputs, h12

            h13, s = tfu.convReluPoolLayer(h12,
                                           512, 512,
                                           scopeName='block5_conv3',
                                           isTargetNN=isTargetNN,
                                           is_training=self.isTraining,
                                           norm=self.batchnorm)
            self.summaries += s
            if top == 13:
                return inputs, h13

            remSz = int(self.state_dim / 2**5)
            h13_f = tf.reshape(h13, [-1, remSz*remSz*512],
                               name='flatten')

            if self.useDrop:
                h14, s = tfu.fullyConReluDrop(h13_f, remSz*remSz*512, 4096,
                                              self.keep_prob, scopeName='fc1',
                                              norm=self.batchnorm,
                                              isTargetNN=isTargetNN,
                                              is_training=self.isTraining)
            else:
                h14, s = tfu.fullyConRelu(h13_f, remSz*remSz*512, 4096,
                                          scopeName='fc1',
                                          norm=self.batchnorm,
                                          isTargetNN=isTargetNN,
                                          is_training=self.isTraining)
            self.summaries += s

            if self.useDrop:
                h15, s = tfu.fullyConReluDrop(h14, 4096, 4096,
                                              self.keep_prob, scopeName='fc2',
                                              norm=self.batchnorm,
                                              isTargetNN=isTargetNN,
                                              is_training=self.isTraining)
            else:
                h15, s = tfu.fullyConReluDrop(h14, 4096, 4096,
                                              scopeName='fc2',
                                              norm=self.batchnorm,
                                              isTargetNN=isTargetNN,
                                              is_training=self.isTraining)
            self.summaries += s

            o, s = tfu.fullyCon(h15, 4096, self.O, scopeName='predictions',
                            isTargetNN=isTargetNN,
                            is_training=self.isTraining)
            self.summaries += s
            r, th, t = tf.split(1, 3, o)
            r_o = 50.0 * tf.sigmoid(r)
            th_o = 9000.0 * tf.sigmoid(th)
            t_o = 4000.0 * tf.sigmoid(t)
            if not isTargetNN:
                self.summaries += [
                    tf.summary.histogram(tf.get_default_graph().unique_name(
                        'out' + '/radius_action',
                        mark_as_used=False), r_o),
                    tf.summary.histogram(tf.get_default_graph().unique_name(
                        'out' + '/theta_action',
                        mark_as_used=False), th_o),
                    tf.summary.histogram(tf.get_default_graph().unique_name(
                        'out' + '/time_delay_action',
                        mark_as_used=False), t_o),
                    tf.summary.histogram(tf.get_default_graph().unique_name(
                        'out' + '/radius_action_before_sig',
                        mark_as_used=False), r),
                    tf.summary.histogram(tf.get_default_graph().unique_name(
                        'out' + '/theta_action_before_sig',
                        mark_as_used=False), th),
                    tf.summary.histogram(tf.get_default_graph().unique_name(
                        'out' + '/time_delay_action_before_sig',
                        mark_as_used=False), t)
                ]
            # outputs = tf.concat(1, [r_o, th_o, t_o])
            outputs = tf.sigmoid(o)
            return inputs, outputs

    def defineNNVGGTop(self, bottom, top=13, isTargetNN=False):
        with tf.variable_scope('top') as scope:
            actions = tf.placeholder(tf.float32,
                                     shape=[None, self.actions_dim],
                                     name='ActorActions')

            print(bottom)
            if top == 1:
                numPool = 0
                self.HFC1VGG = 256
                self.HFC2VGG = 256
            if top <= 2:
                numPool = 1
                H = 64
                self.HFC1VGG = 256
                self.HFC2VGG = 256
            elif top <= 4:
                numPool = 2
                H = 128
                self.HFC1VGG = 512
                self.HFC2VGG = 512
            elif top <= 7:
                numPool = 3
                H = 256
                self.HFC1VGG = 1024
                self.HFC2VGG = 1024
            elif top <= 10:
                numPool = 4
                H = 512
                self.HFC1VGG = 2048
                self.HFC2VGG = 2048
            elif top <= 13:
                numPool = 5
                H = 512
                self.HFC1VGG = 2048
                self.HFC2VGG = 2048
            remSz = int(self.vgg_state_dim / 2**numPool)
            h13_f = tf.reshape(bottom, [-1, remSz*remSz*H],
                               name='flatten')

            bottom_a = tf.concat(1, [
                tf.reshape(bottom, [-1, remSz*remSz*H], name='flatten'),
                actions])

            if self.useDrop:
                h14, s = tfu.fullyConReluDrop(bottom_a,
                                              remSz*remSz*H+self.actions_dim,
                                              self.HFC1VGG,
                                              self.keep_prob, scopeName='hfc1',
                                              norm=self.batchnorm,
                                              isTargetNN=isTargetNN,
                                              is_training=self.isTraining)
            else:
                h14, s = tfu.fullyConRelu(bottom_a,
                                          remSz*remSz*H+self.actions_dim,
                                          self.HFC1VGG,
                                          scopeName='hfc1',
                                          norm=self.batchnorm,
                                          isTargetNN=isTargetNN,
                                          is_training=self.isTraining)
            self.summaries += s

            if self.useDrop:
                h15, s = tfu.fullyConReluDrop(h14, self.HFC1VGG, self.HFC2VGG,
                                          self.keep_prob, scopeName='hfc2',
                                          norm=self.batchnorm,
                                          isTargetNN=isTargetNN,
                                          is_training=self.isTraining)
            else:
                h15, s = tfu.fullyConRelu(h14, self.HFC1VGG, self.HFC2VGG,
                                          scopeName='hfc2',
                                          norm=self.batchnorm,
                                          isTargetNN=isTargetNN,
                                          is_training=self.isTraining)
            self.summaries += s

            o, s = tfu.fullyCon(h15, self.HFC2VGG, self.O, scopeName='out',
                                isTargetNN=isTargetNN,
                                is_training=self.isTraining)
            self.summaries += s
            return actions, o

    def define_update_target_nn_op(self):
        with tf.variable_scope('update'):
            tau = tf.constant(self.tau, name='tau')
            invtau = tf.constant(1.0-self.tau, name='invtau')
            if self.useVGG:
                tmp = []
                for i in range(len(self.target_nn_params)):
                    for j in range(len(self.nn_params)):
                        p1 = self.nn_params[j]
                        p2 = self.target_nn_params[i]
                        if p2.name.split("/")[-1] in p1.name:
                            print(p1.name, p2.name)
                            tmp.append(self.target_nn_params[i].assign(
                                tf.mul(self.nn_params[j], tau) +
                                tf.mul(self.target_nn_params[i], invtau)))
                            break
                return tmp
            else:
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
                if "W" in v.name:
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

    def defineTrainingVGG(self, loss, conv=False):
        with tf.variable_scope('train'):
            variables = tf.get_collection(
                ops.GraphKeys.TRAINABLE_VARIABLES,
                scope="Critic/VGG")
            var_list = []
            for v in variables:
                if "block" not in v.name:
                    var_list.append(v)
                    # print(v.name)

                if conv:
                    if self.top <= 2:
                        if "block1_conv2" in v.name:
                            var_list.append(v)
                    elif self.top <= 4:
                        if "block2_conv2" in v.name:
                            var_list.append(v)
                    elif self.top <= 7:
                        if "block3_conv3" in v.name:
                            var_list.append(v)
                    elif self.top <= 13:
                        if "block4_conv3" in v.name:
                            var_list.append(v)
                    elif self.top > 13:
                        if "block5_conv3" in v.name:
                            var_list.append(v)
            for v in var_list:
                print(v.name)

            # optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            # optimizer = tf.train.RMSPropOptimizer(self.learning_rate)

            print(optimizer)
            return optimizer.minimize(loss, var_list=var_list,
                                      global_step=self.global_step)

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
        if (step+1) % 1 == 0:
            out, loss, _, summaries = self.sess.run([self.nn,
                                                     self.loss_op,
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
            out, loss, _ = self.sess.run(
                [self.nn, self.loss_op, self.train_op],
                feed_dict={
                    self.input_pl: inputs,
                    self.actions_pl: actions,
                    self.td_targets_pl: targets,
                    self.isTraining: True,
                    self.keep_prob: 0.5
                })
        print("loss: {}".format(loss))
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
