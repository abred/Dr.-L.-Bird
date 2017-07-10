import sys
import numpy as np

from ddpgActor import Actor

from ddpgCritic import Critic

import tensorflow as tf
from tensorflow.python.framework import ops

class DDPGPolicy:
    def __init__(self, sess, out_dir, glStep, params):
        if params['useVGG']:
            self.loadVGG()
            self.actor = Actor(sess, out_dir, params,
                               inputs=self.images)
            self.critic = Critic(sess, out_dir,
                                 glStep, params,
                                 inputs=self.images)
        else:
            self.actor = Actor(sess, out_dir, params)
            self.critic = Critic(sess, out_dir, glStep, params)

        self.cnt = 0
        self.out_dir = out_dir

    def getActions(self, state):
        a = self.actor.run_predict(state)
        return a

    def update(self, states, actions, targets):
        step, out, delta = self.critic.run_train(states, actions, targets)
        # print("step: {}, loss: {}".format(step, loss))
        # ac = self.actor.run_predict(states)
        # a_grad = self.critic.run_get_action_gradients(states, ac)
        self.actor.run_train(states, delta, step)
        return step, out, delta

    def update_targets(self):
        self.critic.run_update_target_nn()
        self.actor.run_update_target_nn()

    def predict_target_nn(self, state):
        a = self.actor.run_predict_target(state)
        return self.critic.run_predict_target(state, a)

    def predict_nn(self, state):
        a = self.actor.run_predict(state)
        return self.critic.run_predict(state, a)

    def loadVGG(self):
        self.images = tf.placeholder(
            tf.float32,
            shape=[None,
                   224,
                   224,
                   3],
            name='input')

        varTmp = tf.get_variable("tmp", shape=[1,1])
        self.vggsaver = tf.train.import_meta_graph(
            '/home/s7550245/vggFcNet/vgg-model.meta',
            import_scope="VGG",
            clear_devices=True,
            input_map={'input_1':self.images})

        _VARSTORE_KEY = ("__variable_store",)
        varstore = ops.get_collection(_VARSTORE_KEY)[0]
        variables = tf.get_collection(
            ops.GraphKeys.GLOBAL_VARIABLES,
            scope="VGG")

        for v in variables:
            varstore._vars[v.name.split(":")[0]] = v
