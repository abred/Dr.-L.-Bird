import sys
import numpy as np

from ddpgActor import Actor

from ddpgCritic import Critic

import tensorflow as tf

class DDPGPolicy:
    def __init__(self, sess, out_dir, glStep, params):
        self.actor = Actor(sess, out_dir, params)
        self.critic = Critic(sess, out_dir, glStep, params)

        self.cnt = 0
        self.out_dir = out_dir

    def getActions(self, state):
        a = self.actor.run_predict(state)
        return a

    def update(self, states, actions, targets):
        step, out, loss = self.critic.run_train(states, actions, targets)
        print("step: {}, loss: {}".format(step, loss))
        ac = self.actor.run_predict(states)
        a_grad = self.critic.run_get_action_gradients(states, ac)
        self.actor.run_train(states, a_grad[0], step)
        return out

    def update_targets(self):
        self.critic.run_update_target_nn()
        self.actor.run_update_target_nn()

    def predict_target_nn(self, state):
        a = self.actor.run_predict_target(state)
        return self.critic.run_predict_target(state, a)

    def predict_nn(self, state):
        a = self.actor.run_predict(state)
        return self.critic.run_predict(state, a)
