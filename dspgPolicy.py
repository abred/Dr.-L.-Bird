from dspgActor import Actor

from dspgCritic import Critic

import numpy as np


class DSPGPolicy:
    def __init__(self, sess, out_dir, glStep):
        self.actor = Actor(sess, out_dir)
        self.critic = Critic(sess, out_dir, glStep)

    def getActions(self, state):
        o = self.actor.run_predict(state)
        mu = o[:, :3]
        var = o[:, 3:]
        sigma = np.sqrt(var)
        print("mu:  {}".format(mu))
        print("var: {}".format(var))
        act = sigma * np.random.randn(o.shape[0]) + mu

        return act

    def update(self, states, actions, targets):
        step = self.critic.run_train(states, targets)
        # ac = self.getActions(states)
        advantage = targets - self.critic.run_predict(states)
        print("step: {}".format(step))
        print("advantages: {}".format(advantage))
        self.actor.run_train(states, advantage, actions, step)

    def update_targets(self):
        self.critic.run_update_target_nn()
        self.actor.run_update_target_nn()

    def predict_target_nn(self, state):
        # a = self.actor.run_predict_target(state)
        return self.critic.run_predict_target(state)
