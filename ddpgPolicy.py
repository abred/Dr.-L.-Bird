import numpy as np

from ddpgActor import Actor

from ddpgCritic import Critic


class DDPGPolicy:
    def __init__(self, sess, resume=False):
        self.actor = Actor(sess)
        self.critic = Critic(sess)

    def getAction(self, state, noise):
        a = self.actor.run_predict(state) + noise
        print("Next action: {}\n".format(a))
        return a

    def update(self, states, actions, targets):
        self.critic.run_train(states, actions, targets)
        a_grad = self.critic.run_get_action_gradients(states, actions)
        self.actor.run_train(states, a_grad[0])

    def update_targets(self):
        self.critic.run_update_target_nn()
        self.actor.run_update_target_nn()

    def predict_target_nn(self, state):
        a = self.actor.run_predict_target(state)
        return self.critic.run_predict_target(state, a)
