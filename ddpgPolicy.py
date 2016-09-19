import numpy as np

from ddpgActor import Actor

from ddpgCritic import Critic


class DDPGPolicy:
    def __init__(self, sess, out_dir, resume=False):
        self.actor = Actor(sess, out_dir)
        self.critic = Critic(sess, out_dir)

    def getAction(self, state, noise):
        a = self.actor.run_predict(state)
        a_scaled = np.copy(a)
        a_scaled[0][0] *= -50.0
        a_scaled[0][1] *= 50.0
        a_scaled[0][2] *= 4000.0
        print("Next action: {}\n".format(a))
        print("Next action(scaled): {}\n".format(a_scaled))
        a_scaled +=  noise
        print("Next action(noised): {}\n".format(a_scaled))

        return a, a_scaled

    def update(self, states, actions, targets):
        step = self.critic.run_train(states, actions, targets)
        ac = self.actor.run_predict(states)
        a_grad = self.critic.run_get_action_gradients(states, ac)
        self.actor.run_train(states, a_grad[0], step)

    def update_targets(self):
        self.critic.run_update_target_nn()
        self.actor.run_update_target_nn()

    def predict_target_nn(self, state):
        a = self.actor.run_predict_target(state)
        return self.critic.run_predict_target(state, a)
