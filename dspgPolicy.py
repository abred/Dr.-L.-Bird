import numpy as np

from dspgActor import Actor

from dspgCritic import Critic


class DSPGPolicy:
    def __init__(self, sess, out_dir, resume=False):
        self.actor = Actor(sess, out_dir)
        self.critic = Critic(sess, out_dir)

    def getAction(self, state, noise):
        a = self.actor.run_predict(state)
        a_scaled = np.copy(a)
        a_scaled[0][0] *= 50.0
        a_scaled[0][1] *= 9000.0
        a_scaled[0][2] *= 4000.0
        a_scaled[0][3] *= 50.0
        a_scaled[0][4] *= 9000.0
        a_scaled[0][5] *= 4000.0
        mu = a_scaled[0][:3]
        sigma = a_scaled[0][3:]
        act = sigma * np.random.randn(3) + mu
        print("Next action: {}\n".format(a))
        print("Next action(mu): {}\n".format(mu))
        print("Next action(sigma): {}\n".format(sigma))
        # a_scaled +=  noise
        print("Next action(noised): {}\n".format(act))

        return a, act

    def update(self, states, actions, targets):
        print(actions.shape);
        step = self.critic.run_train(states, actions, targets)
        ac = self.actor.run_predict(states)
        a_grad = self.critic.run_get_action_gradients(states, ac[:,:3])
        print("action gradients: {}".format(a_grad[0]))
        self.actor.run_train(states, a_grad[0], actions, step)

    def update_targets(self):
        self.critic.run_update_target_nn()
        self.actor.run_update_target_nn()

    def predict_target_nn(self, state):
        a = self.actor.run_predict_target(state)
        return self.critic.run_predict_target(state, a[:,:3])
