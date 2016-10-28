import numpy as np

from dspgActor import Actor

from dspgCritic import Critic


class DSPGPolicy:
    def __init__(self, sess, out_dir, glStep):
        self.actor = Actor(sess, out_dir)
        self.critic = Critic(sess, out_dir, glStep)

    def getAction(self, state, noise):
        a = self.actor.run_predict(state)
        a_scaled = np.copy(a)   #
        # a_scaled[0][0] *= 50.0
        # a_scaled[0][1] *= 9000.0
        # a_scaled[0][2] *= 4000.0
        # a_scaled[0][3] *= 25.0
        # a_scaled[0][4] *= 4500.0
        # a_scaled[0][5] *= 2000.0
        mu = a_scaled[0][:3]
        sigma = a_scaled[0][3:]
        if sigma[0] < 2.0:
            sigma[0] = 2.0
        if sigma[1] < 100.0:
            sigma[1] = 100.0
        if sigma[2] < 50.0:
            sigma[2] = 50.0

        act = sigma * np.random.randn(3) + mu
        print("Next action: {}\n".format(a))
        print("Next action(mu): {}\n".format(mu))
        print("Next action(sigma): {}\n".format(sigma))
        # a_scaled +=  noise
        print("Next action(noised): {}\n".format(act))

        if act[0] < 0.0:
            act[0] = 0.0
        if act[1] < 0.0:
            act[1] = 0.0
        if act[2] < 0.0:
            act[2] = 0.0
        if act[0] > 50.0:
            act[0] = 50.0
        if act[1] > 9000.0:
            act[1] = 9000.0
        if act[2] > 4000.0:
            act[2] = 4000.0
        return a, act

    def getActions(self, state):
        a = self.actor.run_predict(state)
        a_scaled = np.copy(a)
        for i in range(8):
            if a_scaled[i][3] < 2.0:
                a_scaled[i][3] = 2.0
            if a_scaled[i][4] < 100.0:
                a_scaled[i][4] = 100.0
            if a_scaled[i][5] < 50.0:
                a_scaled[i][5] = 50.0
        mu = a_scaled[:,:3]
        sigma = a_scaled[:,3:]
        for i in range(8):
            if sigma[i][0] < 2.0:
                sigma[i][0] = 2.0
            if sigma[i][1] < 100.0:
                sigma[i][1] = 100.0
            if sigma[i][2] < 50.0:
                sigma[i][2] = 50.0

        act = sigma * np.random.randn(8, 3) + mu

        for i in range(8):
            if act[i][0] < 0.0:
                act[i][0] = 0.0
            if act[i][1] < 0.0:
                act[i][1] = 0.0
            if act[i][2] < 0.0:
                act[i][2] = 0.0
            if act[i][0] > 50.0:
                act[i][0] = 50.0
            if act[i][1] > 9000.0:
                act[i][1] = 9000.0
            if act[i][2] > 4000.0:
                act[i][2] = 4000.0
        return a, act

    def update(self, states, actions, targets):
        print(actions.shape);
        step = self.critic.run_train(states, actions, targets)
        # ac = self.actor.run_predict(states)
        ac, acnoised = self.getActions(states)
        # for i in range(8):
        #     if ac[i][3] < 2.0:
        #         ac[i][3] = 2.0
        #     if ac[i][4] < 100.0:
        #         ac[i][4] = 100.0
        #     if ac[i][5] < 50.0:
        #         ac[i][5] = 50.0
        a_grad = self.critic.run_get_action_gradients(states, acnoised[:,:3])

        print("step: {}".format(step))
        print("action gradients: {}".format(a_grad[0]))
        self.actor.run_train(states, a_grad[0], acnoised[:,:3], step)

    def update_targets(self):
        self.critic.run_update_target_nn()
        self.actor.run_update_target_nn()

    def predict_target_nn(self, state):
        a = self.actor.run_predict_target(state)
        return self.critic.run_predict_target(state, a[:,:3])
