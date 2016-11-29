import numpy as np

from ddpgActor import Actor

from ddpgCritic import Critic


class DDPGPolicy:
    def __init__(self, sess, out_dir, glStep):
        self.actor = Actor(sess, out_dir)
        self.critic = Critic(sess, out_dir, glStep)
        self.cnt = 0

    def getActions(self, state):
        a = self.actor.run_predict(state)
        return a

    def update(self, states, actions, targets):
        # print("states: {}".format(states))
        np.set_printoptions(threshold=np.nan)
        self.cnt += 1
        with open(str(self.cnt) +"h3w.log", "a") as myfile:
            tmp = self.critic.w.eval()
            myfile.write("{}".format(tmp))
            np.save(str(self.cnt) +"h3w.npy", tmp)

        with open(str(self.cnt) +"grad.log", "a") as myfile:
            grad = self.critic.run_get_gradients(states, actions, targets)
            for v,g in grad:
                if "Critic/h3" in v.name:
                    n = v.name.replace("/", "_")
                    myfile.write("{}, {}".format(v.name, g[1]))
                    np.save(str(self.cnt) + n + "grad.npy", tmp)

        step = self.critic.run_train(states, actions, targets)

        with open(str(self.cnt) +"h3wafter.log", "a") as myfile:
            tmp = self.critic.w.eval()
            myfile.write("{}".format(tmp))
            np.save(str(self.cnt) +"h3wafter.npy", tmp)

        with open(str(self.cnt) +"gradafter.log", "a") as myfile:
            grad = self.critic.run_get_gradients(states, actions, targets)
            for v,g in grad:
                if "Critic/h3" in v.name:
                    n = v.name.replace("/", "_")
                    myfile.write("{}, {}".format(v.name, g[1]))
                    np.save(str(self.cnt) + n + "gradafter.npy", tmp)

        # print("predict critic: {}".format(self.critic.run_predict(states, actions)))
        # myfile.write("STEP: {}".format(step))
        # print("actions:", actions)
        # print("targets", targets)
        ac = self.actor.run_predict(states)
        # print("predict actions: {}".format(ac))
        # print("state shape {}".format(states.shape))
        middle, q, a_grad = self.critic.run_get_action_gradients(states, ac)
        # print("middle: {}".format(middle))
        # print("critic q: {}".format(q))
        # print("action gradients: {}".format(a_grad))
        # print("{} {} {} {} {}".format(a_grad[0][0] == a_grad[0][1],
        #                               a_grad[0][2] == a_grad[0][3],
        #                               a_grad[0][4] == a_grad[0][6],
        #                               a_grad[0][0] - a_grad[0][1],
        #                               a_grad[0][2] == a_grad[0][3]))
        self.actor.run_train(states, a_grad[0], step)

    def update_targets(self):
        self.critic.run_update_target_nn()
        self.actor.run_update_target_nn()

    def predict_target_nn(self, state):
        a = self.actor.run_predict_target(state)
        return self.critic.run_predict_target(state, a)
