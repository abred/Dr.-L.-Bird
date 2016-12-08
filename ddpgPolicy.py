import numpy as np

from ddpgActor import Actor

from ddpgCritic import Critic

import tensorflow as tf

class DDPGPolicy:
    def __init__(self, sess, out_dir, glStep):
        self.actor = Actor(sess, out_dir)
        self.critic = Critic(sess, out_dir, glStep)
        self.cnt = 0
        self.out_dir = out_dir

    def getActions(self, state):
        a = self.actor.run_predict(state)
        return a

    def update(self, states, actions, targets):
        # print("states: {}".format(states))
        # np.set_printoptions(threshold=np.nan)
        # self.cnt += 1
        # bnvars = tf.get_collection('batchnorm')
        # for v in bnvars:
        #     n = v.name.replace("/", "_")
        #     with open(self.out_dir + "/" + n + str(self.cnt) +
        #               "bn.log", "a") as myfile:
        #         tmp = v.eval()
        #         myfile.write("{}".format(tmp))
        #         np.save(self.out_dir + "/" + n + str(self.cnt) +"bn.npy", tmp)

        # with open(self.out_dir + "/" + str(self.cnt) +
        #           "h3w.log", "a") as myfile:
        #     tmp = self.critic.w.eval()
        #     myfile.write("{}".format(tmp))
        #     np.save(self.out_dir + "/" + str(self.cnt) +"h3w.npy", tmp)
        # with open(self.out_dir + "/" + str(self.cnt) +
        #           "h3b.log", "a") as myfile:
        #     tmp = self.critic.b.eval()
        #     myfile.write("{}".format(tmp))
        #     np.save(self.out_dir + "/" + str(self.cnt) +"h3b.npy", tmp)

        # with open(self.out_dir + "/" + str(self.cnt) +
        #           "grad.log", "a") as myfile:
        #     grad = self.critic.run_get_gradients(states, actions, targets)
        #     for v,g in grad:
        #         if "Critic/h3" in v.name:
        #             n = v.name.replace("/", "_")
        #             myfile.write("{}, {}".format(v.name, g[1]))
        #             np.save(self.out_dir + "/" + str(self.cnt) +
        #                     n + "grad.npy", tmp)
        # with open(self.out_dir + "/" + str(self.cnt) +
        #           "h1.log", "a") as myfile:
        #     tmp = self.critic.h1.eval(feed_dict={
        #         self.critic.input_pl: states,
        #         self.critic.isTraining: False})
        #     myfile.write("{}".format(tmp))
        #     np.save(self.out_dir + "/" + str(self.cnt) +"h1.npy", tmp)
        # with open(self.out_dir + "/" + str(self.cnt) +
        #           "h2.log", "a") as myfile:
        #     tmp = self.critic.h2.eval(feed_dict={
        #         self.critic.input_pl: states,
        #         self.critic.actions_pl: actions,
        #         self.critic.isTraining: True})
        #     myfile.write("{}".format(tmp))
        #     np.save(self.out_dir + "/" + str(self.cnt) +"h2.npy", tmp)
        # with open(self.out_dir + "/" + str(self.cnt) +
        #           "h3.log", "a") as myfile:
        #     tmp = self.critic.h3.eval(feed_dict={
        #         self.critic.input_pl: states,
        #         self.critic.actions_pl: actions,
        #         self.critic.isTraining: True})
        #     myfile.write("{}".format(tmp))
        #     np.save(self.out_dir + "/" + str(self.cnt) +"h3.npy", tmp)

        step = self.critic.run_train(states, actions, targets)

        # with open(self.out_dir + "/" + str(self.cnt) +
        #           "h3wafter.log", "a") as myfile:
        #     tmp = self.critic.w.eval()
        #     myfile.write("{}".format(tmp))
        #     np.save(self.out_dir + "/" + str(self.cnt) +"h3wafter.npy", tmp)

        # with open(self.out_dir + "/" + str(self.cnt) +
        #           "gradafter.log", "a") as myfile:
        #     grad = self.critic.run_get_gradients(states, actions, targets)
        #     for v,g in grad:
        #         if "Critic/h3" in v.name:
        #             n = v.name.replace("/", "_")
        #             myfile.write("{}, {}".format(v.name, g[1]))
        #             np.save(self.out_dir + "/" + str(self.cnt) +
        #                     n + "gradafter.npy", tmp)

        # print("predict critic: {}".format(self.critic.run_predict(states, actions)))
        # myfile.write("STEP: {}".format(step))
        print("step: {}".format(step))
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
