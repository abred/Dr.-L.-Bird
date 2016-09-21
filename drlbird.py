import os
import time

import numpy as np
import tensorflow as tf

from driver import Driver

from gaussianPolicy import GaussianPolicy
from ddpgPolicy import DDPGPolicy

from ouNoise import OUNoise
from replay_buffer import ReplayBuffer


class DrLBird(Driver):
    # policy = GaussianPolicy()

    def DDPG(self):
        with tf.Session() as sess:
            episode_reward = tf.Variable(0., name="episodeReward")
            tf.scalar_summary("Reward", episode_reward)
            episode_ave_max_q = tf.Variable(0., name='epsideAvgMaxQ')
            tf.scalar_summary("Qmax Value", episode_ave_max_q)
            summary_vars = [episode_reward, episode_ave_max_q]
            summary_ops = tf.merge_all_summaries()

            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir,
                                                   "runs", timestamp))
            print("Summaries will be written to: {}\n".format(out_dir))

            self.policy = DDPGPolicy(sess, out_dir)
            writer = tf.train.SummaryWriter(out_dir, sess.graph)

            sess.run(tf.initialize_all_variables())

            maxEpisodes = 1000
            replayBufferSize = 1000
            miniBatchSize = 8
            gamma = 0.99
            replay = ReplayBuffer(replayBufferSize)

            for e in range(maxEpisodes):
                # noise = OUNoise(3)
                noise = OUNoise(3, sigma=[1.5, 180.0, 90.0])
                oldScore = 0
                terminal = False
                ep_reward = 0
                ep_ave_max_q = 0
                self.loadRandLevel()

                step = 0
                while not terminal:
                    step += 1
                    self.fillObs()
                    state = self.preprocessDataForNN()
                    action, a_scaled = self.policy.getAction(state,
                                                             noise.noise())
                    score, terminal, newState = self.actionResponse(a_scaled)
                    reward = score - oldScore
                    reward *= 0.0001
                    oldScore = score
                    replay.add(state, action, reward, terminal, newState)

                    if replay.size() > miniBatchSize:
                        s_batch, a_batch, r_batch, t_batch, ns_batch =\
                            replay.sample_batch(miniBatchSize)

                        qValsNewState = self.policy.predict_target_nn(ns_batch)
                        y_batch = np.zeros((miniBatchSize, 1))
                        for i in range(miniBatchSize):
                            if t_batch[i] == terminal:
                                y_batch[i] = r_batch[i]
                            else:
                                y_batch[i] = r_batch[i] + \
                                    gamma * qValsNewState[i]

                        ep_ave_max_q += np.amax(qValsNewState)

                        self.policy.update(s_batch, a_batch, y_batch)
                        self.policy.update_targets()

                    ep_reward += reward

                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: ep_reward,
                    summary_vars[1]: ep_ave_max_q / float(step)
                })

                writer.add_summary(summary_str, e)
                writer.flush()

                print('| Reward: {}, | Episode {}, | Qmax: {}'.
                      format(ep_reward, e,
                             ep_ave_max_q / float(step)))

    def REINFORCE(self):
        self.policy = GaussianPolicy()

        while True:
            self.loadRandLevel()

            scores = []
            states = []
            actions = []
            os = []
            st = self.getStatePrint()
            numAct = 0
            while True:
                self.fillObs()
                states.append(self.preprocessDataForNN())
                o = self.policy.forward(states[-1])
                os.append(o)

                print("nn output: {}".format(o))
                actions.append(self.policy.getAction())
                self.act(actions[-1])
                numAct += 1
                time.sleep(3)
                tempScore = self.getCurrScore()
                time.sleep(11)
                st = self.getStatePrint()

                if st == 5:  # playing
                    scores.append(self.getCurrScore())
                elif st == 6:  # won
                    scores.append(self.getEndScore())
                    break
                elif st == 7:  # lost
                    scores.append(scores[-1]-10000)
                    break
                else:
                    raise Exception("should not get here")

            print("Scores: {}".format(scores))
            rewards = []
            returns = []
            rewards.append(scores[0])
            returns.append(scores[numAct-1])
            for b in range(1, numAct):
                rewards.append(scores[b] - scores[b-1])
                returns.append(returns[b-1] - rewards[b-1])

                print("Rewards: {}".format(rewards))
                print("Returns: {}".format(returns))

            self.policy.backward(np.vstack(states),
                                 np.vstack(actions),
                                 np.vstack(returns))
