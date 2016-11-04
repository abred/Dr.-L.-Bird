import os
import time

import numpy as np
import tensorflow as tf

from driver import Driver

from dspgPolicy import DSPGPolicy

from ouNoise import OUNoise
from replay_buffer import ReplayBuffer


class DrLBird(Driver):

    def DSPG(self, resume=False):
        with tf.Session() as sess:
            episode_reward = tf.Variable(0., name="episodeReward")
            tf.scalar_summary("Reward", episode_reward)
            episode_ave_max_q = tf.Variable(0., name='epsideAvgMaxQ')
            tf.scalar_summary("Qmax Value", episode_ave_max_q)
            summary_vars = [episode_reward, episode_ave_max_q]
            summary_ops = tf.merge_all_summaries()

            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir,
                                                   "runsDSPG", timestamp))
            print("Summaries will be written to: {}\n".format(out_dir))

            self.global_step = tf.Variable(0, name='global_step',
                                           trainable=False)

            self.policy = DSPGPolicy(sess, out_dir, self.global_step)
            writer = tf.train.SummaryWriter(out_dir, sess.graph)

            sess.run(tf.initialize_all_variables())

            maxEpisodes = 1000
            replayBufferSize = 1000
            miniBatchSize = 16
            gamma = 0.99
            replay = ReplayBuffer(replayBufferSize)

            self.saver = tf.train.Saver()
            if resume:
                self.saver.restore(sess, tf.train.latest_checkpoint(out_dir))
                print("Model restored.")

            for e in range(maxEpisodes):
                oldScore = 0
                terminal = False
                ep_reward = 0
                ep_ave_max_q = 0
                # self.loadRandLevel()
                self.loadLevel(1)

                step = 0
                while not terminal:
                    self.birdCnt = self.birdCount()
                    self.fillObs()
                    self.findSlingshot()
                    if self.currCenterX == 0:
                        if self.getState() != 5:
                            break
                        else:
                            continue
                    step += 1
                    state = self.preprocessDataForNN()
                    action = self.policy.getActions(state)
                    score, terminal, newState = self.actionResponse(action[0])
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
                            if t_batch[i]:
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

                if (e+1) % 50 == 0:
                    save_path = self.saver.save(sess,
                                                out_dir + "/model.ckpt",
                                                global_step=self.global_step)
                    print("Model saved in file: %s" % save_path)
