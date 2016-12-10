import os
import sys
import time

import numpy as np
import logging
logging.getLogger("tensorflow").setLevel(logging.INFO)
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.DEBUG)

from driver import Driver

from ddpgPolicy import DDPGPolicy

from ouNoise import OUNoise
from replay_buffer import ReplayBuffer


class DrLBird(Driver):
    # policy = GaussianPolicy()

    def DDPG(self, resume=False, out_dir=None):


        with tf.Session() as sess:
            episode_reward = tf.Variable(0., name="episodeReward")
            tf.scalar_summary("Reward", episode_reward)
            episode_ave_max_q = tf.Variable(0., name='epsideAvgMaxQ')
            tf.scalar_summary("Qmax Value", episode_ave_max_q)
            summary_vars = [episode_reward, episode_ave_max_q]
            summary_ops = tf.merge_all_summaries()

            if not resume:
                timestamp = str(int(time.time()))
                out_dir = os.path.abspath(os.path.join(
                    '/scratch/s7550245/Dr.-L.-Bird', "runsDDPG", timestamp))
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
            print("Summaries will be written to: {}\n".format(out_dir))

            self.global_step = tf.Variable(0, name='global_step',
                                           trainable=False)

            self.policy = DDPGPolicy(sess, out_dir, self.global_step)
            writer = tf.train.SummaryWriter(out_dir, sess.graph)

            self.cno = tf.add_check_numerics_ops()
            sess.run(tf.initialize_all_variables())

            maxEpisodes = 100000
            replayBufferSize = 10000
            miniBatchSize = 32
            gamma = 0.99
            replay = ReplayBuffer(replayBufferSize)
            epsilon = 0.2
            explT = 50

            self.saver = tf.train.Saver()
            if resume:
                explT = 0
                self.saver.restore(sess, tf.train.latest_checkpoint(out_dir))
                replay.load(os.path.join(out_dir, "replayBuffer.pickle"))
                print("Model restored.")

            for e in range(maxEpisodes):
                # noise = OUNoise(3)
                # noise = OUNoise(3, sigma=[1.5, 180.0, 90.0])
                oldScore = 0
                terminal = False
                ep_reward = 0
                ep_ave_max_q = 0
                # self.loadLevel(11)
                self.loadRandLevel()

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
                    # start = time.time()
                    if e < explT or np.random.rand() < epsilon:
                        action = np.array([[np.random.rand() * 50.0,
                                            np.random.rand() * 9000.0,
                                            np.random.rand() * 4000.0]])
                        print("Next action (e-greedy): {}\n".format(action))
                        # time.sleep(1.0)
                    else:
                        action = self.policy.getActions(state)
                        print("Next action: {}\n".format(action))
                        # time.sleep(1.0)
                    # end = time.time()
                    # print("time", end-start)

                    score, terminal, newState = \
                        self.actionResponse(action[0])

                    if score < oldScore:
                        print("INVALID SCORE {} {}".format(score, oldScore))
                        reward = 0
                    else:
                        reward = score - oldScore
                    reward *= 0.0001
                    print(score, oldScore, reward, ep_reward)
                    oldScore = score
                    replay.add(state, action, reward, terminal, newState)

                    if replay.size() > miniBatchSize:
                        s_batch, a_batch, r_batch, t_batch, ns_batch =\
                            replay.sample_batch(miniBatchSize)

                        qValsNewState = self.policy.predict_target_nn(ns_batch)
                        # print("target qs: {}".format(qValsNewState))
                        # print("t_batch: {}".format(t_batch))
                        y_batch = np.zeros((miniBatchSize, 1))
                        for i in range(miniBatchSize):
                            if t_batch[i]:
                                y_batch[i] = r_batch[i]
                            else:
                                y_batch[i] = r_batch[i] + \
                                    gamma * qValsNewState[i]
                        # print("y_batch: {}".format(y_batch))
                        # print("rewards: {}".format(r_batch))
                        ep_ave_max_q += np.amax(qValsNewState)

                        self.policy.update(s_batch, a_batch, y_batch)
                        self.policy.update_targets()
                    else:
                        time.sleep(2.0)

                    ep_reward += reward
                    sys.stdout.flush()

                print("episode reward: {}".format(ep_reward))
                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: ep_reward,
                    summary_vars[1]: ep_ave_max_q / float(step)
                })

                writer.add_summary(summary_str, e)
                writer.flush()

                print('| Reward: {}, | Episode {}, | Qmax: {}'.
                      format(ep_reward, e,
                             ep_ave_max_q / float(step)))

                if (e+1) % 5 == 0:
                    save_path = self.saver.save(sess,
                                                out_dir + "/model.ckpt",
                                                global_step=self.global_step)
                    replay.dump(os.path.join(out_dir, "replayBuffer.pickle"))
                    print("Model saved in file: %s" % save_path)
                sys.stdout.flush()
