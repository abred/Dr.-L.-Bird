import math
import os
import shutil
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

    def DDPG(self, resume=False, out_dir=None, evalu=False,
             useVGG=False, top=None):
        with tf.Session() as sess:
            # if evalu:
            #     episode_reward = tf.Variable(0., name="episodeRewardEval")
            #     tf.scalar_summary("RewardEval", episode_reward)
            #     episode_ave_max_q = tf.Variable(0., name='epsideAvgMaxQEval')
            #     tf.scalar_summary("Qmax ValueEval", episode_ave_max_q)
            # else:
            episode_reward = tf.Variable(0., name="episodeReward")
            tf.summary.scalar("Reward", episode_reward)
            episode_ave_max_q = tf.Variable(0., name='epsideAvgMaxQ')
            tf.summary.scalar("Qmax_Value", episode_ave_max_q)
            summary_vars = [episode_reward, episode_ave_max_q]
            summary_ops = tf.summary.merge_all()

            if not resume:
                timestamp = str(int(time.time()))
                out_dir = os.path.abspath(os.path.join(
                    '/scratch/s7550245/Dr.-L.-Bird', "runsDDPG", timestamp))
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
            shutil.copy2('drlbird.py', os.path.join(out_dir, 'drlbird.py'))
            shutil.copy2('tfUtils.py', os.path.join(out_dir, 'tfUtils.py'))
            shutil.copy2('ddpgActor.py', os.path.join(out_dir, 'ddpgActor.py'))
            shutil.copy2('ddpgCritic.py', os.path.join(out_dir, 'ddpgCritic.py'))
            shutil.copy2('ddpgPolicy.py', os.path.join(out_dir, 'ddpgPolicy.py'))
            print("Summaries will be written to: {}\n".format(out_dir))

            self.global_step = tf.Variable(0, name='global_step',
                                           trainable=False)

            self.policy = DDPGPolicy(sess, out_dir,
                                     self.global_step, useVGG=useVGG, top=top)
            writerTrain = tf.summary.FileWriter(out_dir+"/train", sess.graph)
            writerTest = tf.summary.FileWriter(out_dir+"/test", sess.graph)

            self.cno = tf.add_check_numerics_ops()
            self.episode_step = tf.Variable(0, name='episode_step',
                                            trainable=False, dtype=tf.int32)
            self.increment_ep_step_op = tf.assign(self.episode_step,
                                                  self.episode_step+1)
            sess.run(tf.initialize_all_variables())

            maxEpisodes = 100000
            replayBufferSize = 10000
            miniBatchSize = 16
            gamma = 0.99
            replay = ReplayBuffer(replayBufferSize)
            epsilon = 0.2
            explT = 25
            cnt = 1

            self.saver = tf.train.Saver()
            if resume:
                explT = 0
                self.saver.restore(sess, tf.train.latest_checkpoint(out_dir))
                if not evalu:
                    replay.load(os.path.join(out_dir, "replayBuffer.pickle"))
                print("Model restored.")

            if evalu:
                episode_reward = tf.Variable(0., name="episodeRewardEval")
                tf.summary.scalar("RewardEval", episode_reward)
                episode_ave_max_q = tf.Variable(0., name='epsideAvgMaxQEval')
                tf.summary.scalar("Qmax_ValueEval", episode_ave_max_q)

            fs = sess.run(self.episode_step)
            for e in range(fs, maxEpisodes):
                sess.run(self.increment_ep_step_op)
                # noise = OUNoise(3)
                # noise = OUNoise(3, sigma=[1.5, 180.0, 90.0])
                epsilon = 1.0 / (math.pow(e+1, 1.0/3.0))
                oldScore = 0
                terminal = False
                ep_reward = 0
                ep_ave_max_q = 0
                self.loadLevel(11)
                # self.loadRandLevel()

                step = 0
                while not terminal:
                    self.birdCnt = self.birdCount()
                    # time.sleep(.0)
                    self.findSlingshot()
                    if self.currCenterX == 0:
                        if self.getState() != 5:
                            break
                        else:
                            continue
                    # if step == 0:
                    #     self.fillObs(store=cnt)
                    #     cnt += 1
                    # else:
                    self.fillObs()
                    state = self.preprocessDataForNN(vgg=useVGG)
                    # start = time.time()
                    step += 1
                    if ((not evalu) and
                        (e < explT or np.random.rand() < epsilon)):
                        action = np.array([[np.random.rand(),
                                            np.random.rand(),
                                            np.random.rand()]])
                        a_scaled = action * np.array([[50.0, 9000.0, 4000.0]])
                        print("Next action (e-greedy): {}\n".format(a_scaled))
                        # time.sleep(1.0)
                    else:
                        action = self.policy.getActions(state)
                        a_scaled = action * np.array([[50.0, 9000.0, 4000.0]])
                        print("Next action: {}\n".format(a_scaled))
                        # time.sleep(1.0)
                    # end = time.time()
                    # print("time", end-start)

                    score, terminal, newState = \
                        self.actionResponse(a_scaled[0], vgg=useVGG)

                    print("Current score: {}".format(score))
                    if score < oldScore:
                        print("INVALID SCORE {} {}".format(score, oldScore))
                        reward = 0
                    else:
                        reward = score - oldScore
                    reward *= 0.0001
                    # print(score, oldScore, reward, ep_reward)
                    oldScore = score
                    ep_reward += reward

                    if evalu:
                        continue

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

                        qs = self.policy.update(s_batch, a_batch, y_batch)
                        ep_ave_max_q += np.amax(qs)

                        self.policy.update_targets()
                    else:
                        time.sleep(2.0)

                    sys.stdout.flush()

                # print("episode reward: {}".format(ep_reward))
                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: ep_reward,
                    summary_vars[1]: ep_ave_max_q / float(step)
                })

                if evalu:
                    writerTest.add_summary(summary_str, e-fs)
                    writerTest.flush()
                else:
                    writerTrain.add_summary(summary_str, e)
                    writerTrain.flush()

                print('| Reward: {}, | Episode {}, | Qmax: {}'.
                      format(ep_reward, e,
                             ep_ave_max_q / float(step)))

                if ((not evalu) and ((e+1) % 5)) == 0:
                    save_path = self.saver.save(sess,
                                                out_dir + "/model.ckpt",
                                                global_step=self.global_step)
                    replay.dump(os.path.join(out_dir, "replayBuffer.pickle"))
                    print("Model saved in file: %s" % save_path)
                sys.stdout.flush()
