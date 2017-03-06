import json
import math
import os
import shutil
import sys
import time
import threading

import numpy as np
import logging
logging.getLogger("tensorflow").setLevel(logging.INFO)
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.DEBUG)

from driver import Driver

from ddpgPolicy import DDPGPolicy

from ouNoise import OUNoise
from replay_buffer import ReplayBuffer
from sumTree import SumTree


class DrLBird(Driver):
    # policy = GaussianPolicy()

    def DDPG(self, params, out_dir):
        self.params = params
        evalu = params['evaluation']
        useVGG = params['useVGG']
        prioritized = params['prioritized']
        self.lock = threading.Lock()

        with tf.Session() as sess:
            episode_reward = tf.Variable(0., name="episodeReward")
            tf.summary.scalar("Reward", episode_reward)
            episode_ave_max_q = tf.Variable(0., name='epsideAvgMaxQ')
            tf.summary.scalar("Qmax_Value", episode_ave_max_q)
            actionDx = tf.Variable(0., name='actionDx')
            tf.summary.scalar("Action_Dx", actionDx)
            actionDy = tf.Variable(0., name='actionDy')
            tf.summary.scalar("Action_Dy", actionDy)
            actionT = tf.Variable(0., name='actionT')
            tf.summary.scalar("Action_T", actionT)
            summary_vars = [episode_reward, episode_ave_max_q,
                            actionDx, actionDy, actionT]
            summary_ops = tf.summary.merge_all()

            if not params['resume']:
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                print("new start...")
                config = json.dumps(params)
                with open(os.path.join(out_dir, "config"), 'w') as f:
                    f.write(config)
            else:
                out_dir = params['resume']
                print("resuming... ", out_dir)

            shutil.copy2('drlbird.py', os.path.join(out_dir, 'drlbird.py'))
            shutil.copy2('tfUtils.py', os.path.join(out_dir, 'tfUtils.py'))
            shutil.copy2('ddpgActor.py', os.path.join(out_dir, 'ddpgActor.py'))
            shutil.copy2('ddpgCritic.py', os.path.join(out_dir, 'ddpgCritic.py'))
            shutil.copy2('ddpgPolicy.py', os.path.join(out_dir, 'ddpgPolicy.py'))
            print("Summaries will be written to: {}\n".format(out_dir))

            self.global_step = tf.Variable(0, name='global_step',
                                           trainable=False)

            self.policy = DDPGPolicy(sess, out_dir,
                                     self.global_step,
                                     params)
            writerTrain = tf.summary.FileWriter(out_dir+"/train", sess.graph)
            writerTest = tf.summary.FileWriter(out_dir+"/test", sess.graph)

            self.cno = tf.add_check_numerics_ops()
            episode_step = tf.Variable(0, name='episode_step',
                                       trainable=False, dtype=tf.int32)
            increment_ep_step_op = tf.assign(episode_step, episode_step+1)
            action_step = tf.Variable(0, name='action_step',
                                      trainable=False, dtype=tf.int32)
            increment_ac_step_op = tf.assign(action_step, action_step+1)
            sess.run(tf.initialize_all_variables())

            replayBufferSize = 10000
            self.maxEpisodes = self.params['numEpochs']
            self.miniBatchSize = self.params['miniBatchSize']
            self.gamma = self.params['gamma']
            self.startLearning = self.params['startLearning']
            if prioritized:
                self.replay = SumTree(replayBufferSize)
                print("using SumTree")
            else:
                self.replay = ReplayBuffer(replayBufferSize)
                print("using linear Buffer")
            epsilon = 0.2
            explT = 50
            cnt = 1

            self.saver = tf.train.Saver()
            if params['resume']:
                explT = 0
                self.saver.restore(sess, tf.train.latest_checkpoint(out_dir))
                if not evalu:
                    self.replay.load(os.path.join(out_dir,
                                                  "replayBuffer.pickle"))
                print("Model restored.")

            if evalu:
                episode_reward = tf.Variable(0., name="episodeRewardEval")
                tf.summary.scalar("RewardEval", episode_reward)
                episode_ave_max_q = tf.Variable(0., name='epsideAvgMaxQEval')
                tf.summary.scalar("Qmax_ValueEval", episode_ave_max_q)

            if self.params['async']:
                t = threading.Thread(target=self.learn)
                t.daemon = True
                t.start()

            fs = sess.run(episode_step)
            acs = sess.run(action_step)
            ac = acs
            for e in range(fs, self.maxEpisodes):
                if self.params['async']:
                    if not t.isAlive():
                        break

                sess.run(increment_ep_step_op)
                epsilon = 1.0 / (math.pow(e+1, 1.0/3.0))
                oldScore = 0
                terminal = False
                ep_reward = 0
                ep_ave_max_q = 0
                if params['loadLevel'] is None:
                    lvl = self.loadRandLevel()
                else:
                    lvl = params['loadLevel']
                    self.loadLevel(lvl)

                self.fillObs()
                state = self.preprocessDataForNN(vgg=useVGG)
                self.findSlingshot()

                step = 0
                while not terminal:
                    self.birdCnt = self.birdCount(lvl) - step
                    if self.getState() != 5:
                        print("does this happen?")
                        break

                    step += 1
                    if ((not evalu) and
                        (e < explT or np.random.rand() < epsilon)):
                        action = np.array([[np.random.rand(),
                                            np.random.rand(),
                                            np.random.rand()]])
                        a_scaled = action * np.array([[50.0, 9000.0, 4000.0]])
                        print("Next action (e-greedy): {}\n".format(a_scaled))
                    else:
                        action = self.policy.getActions(state)
                        a_scaled = action * np.array([[50.0, 9000.0, 4000.0]])
                        print("Next action: {}\n".format(a_scaled))

                    sess.run(increment_ac_step_op)
                    ac += 1
                    action_summary_str = sess.run(action_summary_ops,
                                                  feed_dict={
                        action_summary_vars[0]: a_scaled[0],
                        action_summary_vars[1]: a_scaled[1],
                        action_summary_vars[2]: a_scaled[2],
                                                  })

                    if evalu:
                        writerTest.add_summary(action_summary_str, ac-acs)
                        writerTest.flush()
                    else:
                        writerTrain.add_summary(action_summary_str, ac)
                        writerTrain.flush()

                    score, terminal, newState = \
                        self.actionResponse(a_scaled[0], vgg=useVGG)

                    print("Current score: {}".format(score))
                    if score == -1:
                        reward = 0
                    elif score < oldScore:
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

                    if prioritized:
                        self.replay.add(
                            999.9,
                            (state, action, reward, terminal, newState))
                    else:
                        self.replay.add(
                            state, action, reward, terminal, newState)

                    if not self.params['async']:
                        self.learn()


                    state = newState
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

                # print('| Reward: {}, | Episode {}, | Qmax: {}'.
                #       format(ep_reward, e,
                #              ep_ave_max_q / float(step)))
                print('| Reward: {}, | Episode {}'.
                      format(ep_reward, e))

                if ((not evalu) and ((e+1) % 5)) == 0:
                    save_path = self.saver.save(sess,
                                                out_dir + "/model.ckpt",
                                                global_step=self.global_step)
                    self.replay.dump(os.path.join(out_dir,
                                                  "replayBuffer.pickle"))
                    print("Model saved in file: %s" % save_path)
                sys.stdout.flush()

    def learn(self):
        while True:
            if self.replay.size() < self.startLearning:
                if self.params['async']:
                    continue
                else:
                    return

            if self.params['prioritized']:
                ids, \
                    s_batch, a_batch, r_batch, t_batch, ns_batch = \
                        self.replay.sample_batch(self.miniBatchSize)
                print(ids)
            else:
                s_batch, a_batch, r_batch, t_batch, ns_batch = \
                    self.replay.sample_batch(self.miniBatchSize)

            qValsNewState = self.policy.predict_target_nn(ns_batch)
            y_batch = np.zeros((self.miniBatchSize, 1))
            for i in range(self.miniBatchSize):
                if t_batch[i]:
                    y_batch[i] = r_batch[i]
                else:
                    y_batch[i] = r_batch[i] + \
                        self.gamma * qValsNewState[i]

            if self.params['prioritized']:
                for i in range(self.miniBatchSize):
                    self.replay.update(ids[i], abs(y_batch[i]))

            qs = self.policy.update(s_batch, a_batch, y_batch)
            # ep_ave_max_q += np.amax(qs)

            self.policy.update_targets()

            if not self.params['async']:
                return
