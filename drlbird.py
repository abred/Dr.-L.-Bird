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
from tensorflow.python.framework import ops
tf.logging.set_verbosity(tf.logging.DEBUG)

from driver import Driver

from ddpgPolicy import DDPGPolicy

# from ouNoise import OUNoise
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
        self.annealSteps = float(2000)
        self.cnt=0

        with tf.Session() as sess:
            episode_reward = tf.Variable(0., name="episodeReward")
            sumRew = tf.summary.scalar("Reward", episode_reward)
            episode_ave_max_q = tf.Variable(0., name='epsideAvgMaxQ')
            sumQ = tf.summary.scalar("Qmax_Value", episode_ave_max_q)
            actionDx = tf.Variable(0., name='actionDx')
            sumDx = tf.summary.scalar("Action_Dx", actionDx)
            actionDy = tf.Variable(0., name='actionDy')
            sumDy = tf.summary.scalar("Action_Dy", actionDy)
            actionT = tf.Variable(0., name='actionT')
            sumT = tf.summary.scalar("Action_T", actionT)
            summary_vars = [episode_reward, episode_ave_max_q]
            summary_ops = tf.summary.merge([sumRew, sumQ])
            action_summary_vars = [actionDx, actionDy, actionT]
            action_summary_ops = tf.summary.merge([sumDx, sumDy, sumT])

            if not params['resume']:
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                print("new start... {}".format(out_dir))
                config = json.dumps(params)
                with open(os.path.join(out_dir, "config"), 'w') as f:
                    f.write(config)
            else:
                out_dir = params['resume']
                print("resuming... ", out_dir)

            if os.environ['SLURM_JOB_NAME'] != 'zsh':
                sys.stdout.flush()
                sys.stdout = open(os.path.join(out_dir, "log"), 'w')
            print("slurm jobid: {}".format(os.environ['SLURM_JOBID']))

            shutil.copy2('drlbird.py', os.path.join(out_dir, 'drlbird.py'))
            shutil.copy2('tfUtils.py', os.path.join(out_dir, 'tfUtils.py'))
            shutil.copy2('ddpgActor.py', os.path.join(out_dir, 'ddpgActor.py'))
            shutil.copy2('ddpgCritic.py',
                         os.path.join(out_dir, 'ddpgCritic.py'))
            shutil.copy2('ddpgPolicy.py',
                         os.path.join(out_dir, 'ddpgPolicy.py'))
            shutil.copy2("replay_buffer.py",
                         os.path.join(out_dir, "replay_buffer.py"))
            shutil.copy2("sumTree.py",os.path.join(out_dir, "sumTree.py"))

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
            action_step_sum = tf.Variable(0, name='action_step_sum',
                                      trainable=False, dtype=tf.int32)
            increment_ac_sum_step_op = tf.assign(action_step_sum,
                                                 action_step_sum+1)
            action_step_eps = tf.Variable(0, name='action_step_eps',
                                      trainable=False, dtype=tf.int32)
            increment_ac_eps_step_op = tf.assign(action_step_eps,
                                             action_step_eps+1)
            sess.run(tf.initialize_all_variables())

            replayBufferSize = 10000
            self.maxEpisodes = self.params['numEpochs']
            self.miniBatchSize = self.params['miniBatchSize']
            self.gamma = self.params['gamma']
            self.startLearning = self.params['startLearning']
            if prioritized:
                self.replay = SumTree(replayBufferSize,
                                  self.params['miniBatchSize'],
                                  self.annealSteps*20)
                print("using SumTree")
            else:
                self.replay = ReplayBuffer(replayBufferSize)
                print("using linear Buffer")
            self.startEpsilon = 1.0
            self.endEpsilon = 0.1
            self.epsilon = self.startEpsilon
            explT = 50
            cnt = 1

            if self.params['useVGG'] and not self.params['resume']:
                self.policy.vggsaver.restore(
                    sess,
                    '/home/s7550245/convNet/vgg-model')
                print("VGG restored.")

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
            acS = sess.run(action_step_sum)
            acE = sess.run(action_step_eps)
            acs = acS
            ace = acE
            for e in range(fs, self.maxEpisodes):
                if self.params['async']:
                    if not t.isAlive():
                        break

                sess.run(increment_ep_step_op)
                # epsilon = 1.0 / (math.pow(e+1, 1.0/3.0))
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
                    sess.run(increment_ac_eps_step_op)
                    ace += 1
                    tmp_step = min(ace, self.annealSteps)
                    self.epsilon = (self.startEpsilon - self.endEpsilon) * \
                                   (1 - tmp_step / self.annealSteps) + \
                                   self.endEpsilon
                    if ((not evalu) and
                        (e < explT or np.random.rand() < self.epsilon)):
                        action = np.array([[np.random.rand(),
                                            np.random.rand(),
                                            np.random.rand()]])
                        # a_scaled = action * np.array([[-50.0, 50.0, 4000.0]])

                        # a_scaled = action * np.array([[70.0, 70.0, 4000.0]])
                        # a_scaled = a_scaled + np.array([[10.0, 10.0, 0.0]])

                        # a_scaled = action * np.array([[160.0, 90.0, 4000.0]])
                        # a_scaled = a_scaled + np.array([[-80.0, 0.0, 0.0]])
                        a_scaled = action * np.array([[-100.0, 100.0, 6000.0]])
                        print("Step: {} Next action (e-greedy {}): {}".format(
                            ace,
                            self.epsilon,
                            a_scaled))
                    else:
                        action = self.policy.getActions(state)
                        # a_scaled = action

                        # a_scaled = action * np.array([[70.0, 70.0, 4000.0]])
                        # a_scaled = a_scaled + np.array([[10.0, 10.0, 0.0]])

                        # a_scaled = action * np.array([[160.0, 160.0,4000.0]])
                        # a_scaled = a_scaled + np.array([[-80.0, -80.0, 0.0]])
                        # a_scaled = action * np.array([[-100.0, 100.0, 6000.0]])
                        a_scaled = action
                        print("Step: {} Next action: {}".format(ace, a_scaled))

                        sess.run(increment_ac_sum_step_op)
                        acs += 1
                        action_summary_str = sess.run(action_summary_ops,
                                                      feed_dict={
                            action_summary_vars[0]: a_scaled[0][0],
                            action_summary_vars[1]: a_scaled[0][1],
                            action_summary_vars[2]: a_scaled[0][2],
                                                      })

                        if evalu:
                            writerTest.add_summary(action_summary_str, acs-acS)
                            writerTest.flush()
                        else:
                            writerTrain.add_summary(action_summary_str, acs)
                            writerTrain.flush()

                    score, terminal, newState = \
                        self.actionResponse(a_scaled[0], vgg=useVGG)

                    print("Time: {} Current score: {}".format(time.ctime(),
                                                              score))
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

                    self.insertSamples(state, action, reward, terminal,
                                       newState)

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
                print('Time: {} | Reward: {}, | Episode {}'.
                      format(time.ctime(), ep_reward, e))

                if ((not evalu) and ((e+1) % 5)) == 0:
                    print("Saving model... (Time: {})".format(time.ctime()))
                    save_path = self.saver.save(sess,
                                                out_dir + "/model.ckpt",
                                                global_step=self.global_step)
                    self.replay.dump(os.path.join(out_dir,
                                                  "replayBuffer.pickle"))
                    print("Model saved in file: {} (Time: {}), dumping buffer ...".format(save_path, time.ctime()))
                sys.stdout.flush()

    def learn(self):
        while True:
            if self.replay.size() < self.startLearning or \
               self.replay.size() < self.miniBatchSize:
                if self.params['async']:
                    continue
                else:
                    return

            self.lock.acquire()
            if self.params['prioritized']:
                ids, w_batch, s_batch, a_batch, r_batch, t_batch, ns_batch = \
                        self.replay.sample_batch(self.miniBatchSize)
                # print(ids)
            else:
                s_batch, a_batch, r_batch, t_batch, ns_batch = \
                    self.replay.sample_batch(self.miniBatchSize)
            self.lock.release()

            qValsNewState = self.policy.predict_target_nn(ns_batch)
            y_batch = np.zeros((self.miniBatchSize, 1))
            if self.params['importanceSampling']:
                wMax = np.max(w_batch)
                for i in range(self.miniBatchSize):
                    if t_batch[i]:
                        y_batch[i] = w_batch[i] / wMax * r_batch[i]
                    else:
                        y_batch[i] = w_batch[i] / wMax * \
                            (r_batch[i] + self.gamma * qValsNewState[i])
            else:
                for i in range(self.miniBatchSize):
                    if t_batch[i]:
                        y_batch[i] = r_batch[i]
                    else:
                        y_batch[i] = r_batch[i] + \
                            self.gamma * qValsNewState[i]

            # print(r_batch, qValsNewState, y_batch)
            if self.params['prioritized']:
                for i in range(self.miniBatchSize):
                    self.replay.update(ids[i], abs(y_batch[i]))

            gs, qs, delta = self.policy.update(s_batch, a_batch, y_batch)
            if self.params['prioritized']:
                self.lock.acquire()
                for i in range(self.miniBatchSize):
                    self.replay.update(ids[i], abs(delta[i]))
                self.lock.release()
            # ep_ave_max_q += np.amax(qs)

            self.policy.update_targets()

            if not self.params['async']:
                return

    def insertSamples(self, state, action, reward, terminal, newState):
        # print(state.shape)
        # print(newState.shape)
        # state.shape = (state.shape[1],
        #                state.shape[2],
        #                state.shape[3])
        # newState.shape = (newState.shape[1],
        #                   newState.shape[2],
        #                   newState.shape[3])

        if self.params['prioritized']:
            self.replay.add(None, (state, action, reward, terminal, newState))
        else:
            self.replay.add(state, action, reward, terminal, newState)
