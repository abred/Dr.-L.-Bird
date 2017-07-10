import socket
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

def printT(s):
    sys.stdout.write(s + '\n')

# def trace(frame, event, arg):
#     print ("{}, {}:{}".format(event, frame.f_code.co_filename, frame.f_lineno))
#     return trace


class DrLBird:
    def __init__(self, params):
        # sys.settrace(trace)
        self.params = params
        self.socs = []
        port = self.params['port']
        if "taurusi" in self.params['host']:
            hosts = self.params['host'].split(',')
        else:
            hostsSlurm = self.params['host'].split(',')
            hosts = []
            for s in range(len(hostsSlurm)):
                with open('/scratch/s7550245/Dr.-L.-Bird/host-'+
                          hostsSlurm[s], 'r') as hFile:
                    hosts.append(hFile.read().replace('\n', ''))
                print(hostsSlurm[s], hosts[s], port+s)
        for s in range(len(hosts)):
            soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            printT("{}, {}".format(hosts[s], port+s))
            soc.connect((hosts[s], port+s))
            printT("test1")
            d = Driver(soc, s)
            d.configure(421337, True)
            printT("test2")
            d.getStatePrint()
            printT("test3")
            self.socs.append(d)

    def DDPG(self, out_dir):
        self.evalu = self.params['evaluation']
        self.useVGG = self.params['useVGG']
        self.prioritized = self.params['prioritized']
        self.annealSteps = self.params['annealSteps']
        self.lock = threading.Lock()

        self.sess = tf.Session()

        episode_rewardEval = tf.Variable(0., name="episodeRewardEval")
        erSumEval = tf.summary.scalar("RewardEval", episode_rewardEval)
        episode_ave_max_qEval = tf.Variable(0., name='epsideAvgMaxQEval')
        eqSumEval = tf.summary.scalar("Qmax_ValueEval", episode_ave_max_qEval)
        actionDxEval = tf.Variable(0., name='actionDxEval')
        aDxSumEval = tf.summary.scalar("Action_DxEval", actionDxEval)
        actionDyEval = tf.Variable(0., name='actionDyEval')
        aDySumEval = tf.summary.scalar("Action_DyEval", actionDyEval)
        actionTEval = tf.Variable(0., name='actionTEval')
        aTSumEval = tf.summary.scalar("Action_TEval", actionTEval)
        self.summary_varsEval = [episode_rewardEval, episode_ave_max_qEval]
        self.summary_opsEval = tf.summary.merge([erSumEval, eqSumEval])
        self.action_summary_varsEval = [actionDxEval,
                                        actionDyEval,
                                        actionTEval]
        self.action_summary_opsEval = tf.summary.merge([aDxSumEval,
                                                        aDySumEval,
                                                        aTSumEval])

        episode_reward = tf.Variable(0., name="episodeReward")
        erSum = tf.summary.scalar("Reward", episode_reward)
        episode_ave_max_q = tf.Variable(0., name='epsideAvgMaxQ')
        eqSum = tf.summary.scalar("Qmax_Value", episode_ave_max_q)
        actionDx = tf.Variable(0., name='actionDx')
        aDxSum = tf.summary.scalar("Action_Dx", actionDx)
        actionDy = tf.Variable(0., name='actionDy')
        aDySum = tf.summary.scalar("Action_Dy", actionDy)
        actionT = tf.Variable(0., name='actionT')
        aTSum = tf.summary.scalar("Action_T", actionT)
        self.summary_vars = [episode_reward, episode_ave_max_q]
        self.summary_ops = tf.summary.merge([erSum, eqSum])
        self.action_summary_vars = [actionDx, actionDy, actionT]
        self.action_summary_ops = tf.summary.merge([aDxSum, aDySum, aTSum])


        if not self.params['resume']:
            self.out_dir = out_dir
            if not os.path.exists(self.out_dir):
                os.makedirs(self.out_dir)
            print("new start...")
            config = json.dumps(self.params)
            with open(os.path.join(self.out_dir, "config"), 'w') as f:
                f.write(config)
        else:
            self.out_dir = self.params['resume']
            print("resuming... ", self.out_dir)
        sys.stdout.flush()
        if os.environ['SLURM_JOB_NAME'] != 'zsh':
            sys.stdout.flush()
            sys.stdout = open(os.path.join(self.out_dir, "log"), 'w')
        print("slurm jobid: {}".format(os.environ['SLURM_JOBID']))
        sys.stdout.flush()
        shutil.copy2('drlbird.py',
                     os.path.join(self.out_dir, 'drlbird.py'))
        shutil.copy2('tfUtils.py',
                     os.path.join(self.out_dir, 'tfUtils.py'))
        shutil.copy2('ddpgActor.py',
                     os.path.join(self.out_dir, 'ddpgActor.py'))
        shutil.copy2('ddpgCritic.py',
                     os.path.join(self.out_dir, 'ddpgCritic.py'))
        shutil.copy2('ddpgPolicy.py',
                     os.path.join(self.out_dir, 'ddpgPolicy.py'))
        print("Summaries will be written to: {}\n".format(self.out_dir))

        self.global_step = tf.Variable(0, name='global_step',
                                       trainable=False)

        self.policy = DDPGPolicy(self.sess, self.out_dir,
                                 self.global_step,
                                 self.params)
        self.writerTrain = tf.summary.FileWriter(self.out_dir+"/train",
                                                 self.sess.graph)
        self.writerTest = tf.summary.FileWriter(self.out_dir+"/test",
                                                self.sess.graph)

        self.cno = tf.add_check_numerics_ops()
        self.episode_step = tf.Variable(0, name='episode_step',
                                        trainable=False, dtype=tf.int32)
        # increment_ep_step_op = tf.assign(self.episode_step,
        #                                       self.episode_step+1)
        self.action_step = tf.Variable(0, name='action_step',
                                       trainable=False, dtype=tf.int32)
        self.action_step_sum = tf.Variable(0, name='action_step',
                                           trainable=False, dtype=tf.int32)

        # increment_ac_step_op = tf.assign(self.action_step,
        #                                       self.action_step+1)
        self.sess.run(tf.initialize_all_variables())

        replayBufferSize = 10000
        self.maxEpisodes = self.params['numEpochs']
        self.miniBatchSize = self.params['miniBatchSize']
        self.gamma = self.params['gamma']
        self.startLearning = self.params['startLearning']
        if os.environ['SLURM_JOB_NAME'] == 'zsh' and not self.params['sleep']:
            self.startLearning = 1
        if self.prioritized:
            self.replay = SumTree(replayBufferSize,
                                  self.params['miniBatchSize'],
                                  self.annealSteps*20)
            print("using SumTree")
        else:
            self.replay = ReplayBuffer(replayBufferSize)
            print("using linear Buffer")
        self.startEpsilon = 1.0
        self.endEpsilon = 0.1
        self.currEpsilon = self.startEpsilon
        self.epsilon = 0.2
        self.explT = 50
        cnt = 1

        self.saver = tf.train.Saver()
        if self.params['resume']:
            self.explT = 0
            self.saver.restore(self.sess,
                               tf.train.latest_checkpoint(self.out_dir))
            if not self.evalu:
                self.replay.load(os.path.join(self.out_dir,
                                              "replayBuffer.pickle"))
            print("Model restored.")

        if self.params['useVGG'] and not self.params['resume']:
            self.policy.vggsaver.restore(
                self.sess,
                '/home/s7550245/convNet/vgg-model')
            print("VGG restored.")

        self.currEpTotal = self.sess.run(self.episode_step)
        self.currEp = self.currEpTotal
        self.currActTotal = self.sess.run(self.action_step)
        self.currAct = self.currActTotal
        self.currActSumTotal = self.sess.run(self.action_step_sum)
        self.currActSum = self.currActSumTotal
        sys.stdout.flush()

        self.sess.graph.finalize()
        if self.params['async']:
            thrds = []
            for s in range(len(self.socs)):
                t = threading.Thread(target=self.play, args=(s,))
                t.daemon = True
                t.start()
                thrds.append(t)
            self.learn()
        else:
            self.play(0)

        # if self.params['async']:
        #     t = threading.Thread(target=self.learn)
        #     t.daemon = True
        #     t.start()
        #     self.play(0)
        # elif self.params['parallel']:
        #     thrds = []
        #     for s in range(self.params['parallel']):
        #         t = threading.Thread(target=self.play, args=(s,))
        #         t.daemon = True
        #         thrds.append(t)
        #         t.start()
        #     self.learn()

    def play(self, tid):
        a = 0
        ac = 0
        localEp = 0
        localAct = 0
        localActSum = 0
        while True:
            localEp += 1
            printT("Agent {}: trying to get lock for currEp counter".format(
                tid))
            with self.lock:
                self.currEp += 1
                e = self.currEp
            self.evalu = False
            if e % 10 == 0:
                self.evalu = True

            # epsilon = 1.0 / (math.pow(e+1, 1.0/3.0))
            oldScore = 0
            terminal = False
            ep_reward = 0
            ep_ave_max_q = 0
            if self.params['loadLevel'] is None:
                if self.evalu:
                    lvl = 21
                    self.socs[tid].loadLevel(lvl)
                else:
                    lvl = self.socs[tid].loadRandLevel()
            else:
                lvl = self.params['loadLevel']
                self.socs[tid].loadLevel(lvl)

            self.socs[tid].fillObs()
            state = self.socs[tid].preprocessDataForNN(vgg=self.useVGG)
            self.socs[tid].findSlingshot()

            step = 0
            samples = []
            while not terminal:
                self.socs[tid].birdCnt = self.socs[tid].birdCount(lvl) - step
                if self.socs[tid].getState() != 5:
                    printT("does this happen?")
                    break

                step += 1
                localAct += 1
                printT("Agent {}: trying to get lock for currAct counter".format(tid))
                with self.lock:
                    self.currAct += 1
                    ac = self.currAct
                tmp_step = min(ac, self.annealSteps)
                currEpsilon = (self.startEpsilon - self.endEpsilon) * \
                               (1 - tmp_step / self.annealSteps) + \
                               self.endEpsilon
                if (False and \
                    (not self.evalu) and
                    (e < self.explT or np.random.rand() < currEpsilon)):
                    action = np.array([[np.random.rand(),
                                        np.random.rand(),
                                        np.random.rand()]])
                    a_scaled = action * np.array([[50.0, 9000.0, 4000.0]])
                    printT("Agent {}: Step: {}/{} ({}/{}) Next action (e-greedy {}): {}".format(
                        tid,
                        localAct-localActSum,
                        localAct,
                        ac-a,
                        ac,
                        currEpsilon,
                        a_scaled))

                else:
                    action = self.policy.getActions(state)
                    a_scaled = action * np.array([[50.0, 9000.0, 4000.0]])
                    localActSum += 1
                    printT("Agent {}: trying to get lock for currActSum counter".format(tid))
                    with self.lock:
                        self.currActSum += 1
                        a = self.currActSum
                    printT("Agent {}: Step: {}/{} ({}/{}) Next action: ({}, {}, {})".format(
                        tid,
                        localActSum,
                        localAct,
                        a,
                        ac,
                        a_scaled[0][0], a_scaled[0][1], a_scaled[0][2]))

                    # self.sess.run(increment_ac_step_op)

                    if self.evalu:
                        action_summary_strEval = self.sess.run(
                            self.action_summary_opsEval,
                            feed_dict={
                                self.action_summary_varsEval[0]:a_scaled[0][0],
                                self.action_summary_varsEval[1]:a_scaled[0][1],
                                self.action_summary_varsEval[2]:a_scaled[0][2],
                            })

                        self.writerTest.add_summary(action_summary_strEval,
                                                    a-self.currActSumTotal)
                        self.writerTest.flush()
                    else:
                        action_summary_str = self.sess.run(
                            self.action_summary_ops,
                            feed_dict={
                                self.action_summary_vars[0]: a_scaled[0][0],
                                self.action_summary_vars[1]: a_scaled[0][1],
                                self.action_summary_vars[2]: a_scaled[0][2],
                            })

                        self.writerTrain.add_summary(action_summary_str,
                                                     a)
                        self.writerTrain.flush()

                score, terminal, newState = \
                    self.socs[tid].actionResponse(a_scaled[0], vgg=self.useVGG)

                printT("Agent {}: Time: {} Current score: {}".format(
                    tid, time.ctime(), score))
                if score == -1:
                    reward = 0
                elif score < oldScore:
                    printT("Agent {}: INVALID SCORE {} {}".format(
                        tid, score, oldScore))
                    reward = 0
                else:
                    reward = score - oldScore
                reward *= 0.0001
                # print(score, oldScore, reward, ep_reward)
                oldScore = score
                ep_reward += reward

                if self.evalu:
                    continue

                if self.params['mc']:
                    samples.append([state, action, reward,
                                    terminal, newState])
                else:
                    printT("Agent {}: trying to get lock for add samples".format(tid))
                    if self.prioritized:
                        with self.lock:
                            self.replay.add(
                                999.9,
                                (state, action, reward, terminal, newState))
                    else:
                        with self.lock:
                            self.replay.add(
                                state, action, reward, terminal, newState)

                if not self.params['async']:
                    self.learn()

                state = newState
                printT("")
                sys.stdout.flush()

            if not self.evalu and self.params['mc']:
                returnV = 0.0
                for samp in reversed(samples):
                    state = samp[0]
                    action = samp[1]
                    reward = samp[2]
                    terminal = samp[3]
                    newState = samp[4]
                    returnV += reward
                    printT("Agent {}: trying to get lock for add samples2".format(tid))
                    if self.prioritized:
                        with self.lock:
                            self.replay.add(999.9,
                                            (state, action, returnV,
                                             terminal, newState))
                    else:
                        with self.lock:
                            self.replay.add(state, action, returnV,
                                            terminal, newState)
            # print("episode reward: {}".format(ep_reward))

            if self.evalu:
                summary_str = self.sess.run(self.summary_opsEval, feed_dict={
                    self.summary_varsEval[0]: ep_reward,
                    self.summary_varsEval[1]: ep_ave_max_q / float(step)
                })

                self.writerTest.add_summary(summary_str, e-self.currEpTotal)
                self.writerTest.flush()
            else:
                summary_str = self.sess.run(self.summary_ops, feed_dict={
                    self.summary_vars[0]: ep_reward,
                    self.summary_vars[1]: ep_ave_max_q / float(step)
                })

                self.writerTrain.add_summary(summary_str, e)
                self.writerTrain.flush()

            # print('| Reward: {}, | Episode {}, | Qmax: {}'.
            #       format(ep_reward, e,
            #              ep_ave_max_q / float(step)))
            printT('Agent {}: Time: {} | Reward: {}, | Local Episode {} | Episode {}'.
                  format(tid, time.ctime(), ep_reward, localEp, e))

            sys.stdout.flush()
            if not self.params['async']:
                self.learn()


    def learn(self):
        while True:
            if self.replay.size() < self.startLearning or \
               self.replay.size() < self.miniBatchSize:
                if self.params['async']:
                    continue
                else:
                    return

            printT("Learning: trying to get lock for sampling")
            if self.params['prioritized']:
                with self.lock:
                    ids, s_batch, a_batch, r_batch, t_batch, ns_batch = \
                        self.replay.sample_batch(self.miniBatchSize)
                print(ids)
            else:
                with self.lock:
                    s_batch, a_batch, r_batch, t_batch, ns_batch = \
                        self.replay.sample_batch(self.miniBatchSize)

            y_batch = np.zeros((self.miniBatchSize, 1))
            if self.params['mc']:
                for i in range(self.miniBatchSize):
                    y_batch[i] = r_batch[i]
            else:
                qValsNewState = self.policy.predict_target_nn(ns_batch)
                for i in range(self.miniBatchSize):
                    if t_batch[i]:
                        y_batch[i] = r_batch[i]
                    else:
                        y_batch[i] = r_batch[i] + \
                            self.gamma * qValsNewState[i]

            # if self.params['prioritized']:
            #     for i in range(self.miniBatchSize):
            #         with self.lock:
            #             self.replay.update(ids[i], abs(y_batch[i]))

            gs, qs, delta = self.policy.update(s_batch, a_batch, y_batch)
            if self.params['prioritized']:
                printT("Learning: trying to get lock for updating samples")
                self.lock.acquire()
                for i in range(self.miniBatchSize):
                    self.replay.update(ids[i], abs(delta[i]))
                self.lock.release()
            # qs = self.policy.update(s_batch, a_batch, y_batch)
            # ep_ave_max_q += np.amax(qs)

            self.policy.update_targets()

            # if ((not self.evalu) and ((gs+1) % 1000)) == 0:
            #     with self.lock:
            #         printT("Saving model... (Time: {})".format(time.ctime()))
            #         self.sess.run(self.episode_step.assign(self.currEp))
            #         self.sess.run(self.action_step.assign(self.currAct))
            #         self.sess.run(self.action_step_sum.assign(self.currActSum))
            #         save_path = self.saver.save(self.sess,
            #                                     self.out_dir + "/model.ckpt",
            #                                     global_step=self.global_step)
            #         printT("Model saved in file: {} (Time: {}), dumping buffer ...".format(save_path, time.ctime()))
            #         self.replay.dump(os.path.join(self.out_dir,
            #                                       "replayBuffer.pickle"))
            #         printT("Buffer dumped.")

            if not self.params['async']:
                return
