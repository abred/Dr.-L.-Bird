import parseNNArgs

import getopt
import time
import socket
import sys
from driver import *
from drlbird import *
import os

# time.sleep(60)

params = parseNNArgs.parse(sys.argv[1:])

timestamp = str(int(time.time()))
jobid = os.environ['SLURM_JOBID']
out_dir = os.path.abspath(os.path.join(
    '/scratch/s7550245/Dr.-L.-Bird', "runsDDPG",
    params['version'], jobid + "_" + timestamp))

print("Number of epochs: ", params['numEpochs'])
out_dir += "_" + str(params['numEpochs'])

print("miniBatchSize: ", params['miniBatchSize'])
out_dir += "_" + str(params['miniBatchSize'])

print("usevgg", params['useVGG'])
if params['useVGG']:
    out_dir += "_" + "VGG" + str(params['top'])
    if params['stopGrad']:
        out_dir += "-stopGrad" + str(params['stopGrad'])
else:
    out_dir += "_" + "noVGG"

print("dropout", params['dropout'])
if params['dropout']:
    out_dir += "_" + "dropout" + str(params['dropout'])
else:
    out_dir += "_" + "noDropout"

print("tau", params['tau'])
if params['tau']:
    out_dir += "_" + "tau-" + str(params['tau'])

print("gamma: ", params['gamma'])
out_dir += "_gamma" + str(params['gamma'])

print("batchnorm", params['batchnorm'])
if params['batchnorm']:
    out_dir += "_" + "batchnorm"
else:
    out_dir += "_" + "noBatchnorm"

if params['prioritized']:
    out_dir += "_" + "prioritized"
else:
    out_dir += "_" + "notPrioritized"

print("weight decayActor", params['weight-decayActor'])
out_dir += "_wdA" + str(params['weight-decayActor'])

print("weight decayCritic", params['weight-decayCritic'])
out_dir += "_wdC" + str(params['weight-decayCritic'])

print("learning rateActor", params['learning-rateActor'])
out_dir += "_lrA" + str(params['learning-rateActor'])

print("learning rateCritic", params['learning-rateCritic'])
out_dir += "_lrC" + str(params['learning-rateCritic'])

print("momentumActor", params['momentumActor'])
out_dir += "_momA" + str(params['momentumActor'])

print("momentumCritic", params['momentumCritic'])
out_dir += "_momC" + str(params['momentumCritic'])

print("optimizerActor", params['optimizerActor'])
out_dir += "_optA" + params['optimizerActor']

print("optimizerCritic", params['optimizerCritic'])
out_dir += "_optC" + params['optimizerCritic']

if params['prioritized']:
    out_dir += "_" + "prio"

if params['async']:
    out_dir += "_" + "async"

if params['importanceSampling']:
    out_dir += "_" + "impSmpl"

print("mc", params['mc'])
if params['mc']:
    out_dir += "_" + "mc"

print("level", params['loadLevel'])
if params['loadLevel']:
    out_dir += "_lvl" + str(params['loadLevel'])
else:
    out_dir += "_lvlRand"

print(params['host'])

time.sleep(60)
# soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# soc.connect((params['host'], 2004))
d = DrLBird(params)
# print("test1")
# d.configure(421337, True)
# print("test2")
# d.getStatePrint()
# print("test3")
# TEST

algo = 0
if algo == 0:
    d.DDPG(out_dir)
elif algo == 3:
    d.DSPG()
elif algo == 1:
    d.REINFORCE()
elif algo == 2:
    d.loadLevel(11)
    # d.loadRandLevel()
    d.getStatePrint()

    d.zoomOut()
    d.getStatePrint()

    mid = 0
    cnt = d.birdCount()
    end = False
    while mid != 999 or end:
        print("State: {}".format(d.getState()))
        if d.getState() != 5:
            break
        d.fillObs()
        d.actManually()
        time.sleep(2)
        if cnt <= 1:
            for i in range(7):
                time.sleep(2)
                if d.getState() != 5:
                    end = True
                    break
        d.getStatePrint()
        if end:
            break

        cnt = d.birdCount()
        print("{} remaining birds".format(cnt))
        if cnt == 0:
            break


# t = input()
