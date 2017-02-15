import parseNNArgs

import getopt
import time
import socket
import sys
from driver import *
from drlbird import *



params = parseNNArgs.parse(sys.argv[1:])

timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(
    '/scratch/s7550245/convNet',
    "runs", timestamp))

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

print("batchnorm", params['batchnorm'])
if params['batchnorm']:
    out_dir += "_" + "batchnorm"
else:
    out_dir += "_" + "noBatchnorm"

if params['prioritized']:
    out_dir += "_" + "prioritized"
else:
    out_dir += "_" + "notPrioritized"

print("weight decay", params['weight-decay'])
out_dir += "_wd" + str(params['weight-decay'])

print("learning rate", params['learning-rate'])
out_dir += "_lr" + str(params['learning-rate'])

print("momentum", params['momentum'])
out_dir += "_mom" + str(params['momentum'])

print("optimizer", params['optimizer'])
out_dir += "_opt" + params['optimizer']

soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
soc.connect((params['host'], 2004))
d = DrLBird(soc)
print("test1")
d.configure(421337, True)
print("test2")
d.getStatePrint()
print("test3")
# TEST

algo = 0
if algo == 0:
    d.DDPG(params)
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


t = input()
