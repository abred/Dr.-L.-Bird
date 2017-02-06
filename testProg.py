import getopt
import time
import socket
import sys
from driver import *
from drlbird import *

resume = False
useVGG = False
top = None
try:
    print(sys.argv)
    opts, args = getopt.getopt(sys.argv[1:],"r:vt:h:")
    print(opts, args)
except getopt.GetoptError:
    print('args parse error')
    print('args: ', argv)
    print('using default values')
for opt, arg in opts:
    print(opt, arg)
    if opt == '-r':
        resume = True
        out_dir = arg
        print("resuming...")
    elif opt == '-v':
        useVGG = True
    elif opt == '-t':
        top = int(arg)
    elif opt == '-h':
        host = arg

print("useVGG", useVGG)
print("host", host)
print("top", top)

soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
soc.connect((host, 2004))
d = DrLBird(soc)
print("test1")
d.configure(421337, True)
print("test2")
d.getStatePrint()
print("test3")
# TEST

algo = 0
if algo == 0:
    if resume:
        print("resuming... ", out_dir)
        if "VGG" in out_dir:
            useVGG = True
            tmp = out_dir.split("top")[1]
            print(tmp[:2])
            top = int(tmp[:2])
        d.DDPG(resume=True, out_dir=out_dir, evalu=False,
               useVGG=useVGG, top=top)
    else:
        print("new start...")
        d.DDPG(useVGG=useVGG, top=top)
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
