import time
import socket
from driver import *
from drlbird import *

soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
soc.connect(("127.0.0.1", 2004))
d = DrLBird(soc)
print("test1")
d.configure(421337, True)
print("test2")
d.getStatePrint()
print("test3")
# TEST

algo = 3
if algo == 0:
    d.DDPG()
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
