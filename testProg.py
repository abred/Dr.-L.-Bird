import socket
from driver import *
from drlbird import *

soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
soc.connect(("127.0.0.1", 2004))
d = DrLBird(soc)

d.configure(421337, True)
d.getStatePrint()

# TEST


if True:

    while True:
    # for i in range(0,3):

        d.REINFORCE()

        st = d.getState()
        print("State: {}".format(st))
        # if st != 5:
            # break
else:
    d.loadLevel(20)
    d.getStatePrint()

    d.zoomOut()
    d.getStatePrint()

    mid = 0
    while mid != 999:
        print("State: {}".format(d.getState()))
        if d.getState() != 5:
            break

        d.fillObs()
        d.actManually()

t = input()
