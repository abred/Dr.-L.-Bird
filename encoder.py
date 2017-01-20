import struct

import numpy as np


class Encoder:
    def __init__(self, soc):
        self.soc = soc

    """
    Configuration
    """
    def configure(self, id):
        self.soc.send(struct.pack('>bi', 1, id))

    """
    Query server
    """
    def takeScreenshot(self):
        self.soc.send(struct.pack('>b', 11))

    def getState(self):
        self.soc.send(struct.pack('>b', 12))

    def getBestScores(self):
        self.soc.send(struct.pack('>b', 13))

    def getCurrLevel(self):
        self.soc.send(struct.pack('>b', 14))

    def getMyScores(self):
        self.soc.send(struct.pack('>b', 23))

    """
    Execute shot
    """
    @staticmethod
    def shot(mid, fx, fy, arg1, arg2, t1, t2):
        # print(mid,fx,fy,arg1,arg2,t1,t2)
        # return struct.pack('>biiiiii', mid, fx, fy, arg1, arg2, t1, t2)
        # print("shot: {} {}".format(fx, fy))
        return struct.pack('>biiiiii', np.int32(mid),
                           np.int32(fx), np.int32(fy),
                           np.int32(arg1), np.int32(arg2),
                           np.int32(t1), np.int32(t2))

    @staticmethod
    def shotSeq(mid, shots, num):
        temp = bytearray(1+len(shots) * (1+6*4))
        struct.pack_into('>b', temp, 0, mid)
        offset = 1
        for s in shots:
            struct.pack_into('>biiiiii', temp, offset, *s[:7])
            offset += 1 + 6*4
        return temp

    def cartSafeShot(self, fx, fy, dx, dy, t1, t2):
        self.soc.send(self.shot(31, fx, fy, dx, dy, t1, t2))

    def cartFastShot(self, fx, fy, dx, dy, t1, t2):
        self.soc.send(self.shot(41, fx, fy, dx, dy, t1, t2))

    def polarSafeShot(self, fx, fy, theta, r, t1, t2):
        self.soc.send(self.shot(32, fx, fy, theta, r, t1, t2))

    def polarFastShot(self, fx, fy, theta, r, t1, t2):
        self.soc.send(self.shot(42, fx, fy, theta, r, t1, t2))

    def safeShotSeq(self, shots, num):
        self.soc.send(self.shotSeq(33, shots, num))

    def fastShotSeq(self, shots, num):
        self.soc.send(self.shotSeq(43, shots, num))


    """
    Zoom, etc
    """
    def zoomOut(self):
        self.soc.send(struct.pack('>b', 34))

    def zoomIn(self):
        self.soc.send(struct.pack('>b', 35))

    def clickCenter(self):
        self.soc.send(struct.pack('>b', 36))

    """
    Select level
    """
    def loadLevel(self, lvl):
        self.soc.send(struct.pack('>bb', 51, lvl))

    def restartLevel(self):
        self.soc.send(struct.pack('>b', 52))
