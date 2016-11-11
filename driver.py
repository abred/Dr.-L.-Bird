import ctypes
import math
import time

from PIL import Image

from birdWrapper import *

from decoder import *
from encoder import *

import numpy as np
import scipy.ndimage

class Driver:
    def __init__(self, soc):
        self.enc = Encoder(soc)
        self.dec = Decoder(soc)
        self.birdCnt = 0

    """
    Configuration
    """
    def configure(self, id, p=False):
        self.enc.configure(id)
        temp = self.dec.configure()
        if p:
            print("Current round: {}".format(temp[0]))
            print("Time limit: {}".format(temp[1]))
            print("Number of levels: {}".format(temp[2]))
        return temp

    """
    Query server
    """
    def takeScreenshot(self):
        self.enc.takeScreenshot()
        return self.dec.takeScreenshot()

    def getState(self):
        self.enc.getState()
        return self.dec.getState()[0]

    def getStatePrint(self):
        self.enc.getState()
        state = self.dec.getState()[0]
        stateMap = [
            "unknown",
            "main menu",
            "episode menu",
            "level selection",
            "loading",
            "playing",
            "won",
            "lost"]
        print("Current state: {}".format(stateMap[state]))
        return state

    def getBestScores(self):
        self.enc.getBestScores()
        return self.dec.getBestScores()

    def getCurrScore(self):
        self.fillObs()
        dataTemp = np.zeros((32, 200), dtype=np.int32)
        score = lib.getCurrScore(ctypes.c_void_p(self.data.ctypes.data),
                                 ctypes.c_void_p(dataTemp.ctypes.data),
                                 ctypes.c_int(self.width),
                                 ctypes.c_int(self.height))
        print("Current Score: {}".format(score))
        # dataTemp = dataTemp * 10000
        # im = Image.fromarray(dataTemp, mode='I')
        # print("Shape: {}".format(im.size))
        # im.save("currScore.png")

        return score

    def getEndScore(self):
        w, h, rawInput = self.takeScreenshot()
        dataTemp = np.zeros((32, 100), dtype=np.int32)
        score = lib.getEndScore(rawInput,#ctypes.c_void_p(rawInput.data),
                                # ctypes.c_void_p(dataTemp.ctypes.data),
                                ctypes.c_int(self.width),
                                ctypes.c_int(self.height),
                                ctypes.c_int(110))
        # print("End Score: {}".format(score))
        # dataTemp = dataTemp * 100000
        # im = Image.fromarray(dataTemp, mode='I')
        # print("Shape: {}".format(im.size))
        # im.save("endScore.png")

        if score < 0 or score > 100000:
            self.fillObs()
            dataTemp = np.zeros((32, 200), dtype=np.int32)
            score = lib.getEndScore(rawInput,
                                    ctypes.c_int(self.width),
                                    ctypes.c_int(self.height),
                                    ctypes.c_int(100))

        if score < 0:
            score = 0
        return score


    def getCurrLevel(self):
        self.enc.getCurrLevel()
        return self.dec.getCurrLevel()

    def getMyScores(self):
        self.enc.getMyScores()
        return self.dec.getMyScores()

    """
    Execute Shot
    """
    def shoot(self, mid, fx, fy, dx, dy, t1, t2):
        if mid == 1:
            self.cartSafeShot(fx, fy, dx, dy, t1, t2)
        elif mid == 2:
            self.cartFastShot(fx, fy, dx, dy, t1, t2)
            # time.sleep(5)
        elif mid == 3:
            self.polarSafeShot(fx, fy, dx, dy, t1, t2)
        else:
            self.polarFastShot(fx, fy, dx, dy, t1, t2)
            # time.sleep(5)

    def cartSafeShot(self, fx, fy, dx, dy, t1, t2):
        self.enc.cartSafeShot(fx, fy, dx, dy, t1, t2)
        return self.dec.shot()

    def cartFastShot(self, fx, fy, dx, dy, t1, t2):
        self.enc.cartFastShot(fx, fy, dx, dy, t1, t2)
        return self.dec.shot()

    def polarSafeShot(self, fx, fy, theta, r, t1, t2):
        self.enc.polarSafeShot(fx, fy, theta, r, t1, t2)
        return self.dec.shot()

    def polarFastShot(self, fx, fy, theta, r, t1, t2):
        self.enc.polarFastShot(fx, fy, theta, r, t1, t2)
        return self.dec.shot()

    def safeShotSeq(self, shots, num):
        self.enc.safeShotSeq(shots, num)
        return self.dec.shotSeq(num)

    def fastShotSeq(self, shots, num):
        self.enc.fastShotSeq(shots, num)
        return self.dec.shotSeq(num)

    """
    Zoom, etc
    """
    def zoomOut(self):
        self.enc.zoomOut()
        return self.dec.zoomOut()

    def zoomIn(self):
        self.enc.zoomIn()
        return self.dec.zoomIn()

    def clickCenter(self):
        self.enc.clickCenter()
        return self.dec.clickCenter()

    """
    Select Level
    """
    def loadLevel(self, lvl):
        self.enc.loadLevel(lvl)
        return self.dec.loadLevel()

    def restartLevel(self):
        self.enc.restartLevel()
        return self.dec.restartLevel()

    def loadRandLevel(self):
        lvl = np.random.randint(1,22)
        print("loading level {}".format(lvl))
        self.loadLevel(lvl)
        # self.loadLevel(1)
        self.zoomOut()
        self.getStatePrint()


    """
    Process data
    """
    def fillObs(self):
        w, h, rawInput = self.takeScreenshot()
        # print("w: {} h: {}".format(w, h))
        npInput = np.frombuffer(rawInput, np.dtype(np.uint8))
        # print("Shape: {}".format(npInput.shape))
        npInput = np.reshape(npInput, (h, w, 3))

        self.data = np.zeros((h * w), dtype=np.int32)

        self.height = h
        self.width = w
        lib.processScreenShot(ctypes.c_void_p(npInput.ctypes.data),
                              ctypes.c_void_p(self.data.ctypes.data),
                              ctypes.c_int(self.width),
                              ctypes.c_int(self.height))
        # temp = np.copy(self.data)
        # temp = temp * 100
        # temp = np.reshape(temp, (h,w))
        # im = Image.fromarray(temp, mode='I')
        # print("Shape: {}".format(im.size))
        # self.birdCnt += 1
        # im.save("test" + str(self.testcnt) + ".png")
        # print("end fillobs")

        self.data.shape = (h, w)
        return self.data

    def preprocessDataForNN(self):
        # self.dataNN = np.zeros((self.height * self.width), dtype=np.float)
        # self.dataNN = np.zeros((1, 448 * 832), dtype=np.float)

        # lib.preprocessDataForNN(ctypes.c_void_p(self.data.ctypes.data),
        #                         ctypes.c_void_p(self.dataNN.ctypes.data),
        #                         ctypes.c_int(self.width),
        #                         ctypes.c_int(self.height))
        # return self.dataNN
        self.dataNN = self.data.astype(np.float32) / 512.0
        self.dataNN = scipy.ndimage.zoom(self.dataNN, 0.125, order=1)
        self.dataNN.shape = (1, self.height * self.width / 8 / 8)
        return self.dataNN

    def findSlingshot(self):
        linPos = lib.findSlingshotCenter(ctypes.c_void_p(self.data.ctypes.data),
                                         self.width, self.height)
        self.currCenterX = linPos % self.width
        self.currCenterY = math.floor(linPos/self.width)
        print("Current slingshot position: {}, {}".format(self.currCenterX,
                                                          self.currCenterY))



    def birdCount(self):
        self.zoomIn()
        self.fillObs()
        # self.findSlingshot()
        # while self.currCenterY == 0 and self.currCenterX == 0:
        #     self.clickCenter()
        #     self.findSlingshot()
        self.birdCnt = lib.calcLives()
        self.zoomOut()
        print("bird count: {}".format(self.birdCnt))
        return self.birdCnt

    def actManually(self):
        self.findSlingshot()

        mid = int(input("Shot type: "))
        # fx = int(input("x-coord: "))
        fx = self.currCenterX
        # fy = int(input("y-coord: "))
        fy = self.currCenterY
        dx = int(input("dx/theta: "))
        dy = int(input("dy/radius: "))
        t1 = int(input("time0: "))
        t2 = int(input("time1: "))

        self.shoot(mid, fx, fy, dx, dy, t1, t2)

    def act(self, action):
        # self.findSlingshot()
        print("action: {}".format(action))
        mid = 4
        fx = self.currCenterX
        fy = self.currCenterY
        dx = action[0]
        dy = action[1]
        t1 = 0
        t2 = action[2]

        self.shoot(mid, fx, fy, dx, dy, t1, t2)

    def actionResponse(self, action):
        score = self.getCurrScore()
        self.act(action)
        time.sleep(0.5)
        scoreTmp = self.getCurrScore()
        if scoreTmp >= score:
            score = scoreTmp
        if self.birdCnt <= 1:
            for i in range(4):
                time.sleep(2)
                if self.getState() != 5:
                    break

        self.fillObs()
        newState = self.preprocessDataForNN()

        gameState = self.getStatePrint()
        terminal = False
        if gameState == 5:    # playing
            score = self.getCurrScore()
        elif gameState == 6:  # won
            score = self.getEndScore()
            terminal = True
        elif gameState == 7:  # lost
            # score = -10000
            terminal = True
        else:
            raise Exception("should not get here")

        return score, terminal, newState
