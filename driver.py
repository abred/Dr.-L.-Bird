from keras.applications.vgg16 import preprocess_input

import sys
import ctypes
import math
import time

# from PIL import Image
import png

from birdWrapper import *

from decoder import *
from encoder import *

import numpy as np
import scipy.ndimage

def printT(s):
    sys.stdout.write(s + '\n')

class Driver:
    def __init__(self, soc, tid):
        self.enc = Encoder(soc)
        self.dec = Decoder(soc)
        self.birdCnt = 0
        self.data = np.zeros((480 * 840), dtype=np.int32)
        self.cnt = 0
        self.tid = tid
        self.wrapper = Wrapper()

        self.birdsPerLevel = [
            3, # 1
            5, # 2
            4, # 3
            4, # 4
            4, # 5
            4, # 6
            4, # 7
            4, # 8
            4, # 9
            5, # 10
            4, # 11
            4, # 12
            4, # 13
            4, # 14
            4, # 15
            5, # 16
            3, # 17
            5, # 18
            4, # 19
            5, # 20
            8, # 20
        ]

    """
    Configuration
    """
    def configure(self, id, p=False):
        self.enc.configure(id)
        temp = self.dec.configure()
        if p:
            printT("Agent {}: Current round: {}".format(self.tid, temp[0]))
            printT("Agent {}: Time limit: {}".format(self.tid, temp[1]))
            printT("Agent {}: Number of levels: {}".format(self.tid, temp[2]))
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
        printT("Agent {}: Current state: {}".format(self.tid, stateMap[state]))
        return state

    def getBestScores(self):
        self.enc.getBestScores()
        return self.dec.getBestScores()

    def getCurrScore(self):
        self.fillObs()
        data = np.zeros((self.height * self.width), dtype=np.int32)
        self.wrapper.processScreenShot(ctypes.c_void_p(self.data.ctypes.data),
                                       ctypes.c_void_p(data.ctypes.data),
                                       ctypes.c_int(self.width),
                                       ctypes.c_int(self.height))

        dataTemp = np.zeros((32, 200), dtype=np.int32)
        score = self.wrapper.getCurrScore(
            ctypes.c_void_p(data.ctypes.data),
            ctypes.c_void_p(dataTemp.ctypes.data),
            ctypes.c_int(self.width),
            ctypes.c_int(self.height))
        printT("Agent {}: Current Score: {}".format(self.tid, score))
        # dataTemp = dataTemp * 10000
        # im = Image.fromarray(dataTemp, mode='I')
        # print("Shape: {}".format(im.size))
        # im.save("currScore.png")

        return score

    def getEndScore(self):
        w, h, rawInput = self.takeScreenshot()
        dataTemp = np.zeros((32, 100), dtype=np.int32)
        score = self.wrapper.getEndScore(
            rawInput,#ctypes.c_void_p(rawInput.data),
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
            score = self.wrapper.getEndScore(rawInput,
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
            return self.cartSafeShot(fx, fy, dx, dy, t1, t2)
        elif mid == 2:
            return self.cartFastShot(fx, fy, dx, dy, t1, t2)
            # time.sleep(5)
        elif mid == 3:
            return self.polarSafeShot(fx, fy, dx, dy, t1, t2)
        else:
            return self.polarFastShot(fx, fy, dx, dy, t1, t2)
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
        lvl = np.random.randint(1,21)
        printT("Agent {}: loading level {}".format(self.tid, lvl))
        self.loadLevel(lvl)
        self.zoomOut()
        self.getStatePrint()
        return lvl


    """
    Process data
    """
    def fillObs(self, store=None):
        w, h, rawInput = self.takeScreenshot()
        npInput = np.frombuffer(rawInput, np.dtype(np.uint8))
        self.data = np.reshape(npInput, (h, w, 3))
        self.height = h
        self.width = w

        if store is not None:
            temp = np.copy(self.data)
            # temp = temp * 10
            temp = np.reshape(temp, (h,w*3))
            # print(temp)
            # print(temp.shape)
            png.from_array(temp.astype(np.uint8), 'RGB').save(
                "/scratch/s7550245/Dr.-L.-Bird/firstFrame_" +
                str(store) + ".png")


        return self.data

    def preprocessDataForNN(self, store=None, vgg=False):
        if vgg:
            # self.dataNN = preprocess_input(self.data)
            self.dataNN = self.data.astype(np.float32) - 127.0
            self.dataNN = scipy.ndimage.zoom(self.dataNN,
                                         (224.0/480.0, 224.0/840.0, 1),
                                         order=1)
        else:
            self.dataNN = self.data.astype(np.float32) / 255.0
            self.dataNN = scipy.ndimage.zoom(self.dataNN, (0.25, 0.25, 1),
                                             order=1)
        # self.dataNN.shape = (1, self.height / 4, self.width / 4, 3)
        self.dataNN.shape = (1,
                             self.dataNN.shape[0],
                             self.dataNN.shape[1],
                             self.dataNN.shape[2])
        # printT(str(self.dataNN.shape))
        if store is not None:
            tmp = np.copy(self.dataNN)
            tmp *= 256.0
            # tmp.shape =  (self.height / 4, self.width / 4 * 3)
            tmp.shape = (self.dataNN.shape[1],
                         self.dataNN.shape[2] * self.dataNN.shape[3])
            png.from_array(tmp.astype(np.uint8), 'RGB').save(
                "/scratch/s7550245/Dr.-L.-Bird/processedFrame_" +
                str(self.cnt) + ".png")
            self.cnt += 1

        return self.dataNN

    def findSlingshot(self):
        self.fillObs()
        data = np.zeros((self.height * self.width), dtype=np.int32)

        self.wrapper.processScreenShot(ctypes.c_void_p(self.data.ctypes.data),
                                       ctypes.c_void_p(data.ctypes.data),
                                       ctypes.c_int(self.width),
                                       ctypes.c_int(self.height))

        linPos = \
            self.wrapper.findSlingshotCenter(ctypes.c_void_p(data.ctypes.data),
                                             self.width, self.height)
        self.currCenterX = linPos % self.width
        self.currCenterY = math.floor(linPos/self.width)
        # printT("Agent {}: Current slingshot position: {}, {}".format(
        #     self.tid, self.currCenterX,
        #     self.currCenterY))



    def birdCount(self, currLvl=None):
        if currLvl is None:
            self.zoomIn()
            self.fillObs()
            data = np.zeros((self.height * self.width), dtype=np.int32)

            self.wrapper.processScreenShot(
                ctypes.c_void_p(self.data.ctypes.data),
                ctypes.c_void_p(data.ctypes.data),
                ctypes.c_int(self.width),
                ctypes.c_int(self.height))

            self.zoomOut()
            return self.wrapper.calcLives()
        else:
            return self.birdsPerLevel[currLvl-1]

    def actManually(self):
        self.findSlingshot()

        mid = int(input("Shot type: "))
        # fx = int(input("x-coord: "))
        fx = self.currCenterX
        # fy = int(input("y-coord: "))
        fy = self.currCenterY
        dx = int(input("dx/radius: "))
        dy = int(input("dy/theta: "))
        t1 = int(input("time0: "))
        t2 = int(input("time1: "))

        self.shoot(mid, fx, fy, dx, dy, t1, t2)

    def act(self, action):
        mid = 4
        fx = self.currCenterX
        fy = self.currCenterY
        dx = action[0]
        dy = action[1]
        t1 = 0
        t2 = action[2]
        # printT("Agent {}: ({}, {}, {})".format(self.tid, action, dx, dy))

        self.shoot(mid, fx, fy, dx, dy, t1, t2)

    # def act(self, action):
    #     mid = 2
    #     fx = self.currCenterX
    #     fy = self.currCenterY
    #     if action[0] < -50:
    #         dx = -100 - action[0]
    #     else:
    #         dx = action[0]
    #     if action[1] > 50:
    #         dy = 100 - action[1]
    #     else:
    #         dy = action[1]
    #     t1 = 0
    #     t2 = action[2]
    #     print(action, dx, dy)

    #     self.shoot(mid, fx, fy, dx, dy, t1, t2)

    def actionResponse(self, action, vgg=False):
        # score = self.getCurrScore()
        self.act(action)
        time.sleep(2)
        # scoreTmp = self.getCurrScore()
        # if scoreTmp >= score:
        #     score = scoreTmp
        # if self.birdCnt <= 1:
        #     for i in range(4):
        #         time.sleep(2)
        #         if self.getState() != 5:
        #             break

        self.fillObs()
        newState = self.preprocessDataForNN(vgg=vgg)

        # gameState = self.getStatePrint()
        gameState = self.getState()
        terminal = False
        if gameState == 5:    # playing
            score = self.getCurrScore()
        elif gameState == 6:  # won
            score = self.getEndScore()
            terminal = True
            printT("Agent {}: WON".format(self.tid))
        elif gameState == 7:  # lost
            score = -1
            terminal = True
            print("Agent {}: LOST".format(self.tid))
        else:
            raise Exception("should not get here")

        return score, terminal, newState
