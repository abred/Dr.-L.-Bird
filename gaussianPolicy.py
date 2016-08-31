import pickle

import numpy as np

from tfActorNetwork import *


defaultSigma = np.array([5.0, 5.0, 250])
defaultMu = np.array([-40.0, 30.0, 2000.0])


class GaussianPolicy:

    def __init__(self, resume=False, mu=defaultMu, sigma=defaultSigma):
        self.mu = mu
        self.sigma = sigma

        if resume:
            self.model = pickle.load(open('save.p', 'rb'))
        else:
            self.model = ActorNetwork()

    def getAction(self):
        a = self.sigma * np.random.randn(3) + self.mu

        print("Next action: {}\n".format(a))
        a[0] = np.clip(a[0], -50, 50)
        a[1] = np.clip(a[1], -50, 50)
        a[2] = np.clip(a[2], 0, 5000)
        print("Next action: {} (clipped)\n".format(a))
        return a

    def forward(self, states):
        o = self.model.run_inference(states)
        self.mu = o[0][:3]
        self.sigma = o[0][3:]
        return o

    def backward(self, states, actions, returns):
        self.model.run_training(states, actions, returns)
