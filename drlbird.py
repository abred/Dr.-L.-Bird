import time

import numpy as np

from driver import Driver

from gaussianPolicy import GaussianPolicy


class DrLBird(Driver):
    policy = GaussianPolicy()

    def REINFORCE(self):
        self.loadRandLevel()

        scores = []
        states = []
        actions = []
        os = []
        st = self.getStatePrint()
        numAct = 0
        while True:
            self.fillObs()
            states.append(self.preprocessDataForNN())
            o = self.policy.forward(states[-1])
            os.append(o)

            print("nn output: {}".format(o))
            actions.append(self.policy.getAction())
            self.act(actions[-1])
            numAct += 1
            time.sleep(3)
            tempScore = self.getCurrScore()
            time.sleep(11)
            st = self.getStatePrint()

            if st == 5:  # playing
                scores.append(self.getCurrScore())
            elif st == 6:  # won
                scores.append(self.getEndScore())

                # numAct += 1
                # scores.append(scores[-1]+50000)
                break
            elif st == 7:  # lost
                # score???
                # scores.append(tempScore)
                scores.append(scores[-1]-10000)
                break
            else:
                raise Exception("should not get here")
                break

        print("Scores: {}".format(scores))
        rewards = []
        returns = []
        rewards.append(scores[0])
        returns.append(scores[numAct-1])
        for b in range(1, numAct):
            rewards.append(scores[b] - scores[b-1])
            returns.append(returns[b-1] - rewards[b-1])

        print("Rewards: {}".format(rewards))
        print("Returns: {}".format(returns))

        self.policy.backward(np.vstack(states),
                             np.vstack(actions),
                             np.vstack(returns))
