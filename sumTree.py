import math
import traceback
import sys
import os
import pickle
import numpy as np

class SumTree:
    write = 0
    epsilon = 0.01
    alpha = 0.6
    beta = 0.4
    maxP = 1.0

    def __init__(self, capacity, miniBatchSize, annealSteps):
        self.capacity = capacity
        self.annealSteps = annealSteps
        self.miniBatchSize = miniBatchSize
        self.tree = np.zeros( 2*capacity - 1 )
        self.data = np.zeros( capacity, dtype=object )
        self.full = False
        self.sumP = 0.0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def size(self):
        if self.full:
            return self.capacity
        else:
            return self.write

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        # print("add data:", data)
        if type(data) is int:
            print("invalid type !!!")
            print("Exception in user code:")
            print('-'*60)
            traceback.print_exc(file=sys.stdout)
            print('-'*60)
            sys.stdout.flush()
            os._exit(-2)
            # exit()
        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
            self.full = True

    def update(self, idx, p):
        if p is None:
            p = self.maxP
        else:
            p = (p + self.epsilon) ** self.alpha
        change = p - self.tree[idx]

        self.sumP += change
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        if self.full:
            if idx >= self.capacity*2 - 1:
                print("invalid access, idx to high", idx, self.full, self.write, s, self.total())
                idx = self.capacity*2 - 2
        else:
            if idx >= self.write + self.capacity - 1:
                print("invalid access, idx to high", idx, self.full, self.write, s, self.total())
                idx = self.write + self.capacity - 2
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])

    def sample_batch(self, batch_size):
        batch = []
        ids = []
        ps = []

        # sample
        # segment = self.total() / batch_size

        for i in range(batch_size):
            # a = segment * i
            # b = segment * (i + 1)
            a = 0.0
            b = self.total()
            s = np.random.uniform(a, b)
            (idx, p, data) = self.get(s)
            # print("sample data:", data)
            if type(data) is int:
                print("sample: invalid type !!!", s, idx, p, self.write, self.total())
                print("Exception in user code:")
                print('-'*60)
                traceback.print_exc(file=sys.stdout)
                print( '-'*60)
                sys.stdout.flush()
                # exit()

            # batch.append((data[0], data[1], data[2], data[3], data[4]))
            batch.append(data)
            ids.append(idx)
            ps.append(p/self.sumP)
            # print(segment, a, b, i, idx, p, self.sumP,
            #       self.total(), self.size())
            # print(a, b, s, i, idx, p, self.sumP,
            #       self.total(), self.size())

        s_batch = np.squeeze(np.array([_[0] for _ in batch]))
        a_batch = np.reshape(np.array([_[1] for _ in batch]), (batch_size, 3))
        r_batch = np.reshape(np.array([_[2] for _ in batch]), (batch_size, 1))
        t_batch = np.reshape(np.array([_[3] for _ in batch]), (batch_size, 1))
        s2_batch = np.squeeze(np.array([_[4] for _ in batch]))
        ps = np.array(ps)

        w_batch = np.zeros((self.miniBatchSize, 1))
        for i in range(self.miniBatchSize):
            w_batch[i] = math.pow(self.size() * ps[i],
                                  -self.beta)
        self.beta = min(1.0,
                        self.beta + 1.0/self.annealSteps)
        self.alpha = max(0.0, self.alpha - 1.0/self.annealSteps)
        return ids, w_batch, s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self):
        self.tree = np.zeros( 2*self.capacity - 1 )
        self.data = np.zeros( self.capacity, dtype=object )
        self.write = 0

    def dump(self, fn):
        with open(fn, 'w') as f:
            pickle.dump((self.tree, self.data, self.write), f)

    def load(self, fn):
        with open(fn, 'r') as f:
            self.tree, self.data, self.write = pickle.load(f)
