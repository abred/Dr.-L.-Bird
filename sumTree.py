import pickle
import numpy as np

class SumTree:
    write = 0
    e = 0.01
    a = 0.6

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros( 2*capacity - 1 )
        self.data = np.zeros( capacity, dtype=object )
        self.full = False

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
        p = (p + self.e) ** self.a

        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
            self.full = True

    def update(self, idx, p):
        p = (p + self.e) ** self.a
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])

    def sample_batch(self, batch_size):
        batch = []
        ids = []

        # sample
        segment = self.total() / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            # print(segment, a, b, i)
            s = np.random.uniform(a, b)
            (idx, p, data) = self.get(s)
            batch.append((data[0], data[1], data[2], data[3], data[4]))
            ids.append(idx)

        s_batch = np.squeeze(np.array([_[0] for _ in batch]))
        a_batch = np.squeeze(np.array([_[1] for _ in batch]))
        r_batch = np.reshape(np.array([_[2] for _ in batch]), (batch_size, 1))
        t_batch = np.reshape(np.array([_[3] for _ in batch]), (batch_size, 1))
        s2_batch = np.squeeze(np.array([_[4] for _ in batch]))

        return ids, s_batch, a_batch, r_batch, t_batch, s2_batch

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
