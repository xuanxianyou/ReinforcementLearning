import collections
import random


class ReplayBuffer(object):

    def __init__(self, max_size):
        self.buffer = collections.deque(maxlen=max_size)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return batch

    def __len__(self):
        return len(self.buffer)