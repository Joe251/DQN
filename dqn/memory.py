from collections import deque
import random
class Memory():
    def __init__(self):
        self.D = []
        self.count = 0
    def add(self, screen, action, reward, terminal):
        self.D.append((screen, action, reward, terminal))
        self.count += 1
    def sample(self, batch_size):
        indexs = random.sample(range(self.count-1), batch_size)
        return [[self.D[i][0], self.D[i][1], self.D[i][2], self.D[i+1][0], self.D[i][3]] for i in indexs]
    def popleft(self):
        self.D = self.D[1:]
        self.count -= 1