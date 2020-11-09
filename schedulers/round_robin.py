from akpy.buffers_at_BS import Buffers
from schedulers.scheduler import Scheduler


class RoundRobin(Scheduler):

    def __init__(self, K: int, F: list, buffers: Buffers):
        self.K = K
        self.F = F
        self.pointer = K-1
        self.alloc_users = [[] for i in range(len(F))]
        self.buffers = buffers
        # important for plot
        self.name = 'Round Robin'

    def policy_action(self) -> int:
        # update UE pointer
        self.pointer += 1
        if self.pointer > self.K-1:
            self.pointer = 0

        return self.pointer


