import numpy as np

from akpy.buffers_at_BS import Buffers
from schedulers.scheduler import Scheduler

class MaxTh(Scheduler):

    def __init__(self, K: int, F: list, buffers: Buffers):
        self.K = K
        self.F = F
        self.pointer = K-1
        self.alloc_users = [[] for i in range(len(F))]
        self.buffers = buffers
        # important for plots
        self.name = 'Max th'
        self.exp_thr = None

    def policy_action(self) -> int:
        # choose between the throughput available to send or the buffer size to send values
        buffer_occ = self.buffers.buffer_occupancies * 1.0
        #using QCI information
        thr = np.minimum(self.exp_thr, buffer_occ)# choose between the throughput available to send or the buffer size to send values

        return (np.argsort(-1 * thr)[: np.sum(self.F)])

    def reset(self):
        self.buffers.reset_buffers()