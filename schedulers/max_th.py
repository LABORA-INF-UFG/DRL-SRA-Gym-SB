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
        thr = np.minimum(self.exp_thr, self.buffers.buffer_occupancies)# choose between the throughput available to send or the buffer size to send values

        #return (
        #        np.argsort(-1 * thr)[: (np.sum(self.F))] + 1
        #)  # Sort UEs throughput in th descending order
        return (np.argsort(-1 * thr)[: np.sum(self.F)])

    def reset(self):
        self.buffers.reset_buffers()