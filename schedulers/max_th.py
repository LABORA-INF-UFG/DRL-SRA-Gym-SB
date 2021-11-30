import numpy as np

from akpy.buffers_at_BS import Buffers
from schedulers.scheduler import Scheduler

class MaxTh(Scheduler):

    def __init__(self, K: int, F: list, buffers: Buffers):
        self.K = K
        self.F = F
        self.alloc_users = [[] for i in range(len(F))]
        self.buffers = buffers
        # important for plots
        self.name = 'Max th'
        self.exp_thr = None
        self.max_rate = 10000000.0
        #self.recent_rate = self.max_rate * np.ones((self.K, len(self.F)))  # in bps/Hz
        self.recent_rate = self.max_rate * np.ones((self.K))  # in bps/Hz

    # partial obs
    def policy_action_(self) -> int:
        # choose between the throughput available to send or the buffer size to send values
        buffer_occ = self.buffers.buffer_occupancies *  1.17 #1.17
        thr = None
        bu_count = 0
        best_users = []
        #using QCI information
        thr = np.minimum(self.recent_rate, buffer_occ)# choose between the throughput available to send or the buffer size to send values
        for bu in (np.argsort(-1 * thr)):
            if bu not in best_users and bu_count < np.sum(self.F):
                best_users.append(bu)
                bu_count += 1

        return best_users

    # full obs
    def policy_action(self) -> int:
        # choose between the throughput available to send or the buffer size to send values
        buffer_occ = self.buffers.buffer_occupancies * 1.17
        #using QCI information
        thr = np.minimum(self.exp_thr, buffer_occ)# choose between the throughput available to send or the buffer size to send values

        return (np.argsort(-1 * thr)[: np.sum(self.F)])

    def reset(self):
        self.buffers.reset_buffers()
        self.alloc_users = [[] for i in range(len(self.F))]
        self.recent_rate = self.max_rate * np.ones((self.K))  # in bps/Hz

    def update_recent_rate_(self, rate):
        for i,r in enumerate(rate):
            if r > 0:
                for fi, f in enumerate(self.F):
                    if i in self.alloc_users[fi]:
                        self.recent_rate[i,fi] = r

    def update_recent_rate(self, rate):
        for i,r in enumerate(rate):
            if r > 0:
                for fi, f in enumerate(self.F):
                    if i in self.alloc_users[fi]:
                        self.recent_rate[i] = r