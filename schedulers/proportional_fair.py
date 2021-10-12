import numpy as np

from akpy.buffers_at_BS import Buffers
from schedulers.scheduler import Scheduler


class ProportionalFair(Scheduler):

    def __init__(self, K: int, F: list, buffers: Buffers):
        self.K = K
        self.F = F
        self.pointer = K-1
        self.alloc_users = [[] for i in range(len(F))]
        self.buffers = buffers
        # important for plots
        self.name = 'Proportional Fair'
        self.exp_thr = None
        self.thr_count = [[1.] for i in range(K)]
        self.max_rate = 1.0
        #self.recent_rate = self.max_rate * np.ones((self.K, len(self.F)))  # in bps/Hz
        self.recent_rate = self.max_rate * np.ones((self.K))  # in bps/Hz



    def policy_action_(self) -> int:
        avg_thr = []
        # compute the throughput mean for each UE
        for i, v in enumerate(self.thr_count):
            avg_thr.append(np.mean(v))
        buffer_occ = self.buffers.buffer_occupancies * 1.0
        thr = [[] for f in self.F]
        metric = [[] for f in self.F]
        bu_count = [0 for f in self.F]
        best_users = []
        # choose between the throughput available to send or the buffer size to send values
        #thr = np.minimum(self.exp_thr, self.buffers.buffer_sizes)
        for i, f in enumerate(self.F):
            thr[i] = np.minimum(self.recent_rate[:, i],
                                buffer_occ)  # choose between the throughput available to send or the buffer size to send values

            # throughput available divided by last throughput reached by each UE
            metric[i] = (thr[i] / avg_thr)
            for bu in (np.argsort(-1 * metric[i])):
                if bu not in best_users and bu_count[i] < f:
                    best_users.append(bu)
                    bu_count[i] += 1

        # Sort UEs metric in th descending order
        return best_users

    def policy_action(self) -> int:
        avg_thr = []
        # compute the throughput mean for each UE
        for i, v in enumerate(self.thr_count):
            avg_thr.append(np.mean(v))
        buffer_occ = self.buffers.buffer_occupancies

        # choose between the throughput available to send or the buffer size to send values
        #thr = np.minimum(self.exp_thr, self.buffers.buffer_sizes)
        thr = np.minimum(self.recent_rate, buffer_occ)
        # throughput available divided by the last known throughput reached by each UE
        metric = (thr / avg_thr)

        # Sort UEs metric in th descending order
        return (np.argsort(-1 * metric)[: np.sum(self.F)])


    def reset(self):
        self.buffers.reset_buffers()
        self.thr_count = [[1.] for i in range(self.K)]
        self.alloc_users = [[] for i in range(len(self.F))]
        self.recent_rate = self.max_rate * np.ones((self.K))  # in bps/Hz
        self.thr_count = [[1.] for i in range(self.K)]

    def update_recent_rate_(self, rate):
        for i,r in enumerate(rate):
            if r > 0:
                for fi, f in enumerate(self.F):
                    if i in self.alloc_users[fi]:
                        self.recent_rate[i,fi] = r

    def update_recent_rate(self, rate):
        for i, r in enumerate(rate):
            if r > 0:
                for fi, f in enumerate(self.F):
                    if i in self.alloc_users[fi]:
                        self.recent_rate[i] = r