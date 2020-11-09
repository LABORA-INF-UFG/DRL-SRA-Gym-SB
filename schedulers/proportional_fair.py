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



    def policy_action(self) -> int:
        avg_thr = []
        # compute the throughput mean for each UE
        for i, v in enumerate(self.thr_count):
            avg_thr.append(np.mean(v))

        # choose between the throughput available to send or the buffer size to send values
        thr = np.minimum(self.exp_thr, self.buffers.buffer_sizes)
        # throughput available divided by last throughput reached by each UE
        metric = (thr / avg_thr)
        # Sort UEs metric in th descending order
        pues = (np.argsort(-1 * metric))
        ues = (np.argsort(-1 * metric)[: np.sum(self.F)])
        if 10 in ues:
            print(ues)


        return (np.argsort(-1 * metric)[: np.sum(self.F)])


    def reset(self):
        self.buffers.reset_buffers()
        self.thr_count = [[1.] for i in range(self.K)]