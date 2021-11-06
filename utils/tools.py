import numpy as np
import scipy.stats
from json import JSONEncoder

class Tools():

    @staticmethod
    def mean_confidence_interval(data, confidence=0.95):
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
        return m, m - h, m + h

    @staticmethod
    def matrix_to_vector(data):
        '''
        transform a matrix of cell lists to a matrix of cell values
        :param data:
        :return:
        '''
        new_data = []
        for v in data:
            new_data.append(np.hstack(v))
        return new_data

    @staticmethod
    def compute_raj(throughput):
        n = len(throughput)
        fairness_index = np.power(np.sum(throughput), 2) / (n * np.sum(np.power(throughput, 2)))
        return fairness_index

    @staticmethod
    def moving_average(a, n=3):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


class NumpyArrayEncoderComplex(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, list([obj.real, obj.imag]))