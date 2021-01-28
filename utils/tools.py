import numpy as np
import scipy.stats

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
