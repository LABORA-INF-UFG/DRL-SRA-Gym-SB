from itertools import combinations
from itertools import permutations

from keras.utils import to_categorical
import numpy as np

class ActionSpace:

    @staticmethod
    def get_action_space(K, F):
        #actions_p = list(combinations(list(range(0, K)), np.sum(F)))
        actions_p = list(permutations(list(range(0, K)), int(np.sum(F))))

        actions = []
        for act in actions_p:
            pa = np.array_split(act,len(F))
            actions.append(pa)
        return actions



    '''
    Generate categorical combinatorial action space
    '''
    @staticmethod
    def get_action_space_cat(K,F):
        actions_p = to_categorical(list(combinations(list(range(0, K)), np.sum(F))), K)
        actions = []

        for act in actions_p:
            sum_act = np.zeros(K)
            for a in act:
                sum_act += a
            actions.append(sum_act)

        return actions