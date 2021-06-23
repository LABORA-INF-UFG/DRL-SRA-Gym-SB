from itertools import combinations
from itertools import permutations

from keras.utils import to_categorical
import numpy as np

class ActionSpace:

    @staticmethod
    def get_action_space_combination(K, F):
        '''
        Return the action space, according to K!/((K-(len(F)*Kmax))! * (Kmax!)^len(F))
        :param K: number of conected UEs
        :param F: list of Kmax per frequency band
        :return: list of possible actions act
        '''
        Kmax = F[0]
        actions_c = list(combinations(list(range(0, K)), Kmax))
        actions_p = list(permutations(actions_c, len(F)))
        act = []

        for action in actions_p:
            if len(list(set(list(np.array(action).flatten())))) == (Kmax * len(F)):
                act.append(action)
        return act


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