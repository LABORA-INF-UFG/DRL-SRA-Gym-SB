import numpy as np
from random import choices, randint

class EnvironmentByNextStateProbs:
    def __init__(self, nextStateProbability, rewardsTable):
        self.__version__ = "0.1.0"
        # print("AK Finite MDP - Version {}".format(self.__version__))

        self.nextStateProbability = nextStateProbability
        self.rewardsTable = rewardsTable

        self.currentObservation = 0

        #(S, A, nS) = self.nextStateProbability.shape #should not require nextStateProbability, which is often unknown
        self.S = nextStateProbability.shape[0]  # number of states
        self.A = nextStateProbability.shape[1]  # number of actions

        self.currentIteration = 0
        self.reset()

    def step(self, action):
        """
        The agent takes a step in the environment.
        Parameters
        ----------
        action : array of topN integers
        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
            reward (float) :
            episode_over (bool) :
            info (dict) :
        """
        s = self.get_state()

        elements = np.arange(self.S)
        # weights = np.squeeze(self.nextStateProbability[s,action])
        weights = self.nextStateProbability[s, action]
        nexts = choices(elements, weights, k=1)[0]

        # p = self.nextStateProbability[s,action]
        # reward = self.rewardsTable[s,action, nexts][0]
        reward = self.rewardsTable[s, action, nexts]

        # fully observable MDP: observation is the actual state
        self.currentObservation = nexts

        gameOver = False
        if self.currentIteration > np.Inf:
            ob = self.reset()
            gameOver = True  # game ends
        else:
            ob = self.get_state()

        history = {"time": self.currentIteration, "state_t": s, "action_t": action,
                   "reward_tp1": reward, "state_tp1": nexts}
        # history version with actions and states, not their indices
        # history = {"time": self.currentIteration, "action_t": self.actionListGivenIndex[action],
        #           "reward_tp1": reward, "observation_tp1": self.stateListGivenIndex[self.get_state()]}
        self.currentIteration += 1
        return ob, reward, gameOver, history

    def get_state(self):
        """Get the current observation."""
        return self.currentObservation

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.currentIteration = 0
        # note there are several versions of randint!
        self.currentObservation = randint(0, self.S - 1)
        return self.get_state()

    # from gym.utils import seeding
    # def seed(self, seed=None):
    #    self.np_random, seed = seeding.np_random(seed)
    #    return [seed]

    def numberOfActions(self):
        return self.A

    def numberOfObservations(self):
        # ob = self.get_state()
        # return len(ob)
        return self.S
