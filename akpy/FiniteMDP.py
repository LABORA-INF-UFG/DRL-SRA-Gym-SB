'''
This is the old version
'''
from __future__ import print_function
import numpy as np
from builtins import print
# from scipy.stats import rv_discrete
from random import choices
import random
from bidict import bidict

#AK-TODO: nextStateProbability is known only by oracle and should not be requested here.
#nextStateProbability should be part of a "optimum" subclass that extends this one

#AK-TODO use a python bijective map such as bidict
#https://stackoverflow.com/questions/1456373/two-way-reverse-map
#https://pypi.org/project/bidict/
#https://stackoverflow.com/questions/3318625/how-to-implement-an-efficient-bidirectional-hash-table/21894086#21894086
#here we can have 2 keys having same value, this is very important. Is it important for me?
class FiniteMDP:
    def __init__(self, discount=0.9):
        self.__version__ = "0.1.0"
        # print("AK Finite MDP - Version {}".format(self.__version__))
        self.discountGamma = discount

        #they will be created by method self.createFiniteMDP()
        self.environment = None
        self.actionDictionaryGetIndex = None
        self.actionListGivenIndex = None
        self.stateDictionaryGetIndex = None
        self.stateListGivenIndex = None
        self.rewardDictionaryGetIndex = None
        self.rewardListGivenIndex = None

        #now they are part of environment:
        #self.nextStateProbability = None
        #self.rewardsTable = None

        self.currentObservation = 0

        self.createFiniteMDP()
        #(S, A, nS) = self.nextStateProbability.shape #should not require nextStateProbability, which is often unknown
        self.S = len(self.stateDictionaryGetIndex)  # number of states
        self.A = len(self.actionDictionaryGetIndex)  # number of actions

        # create random number generators for all next state probabilities
        # From https://stackoverflow.com/questions/11373192/generating-discrete-random-variables-with-specified-weights-using-scipy-or-numpy
        # could use numpy.random.choice or scipy.stats.rv_discrete
        # I will stick with the latter
        # elements = np.array([3.1, -4.2, 6])
        # probabilities = [0.2, 0.5, 0.3]
        # print('A', np.random.choice(elements, 20, p=probabilities))
        # distrib = rv_discrete(values=(range(len(elements)), probabilities))  # This defines a Scipy probability distribution
        # ii=distrib.rvs(size=(3,20))  # samples from range(len(values)) with indices
        # print('B', elements[ii])  # Conversion to specific discrete values (the fact that values is a NumPy array is used for the indexing)
        # AK-TODO did not work
        # self.getRandomNextState = []
        # for s in range(S):
        #    self.getRandomNextState.append([])
        #    for a in range(A):
        #        distrib = rv_discrete(range(S),self.nextStateProbability[s,a])
        #        self.getRandomNextState[s].append(distrib)
        # p = self.getRandomNextState[s][action].rvs(size=1)

        self.currentIteration = 0
        self.environment.reset()

    def createFiniteMDP(self):
        raise NotImplementedError('This works as an abstract class. You need to implement a subclass!')

    def prettyPrint(self):
        '''Print MDP'''
        nextStateProbability = self.environment.nextStateProbability
        rewardsTable = self.environment.rewardsTable
        stateListGivenIndex = self.stateListGivenIndex
        actionListGivenIndex = self.actionListGivenIndex
        S = len(stateListGivenIndex)
        A = len(actionListGivenIndex)
        for s in range(S):
            currentState = stateListGivenIndex[s]
            print('current state s=', s, '=', currentState, sep='')  # ' ',end='')
            for a in range(A):
                currentAction = actionListGivenIndex[a]
                print('  action a', a, '=', currentAction)
                for nexts in range(S):
                    if nextStateProbability[s, a, nexts] == 0:
                        continue
                        #nonZeroIndices = np.where(nextStateProbability[s, a] > 0)[0]
                # if len(nonZeroIndices) != 2:
                #    print('Error in logic, not 2: ', len(nonZeroIndices))
                #    exit(-1)
                    print('    nexts', nexts, '=',
                      stateListGivenIndex[nexts],
                      ' p=', nextStateProbability[s, a, nexts],
                      ' r=', rewardsTable[s, a, nexts], sep='')

    def prettyPrintPolicy(self, action_values):
        for s in range(self.S):
            currentState = self.stateListGivenIndex[s]
            print('\ns=', s, '=', currentState)  # ' ',end='')
            for a in range(self.A):
                if action_values[s, a] == 0:
                    continue
                currentAction = self.actionListGivenIndex[a]
                print(' | a=', a, '=', currentAction, end='')

    def getEquiprobableRandomPolicy(self):
        policy = np.zeros((self.S, self.A))
        uniformProbability = 1.0 / self.A
        for s in range(self.S):
            for a in range(self.A):
                policy[s, a] = uniformProbability
        return policy

    def compute_state_values(self, policy, in_place=False):
        '''Iterative policy evaluation. Page 76 of [Sutton, 2018]'''
        (S, A, nS) = self.environment.nextStateProbability.shape
        # A = len(actionListGivenIndex)
        new_state_values = np.zeros((S,))
        state_values = new_state_values.copy()
        iteration = 1
        while True:
            src = new_state_values if in_place else state_values
            for s in range(S):
                value = 0
                for a in range(A):
                    if policy[s, a] == 0:
                        continue  # save computation
                    for nexts in range(S):
                        p = self.environment.nextStateProbability[s, a, nexts]
                        r = self.environment.rewardsTable[s, a, nexts]
                        value += policy[s, a] * p * (r + self.discountGamma * src[nexts])
                        # value += p*(r+discount*src[nexts])
                new_state_values[s] = value
            improvement = np.sum(np.abs(new_state_values - state_values))
            # print('improvement =', improvement)
            if False:  # debug
                print('state values=', state_values)
                print('new state values=', new_state_values)
                print('it=', iteration, 'improvement = ', improvement)
            if improvement < 1e-4:
                state_values = new_state_values.copy()
                break

            state_values = new_state_values.copy()
            iteration += 1

        return state_values, iteration

    def compute_optimal_state_values(self):
        '''Page 63 of [Sutton, 2018], Eq. (3.19)'''
        (S, A, nS) = self.environment.nextStateProbability.shape
        # A = len(actionListGivenIndex)
        new_state_values = np.zeros((S,))
        state_values = new_state_values.copy()
        iteration = 1
        while True:
            for s in range(S):
                a_candidates = list()
                for a in range(A):
                    value = 0
                    for nexts in range(S):
                        p = self.environment.nextStateProbability[s, a, nexts]
                        r = self.environment.rewardsTable[s, a, nexts]
                        value += p * (r + self.discountGamma * state_values[nexts])
                    a_candidates.append(value)
                new_state_values[s] = np.max(a_candidates)
            improvement = np.sum(np.abs(new_state_values - state_values))
            # print('improvement =', improvement)
            if False:  # debug
                print('state values=', state_values)
                print('new state values=', new_state_values)
                print('it=', iteration, 'improvement = ', improvement)

            state_values = new_state_values.copy()
            if improvement < 1e-4:
                break

            iteration += 1

        return state_values, iteration

    def compute_optimal_action_values(self):
        '''Page 64 of [Sutton, 2018], Eq. (3.20)'''
        (S, A, nS) = self.environment.nextStateProbability.shape
        # A = len(actionListGivenIndex)
        new_action_values = np.zeros((S, A))
        action_values = new_action_values.copy()
        iteration = 1
        while True:
            # src = new_action_values if in_place else action_values
            for s in range(S):
                for a in range(A):
                    value = 0
                    for nexts in range(S):
                        p = self.environment.nextStateProbability[s, a, nexts]
                        r = self.environment.rewardsTable[s, a, nexts]
                        # print('amor', r, p, src[nexts])
                        best_a = -np.Infinity
                        for nexta in range(A):
                            temp = action_values[nexts, nexta]
                            if temp > best_a:
                                best_a = temp
                        value += p * (r + self.discountGamma * best_a)
                        # value += p*(r+discount*src[nexts])
                        # print('aa', value)
                    new_action_values[s, a] = value
            improvement = np.sum(np.abs(new_action_values - action_values))
            # print('improvement =', improvement)
            if False:  # debug
                print('state values=', action_values)
                print('new state values=', new_action_values)
                print('it=', iteration, 'improvement = ', improvement)
            if improvement < 1e-4:
                action_values = new_action_values.copy()
                break

            action_values = new_action_values.copy()
            iteration += 1

        return action_values, iteration

    def createStatesDataStructures(self, possibleStates):
        '''Invoke default method. User can customize overriding it.'''
        return self.defaultCreationOfDataStructures(possibleStates)

    def createActionsDataStructures(self, possibleActions):
        '''Invoke default method. User can customize overriding it.'''
        return self.defaultCreationOfDataStructures(possibleActions)

    def createRewardsDataStructures(self, possibleRewards):
        '''Invoke default method. User can customize overriding it.'''
        return self.defaultCreationOfDataStructures(possibleRewards)

    def defaultCreationOfDataStructures(self, possibleTuples):
        '''Simply organize list and dict, nothing special'''
        dictionaryGetIndex = dict()
        listGivenIndex = list()
        for uniqueIndex in range(len(possibleTuples)):
            dictionaryGetIndex[possibleTuples[uniqueIndex]] = uniqueIndex
            listGivenIndex.append(possibleTuples[uniqueIndex])
        return dictionaryGetIndex, listGivenIndex

    def convert_action_values_into_policy(self, action_values):
        (S, A) = action_values.shape
        policy = np.zeros((S, A))
        for s in range(S):
            maxPerState = max(action_values[s])
            maxIndices = np.where(action_values[s] == maxPerState)
            # maxIndices is a tuple and we want to get first element maxIndices[0]
            policy[s, maxIndices] = 1.0 / len(maxIndices[0])  # impose uniform distribution
        return policy

    def postprocessing_MDP_step(self, history, printPostProcessingInfo):
        '''This method can be overriden by subclass and process history'''
        pass  # no need to do anything here

    def run_MDP_for_given_policy(self, policy, maxNumIterations=100, printInfo=False, printPostProcessingInfo=False):
        self.environment.reset()
        s = self.environment.get_state()
        totalReward = 0
        if printInfo:
            print('Initial state = ', self.stateListGivenIndex[s])
        for it in range(maxNumIterations):
            myweights = np.squeeze(policy[s])
            sumWeights = np.sum(myweights) #AK-TODO: what if there are positive and negative numbers canceling out?
            if sumWeights == 0:
                myweights = np.ones(myweights.shape)
            if True:
                #AK-TODO the choices is giving troubles with negative weights. Make them all positive
                #AK-TODO this changes the relative importances / weights, right?
                minWeight = np.min(myweights)
                if minWeight < 0:
                    myweights += (-minWeight)+1e-30
                sumWeights = np.sum(myweights)
                myweights /= sumWeights
            action = choices(np.arange(self.A), weights=myweights, k=1)[0]
            ob, reward, gameOver, history = self.environment.step(action)

            #AK
            if reward < -5:
                print(history)

            self.postprocessing_MDP_step(history, printPostProcessingInfo)
            if printInfo:
                print(history)
            totalReward += reward
            s = self.environment.get_state()
            if gameOver == True:
                break
        if printInfo:
            print('totalReward = ', totalReward)

    # an episode with Q-Learning
    # @stateActionValues: values for state action pair, will be updated
    # @expected: if True, will use expected Sarsa algorithm
    # @stepSize: step size for updating
    # @return: total rewards within this episode
    def q_learning(self, stateActionValues, maxNumIterationsQLearning=100, stepSizeAlpha=0.1,
                   explorationProbEpsilon=0.01):
        currentState = self.environment.get_state()
        rewards = 0.0
        for numIterations in range(maxNumIterationsQLearning):
            currentAction = self.chooseAction(currentState, stateActionValues, explorationProbEpsilon)

            ob, reward, gameOver, history = self.environment.step(currentAction)
            newState = self.environment.get_state()
            reward = self.environment.rewardsTable[currentState, currentAction, newState]
            rewards += reward
            # Q-Learning update
            stateActionValues[currentState, currentAction] += stepSizeAlpha * (
                    reward + self.discountGamma * np.max(stateActionValues[newState, :]) -
                    stateActionValues[currentState, currentAction])
            currentState = newState
        # normalize rewards to facilitate comparison
        return rewards / maxNumIterationsQLearning

    # choose an action based on epsilon greedy algorithm
    def chooseAction(self, state, stateActionValues, explorationProbEpsilon=0.01):
        if np.random.binomial(1, explorationProbEpsilon) == 1:
            return np.random.choice(np.arange(self.A))
        else:
            values_ = stateActionValues[state, :]
            return np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])

    def execute_q_learning(self, maxNumIterations=100, maxNumIterationsQLearning=10, num_runs=1,
                           stepSizeAlpha=0.1, explorationProbEpsilon=0.01):
        '''Use independent runs instead of a single run.
        maxNumIterationsQLearning is used to smooth numbers'''

        rewardsQLearning = np.zeros(maxNumIterations)
        for run in range(num_runs):
            stateActionValues = np.zeros((self.S, self.A))
            for i in range(maxNumIterations):
                # update stateActionValues in-place
                reward = self.q_learning(stateActionValues,
                                         maxNumIterationsQLearning=maxNumIterationsQLearning,
                                         stepSizeAlpha=stepSizeAlpha,
                                         explorationProbEpsilon=explorationProbEpsilon)
                rewardsQLearning[i] += reward
        rewardsQLearning /= num_runs
        if False:
            print('rewardsQLearning = ', rewardsQLearning)
            print('newStateActionValues = ', stateActionValues)
            # qlearning_policy = self.convert_action_values_into_policy(newStateActionValues)
            print('qlearning_policy = ', self.prettyPrintPolicy(stateActionValues))
        return stateActionValues, rewardsQLearning


if __name__ == '__main__':
    mdp = FiniteMDP(discount=0.9)
