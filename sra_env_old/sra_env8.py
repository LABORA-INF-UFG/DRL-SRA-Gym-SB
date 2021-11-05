from typing import Tuple
from itertools import product
import gym
from gym import spaces
import numpy as np

import copy

import consts
from akpy.MassiveMIMOSystem7 import MassiveMIMOSystem
from akpy.buffers_at_BS import Buffers
from schedulers.max_th import MaxTh
from schedulers.proportional_fair import ProportionalFair
from schedulers.round_robin import RoundRobin

from keras.utils import to_categorical
from itertools import combinations

### dealing with combinatorial action space
from utils.action_space import ActionSpace

'''
Dealing with sequential allocation, i.e., with just K * len(F) actions, 
insted of permutations of K arranged by sum(F) times
ex.: action 1: (1, 1) that means UE1->F1 
'''


class SRAEnv(gym.Env):

    def __init__(self, gamma=0, alpha=0, beta=0):
        #np.random.seed(0) # for reproducibility
        self.episode_number = -1
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.__version__ = "0.8.0"
        self.type = None
        self.running_tp = 0 # 0 - trainning; 1 - validating
        self.par_envs = {'Master': [], 'Slave': []}
        self.ep_count = 1
        self.end_ep = False  # Indicate the end of an episode
        self.K = consts.K
        # Array with the number of times of each user was selected (it is cleaned when the episode finish)
        self.c_users = np.zeros(self.K)
        # The number of elements into the array corresponds to the frequencies available and the number of each
        # element corresponds to the max number of users allocated for each frequency
        #self.F = consts.F
        self.F = [2, 2]
        ## getting combinatorial actions
        self.actions = list(product(list(range(self.K)), [0,1]))
        self.action_space = spaces.Discrete(len(self.actions))

        # Number of slots per Block
        self.slots_block = 2
        # Number of blocks per episode
        self.blocks_ep = 1000
        self.curr_slot = 0
        self.curr_block = 0
        self.count_users = 0  # Count number of users allocated in the current frequency
        self.curr_freq = 0  # Frequency being used now
        # Allocated users per frequency, e.g. [[1,2,3],[4,5,6]] which means that user 1, 2 and 3 were
        # allocated in the first frequency and users 4, 5 and 6 were allocated in the second frequency
        self.alloc_users = [[] for i in range(len(self.F))]
        self.prev_alloc_users = [[] for i in range(len(self.F))]
        # Initialization using the number of elements which it will allocate
        #self.observation_space = spaces.Box(low=0,high=1,shape=((self.K)),dtype=np.float32)
        #self.observation_space = spaces.Box(low=0,high=1,shape=self.c_users.shape,dtype=np.float32)
        self.max_spectral_eff = 7.8  # from dump_SE_values_on_stdout()
        # self.recent_spectral_eff = (self.max_spectral_eff / 2) * np.ones((self.K, len(self.F)))  # in bps/Hz
        self.recent_spectral_eff = (self.max_spectral_eff / len(self.F)) * np.ones((self.K, len(self.F)))  # in bps/Hz
        self.spectral_eff = (self.max_spectral_eff / len(self.F)) * np.ones((self.K, len(self.F)))  # in bps/Hz
        self.rates_pkt_per_s = np.array([])
        self.min_reward = consts.MIN_REWARD
        self.max_reward = consts.MAX_REWARD
        self.penalty = -10
        self.rws = []
        self.rws_hist = []

        # Buffer variables
        self.packet_size_bits = consts.PACKET_SIZE_BITS  # if 1024, the number of packets per bit coincides with kbps
        self.buffer_size = consts.BUFFER_SIZE # size in bits obtained by multiplying by self.packet_size_bits
        self.max_packet_age = 1000  # limits the packet age. Maximum value
        self.buffers = Buffers(num_buffers=self.K, buffer_size=self.buffer_size, max_packet_age=self.max_packet_age)
        self.buffers.reset_buffers()  # Initializing buffers

        # individual rates history
        self.rates_history = []
        self.rates = [self.buffer_size for i in range(self.K)]
        # initialization to avoid error in the first run
        self.rates_history.append([self.buffer_size for i in range(self.K)])

        # Massive MIMO System
        self.mimo_systems = self.makeMIMO(self.F, self.K)
        self.mimo_systems = self.loadEpMIMO(self.mimo_systems, self.F, tp=0)

        self.history_buffers = []
        self.rates = [1.] * self.K

        self.schedulers = []
        # creating the Round Robin scheduler instance with independent buffers
        #self.schedulers.append(RoundRobin(K=self.K, F=self.F, buffers=copy.deepcopy(self.buffers)))
        # another scheduler instance, for testing with multiple schedulers
        #self.schedulers.append(ProportionalFair(K=self.K, F=self.F, buffers=copy.deepcopy(self.buffers)))
        #self.schedulers.append(MaxTh(K=self.K, F=self.F, buffers=copy.deepcopy(self.buffers)))

        obs = self.reset()
        self.observation_space = spaces.Box(low=0,high=1,shape=obs.shape,dtype=np.float32)
        #self.observation_space = spaces.Box(low=0, high=1, shape=(30,), dtype=np.float32)

    def step(self,action_index):
        action = self.actions[action_index]
        # estimate individual rates
        self.rates = self.compute_rates()
        # estimate individual SE per frequency
        #self.estimate_SE_all()

        for act in action:
            self.alloc_users[self.curr_freq] = act
            self.curr_freq += 1

        self.rates_pkt_per_s = self.rateEstimationUsers(self.F, self.alloc_users, self.mimo_systems, self.K,
                                                        self.curr_block, self.packet_size_bits)

        # Updating SE per user selected
        self.recent_spectral_eff = self.updateSEUsers(self.F, self.alloc_users, self.mimo_systems,
                                                      self.recent_spectral_eff, self.curr_block)

        reward, _, _ = self.rewardCalc(self.F, self.alloc_users, self.mimo_systems, self.K, self.curr_block,
                        self.packet_size_bits, self.rates_pkt_per_s, self.buffers,
                        self.min_reward, self.recent_spectral_eff)
        self.rws.append(reward)

        # incrementing slot counter
        self.curr_block += 1

        # resetting the allocated users
        self.alloc_users = [[] for i in range(len(self.F))]

        self.mimo_systems = self.updateMIMO(self.mimo_systems, self.curr_block)  # Update MIMO conditions

        # resetting
        self.curr_freq = 0
        #self.curr_slot = 0


        if (self.curr_block == self.blocks_ep):  # Episode has ended
            #self.reset() # aparently will be reseted by gym controller
            self.end_ep = True
        else:
            self.end_ep = False

        # updating observation space and reward
        self.observation_space = self.updateObsSpace(self.buffers, self.buffer_size, self.recent_spectral_eff,
                                                     self.max_spectral_eff, self.max_packet_age)

        info = {}

        return self.observation_space, reward, self.end_ep, info

    def step_(self, action_index):
        reward_schedulers = []
        reward = 0
        pkt_loss = [[] for i in range(len(self.schedulers) + 1)]
        pkt_delay = [[] for i in range(len(self.schedulers) + 1)]
        self.compute_rate()  # computing the rates per user
        ## allocation
        action = self.actions[action_index]
        self.estimate_SE_all()
        self.recent_spectral_eff = self.spectral_eff

        for act in action:
            self.alloc_users[self.curr_freq] = act
            self.curr_freq += 1
        self.curr_freq = len(self.F) - 1

        ## UEs allocations
        if (self.curr_freq == (len(self.F) - 1)) and \
                (len(self.alloc_users[self.curr_freq]) == self.F[self.curr_freq]):
            # incrementing slot counter
            self.curr_block += 1

            if self.type == "Master":

                ## baseline schedulers allocation
                for sc in self.schedulers:
                    if sc.name == 'Proportional Fair' or sc.name == 'Max th':
                        curr_f = 0
                        sc.exp_thr = self.rates_history[-2]
                        ues = sc.policy_action()
                        for u in ues:
                            sc.alloc_users[curr_f].append(u)
                            if len(sc.alloc_users[curr_f]) == sc.F[curr_f]:
                                curr_f += 1
                    elif sc.name == 'Round Robin':
                        for ci, cf in enumerate(self.F):
                            for af in range(cf):
                                ai = sc.policy_action()
                                sc.alloc_users[ci].append(ai)
                ####################################################
                ## schedulers - computing the reward
                for id, sc in enumerate(self.schedulers):
                    rates_pkt_per_s_schedulers = self.rateEstimationUsers(self.F, sc.alloc_users,
                                                                          self.mimo_systems, self.K,
                                                                          self.curr_block, self.packet_size_bits)
                    # saving the recent rate
                    # for i, rt in enumerate(self.rates):
                    #    self.rates[i] = rates_pkt_per_s_schedulers[i]

                    if sc.name == 'Proportional Fair':
                        for i, v in enumerate(rates_pkt_per_s_schedulers):
                            sc.thr_count[i].append(v)
                    # computing the rewards for each scheduler
                    rws, dpkts = self.rewardCalcSchedulers(self.mimo_systems, rates_pkt_per_s_schedulers, sc.buffers)
                    reward_schedulers.append(rws)
                    # pkt_loss[id + 1].append(dpkts)
                    pkt_loss[id + 1] = -10.
                    pkt_delay[id + 1].append(self.compute_pkt_delay(sc.buffers))
                    # clearing alloc users
                    sc.clear()
                ####################################################

            ## Computing reward for DRL-Agent
            self.rates_pkt_per_s = self.rateEstimationUsers(self.F, self.alloc_users, self.mimo_systems, self.K,
                                                            self.curr_block, self.packet_size_bits)

            # Updating SE per user selected
            self.recent_spectral_eff = self.updateSEUsers(self.F, self.alloc_users, self.mimo_systems,
                                                          self.recent_spectral_eff, self.curr_block)
            #self.spectral_eff = self.recent_spectral_eff

            reward, dropped_pkt, dropped_pkts_percent_mean = self.rewardCalc_(self.F, self.alloc_users,
                                                                              self.mimo_systems, self.K, self.curr_slot,
                                                                              self.packet_size_bits,
                                                                              [self.rates_pkt_per_s, self.rates],
                                                                              self.buffers,
                                                                              self.min_reward, self.recent_spectral_eff,
                                                                              update=True)

            # pkt_loss[0].append(dropped_pkts_percent_mean)
            # returning a fake packet loss. The real loss will be calculated at the reset
            pkt_loss[0] = -10.

            pkt_delay[0].append(self.compute_pkt_delay(self.buffers))

            # resetting the allocated users
            self.alloc_users = [[] for i in range(len(self.F))]

            self.mimo_systems = self.updateMIMO(self.mimo_systems, self.curr_block)  # Update MIMO conditions

            # updating observation space and reward
            self.observation_space = self.updateObsSpace(self.buffers, self.buffer_size, self.recent_spectral_eff,
                                                         self.max_spectral_eff, self.max_packet_age)

            # resetting
            self.curr_freq = 0
            self.curr_slot = 0

        if len(self.alloc_users[self.curr_freq]) == self.F[self.curr_freq]:
            self.curr_freq += 1
            self.curr_slot += 1

        if (self.curr_block == self.blocks_ep):  # Episode has ended
            # before reset, compute the episode loss
            loss = np.sum(self.buffers.buffer_history_dropped) / np.sum(self.buffers.buffer_history_incoming)
            pkt_loss[0] = loss  # changing the fake value -10 by the real loss value
            ## schedulers - computing the reward
            for id, sc in enumerate(self.schedulers):
                loss = np.sum(sc.buffers.buffer_history_dropped) / np.sum(sc.buffers.buffer_history_incoming)
                pkt_loss[id + 1] = loss  # changing the fake value -10 by the real loss value
            self.reset()
            self.ep_count += 1

        info = {}
        # print("Episode " + str(self.ep_count) + " - Block " + str(self.curr_block) + " RW " + str(reward))
        return self.observation_space, [reward, reward_schedulers, pkt_loss, pkt_delay], self.end_ep, info


    def rewardCalc(self, F: list, alloc_users: list, mimo_systems: list, K: int, curr_slot: int, packet_size_bits: int,
                   pkt_rate, buffers: Buffers, min_reward: int, SE: list) -> float:  # Calculating reward value

        buffer_states = buffers.get_buffer_states()
        occup = np.sum(buffers.buffer_occupancies)

        # Oldest packets
        oldest_packet_per_buffer = buffer_states[1]
        # All values above threshold are set to the maximum value allowed
        oldest_packet_per_buffer[oldest_packet_per_buffer > self.max_packet_age] = self.max_packet_age
        # Normalization
        #oldest_packet_per_buffer = oldest_packet_per_buffer / self.max_packet_age
        (tx_pkts, dropped_pkts_sum, dropped_pkts, t_pkts, incoming_pkts, buffers, dropped_pkts_percent_mean) = self.pktFlow(pkt_rate,
                                                                                                                            buffers,
                                                                                                                            mimo_systems)
        ## funcionando para o novo dataset -> gamma = 10, alpha = 100 e beta = 0 (MT) vs beta = 100 (PF)
        #reward = (self.gamma * (tx_pkts*self.packet_size_bits)/1e6) - (self.alpha * (dropped_pkts_sum*self.packet_size_bits)/1e6)
        #reward -= self.beta * np.sum(oldest_packet_per_buffer)
        #print("Rw= " + str(reward) + ' -- tx ' + str(tx_pkts) + ' -- loss ' + str(dropped_pkts_sum) + ' -- oldest sum'
        #      + str(np.sum(oldest_packet_per_buffer)))
        #print("rate " + str((tx_pkts*self.packet_size_bits)/1e6))
        #print("loss " + str((dropped_pkts_sum * self.packet_size_bits) / 1e6))

        reward = (self.gamma * ((tx_pkts / occup) * 100)) - (self.alpha * ((dropped_pkts_sum / tx_pkts) * 100)) - (
                    self.beta * (np.mean(oldest_packet_per_buffer)))

        reward = (self.gamma * (tx_pkts * self.packet_size_bits) / 1e6) - (self.alpha * (dropped_pkts_sum * self.packet_size_bits)/ 1e6)
        reward = (self.gamma * ((tx_pkts / occup) * 100))
        #reward = (self.gamma * (tx_pkts * self.packet_size_bits) / 1e6)

        return reward, dropped_pkts_sum, dropped_pkts_percent_mean

    def rewardCalc_(self, F: list, alloc_users: list, mimo_systems: list, K: int, curr_slot: int, packet_size_bits: int,
                   pkt_rate, buffers: Buffers, min_reward: int, SE: list, update) -> float:  # Calculating reward value

        (tx_pkts, dropped_pkts_sum, dropped_pkts, t_pkts, incoming_pkts, buffers,
         dropped_pkts_percent_mean) = self.pktFlow(pkt_rate[0],
                                                   buffers,
                                                   mimo_systems)
        reward = (tx_pkts * self.packet_size_bits) / 1e6


        return reward, dropped_pkts_sum, dropped_pkts_percent_mean

    def rateEstimationUsersAll(self, F: list, alloc_users: list, mimo_systems: list, K: int, curr_slot: int,
                               packet_size_bits: int) -> list:  # Calculates the rate per second for each user considering if it was selected to transmit in any frequency and
        rates_pkt_per_s = np.zeros((K,))  # Considering rates are per second

        for f in range(len(F)):
            for au in alloc_users[f]:
                se_freq = mimo_systems[f].SE_current_sample(curr_slot, [au])
                rates = (se_freq[0] * mimo_systems[f].BW[f]) / packet_size_bits
                rates_pkt_per_s[au] = rates
        return rates_pkt_per_s

    def rateEstimationUsers(self, F: list, alloc_users: list, mimo_systems: list, K: int, curr_slot: int,
                            packet_size_bits: int) -> list:  # Calculates the rate per second for each user considering if it was selected to transmit in any frequency and
        rates_pkt_per_s = np.zeros((K,))  # Considering rates are per second

        for f in range(len(F)):
            se_freq = mimo_systems[f].SE_current_sample(curr_slot, alloc_users[f])
            #se_freq = mimo_systems[f].SE_for_given_sample(curr_slot, alloc_users[f], F[f],
            #                                              avoid_estimation_errors=False)
            for iu, u in enumerate(alloc_users[f]):
                rates_pkt_per_s[u] += (se_freq[iu] * mimo_systems[f].BW[f]) / packet_size_bits

        return rates_pkt_per_s

    def updateSEUsers(self, F: list, alloc_users: list, mimo_systems: list, recent_spectral_eff: np.ndarray,
                      curr_slot: int):  # Function which update the Spectral efficiency value for each UE used in the last block
        for f in range(len(F)):
            SE = mimo_systems[f].SE_current_sample(curr_slot, alloc_users[f])
            for iu, u in enumerate(alloc_users[f]):
                recent_spectral_eff[u, f] = SE[iu]
        return recent_spectral_eff

    def estimate_SE_all(self):
        users = list(range(self.K))
        for f in range(len(self.F)):
            SE = self.mimo_systems[f].SE_current_sample(self.curr_block, users)
            for iu, u in enumerate(users):
                self.spectral_eff[u, f] = SE[iu]


    def updateMIMO(self, mimo_systems: list,
                   curr_slot: int) -> list:  # Updating MIMO environment to the next slot, recalculating interferences
        for mimo in mimo_systems:
            mimo.current_sample_index = curr_slot
            mimo.update_intercell_interference()

        return mimo_systems


    def updateObsSpace(self, buffers: Buffers, buffer_size: float, recent_spectral_eff: list, max_spectral_eff: float,
                       max_packet_age: int) -> np.ndarray:  # Update the observation space which is composed of buffer occupancy, oldest packets per buffer, spectral efficiency and day time(normalizing all them)
        buffer_states = buffers.get_buffer_states()

        # Buffer occupancy
        buffer_occupancies = buffer_states[0] / buffer_size
        buffer_occupancies = np.array(buffer_occupancies*100).astype(int)

        # Oldest packets
        oldest_packet_per_buffer = buffer_states[1]
        oldest_packet_per_buffer[
            oldest_packet_per_buffer > max_packet_age] = max_packet_age  # All values above threshold are set to the maximum value allowed
        oldest_packet_per_buffer = oldest_packet_per_buffer / np.max(oldest_packet_per_buffer)  # Normalization

        # Spectral efficiency
        #spectral_eff = recent_spectral_eff / max_spectral_eff

        # new spectral eff
        # estimate individual SE per frequency
        self.estimate_SE_all()
        #spectral_eff = self.spectral_eff / np.max(self.spectral_eff)
        spectral_eff = np.array(self.spectral_eff).astype(int)
        #spectral_eff = self.spectral_eff
        # original
        # return np.hstack((buffer_occupancies, oldest_packet_per_buffer, spectral_eff.flatten()))
        # simplificando o estado
        max_rate = np.max(self.rates)
        p_rates = self.rates / max_rate
        thr = np.minimum(self.rates, self.buffers.buffer_occupancies)
        max_rate = np.max(thr)
        p_rates = thr / self.buffer_size

        #return np.hstack((buffer_occupancies, spectral_eff.flatten(), oldest_packet_per_buffer))
        return np.hstack((buffer_occupancies, spectral_eff.flatten()))

    # calculation of packets transmission and reception (from transmit_and_receive_new_packets function)
    def pktFlowNoUpdate(self, pkt_rate: float, buffers: Buffers, mimo_systems: list) -> (
            float, float, float, float, float, list):
        available_rate = np.floor(pkt_rate).astype(int)
        a_buffer = copy.deepcopy(self.buffers)
        t_pkts = available_rate if (np.sum(available_rate) == 0) else self.buffers.packets_departure(
            available_rate)  # transmission pkts
        dropped_pkts = buffers.get_dropped_last_iteration()
        dropped_pkts_sum = np.sum(dropped_pkts)
        incoming_pkts = mimo_systems[0].get_current_income_packets()
        # buffers.packets_arrival(incoming_pkts)  # Updating Buffer
        tx_pkts = np.sum(t_pkts)
        self.buffers = copy.deepcopy(a_buffer)
        return (tx_pkts, dropped_pkts_sum, dropped_pkts, t_pkts, incoming_pkts, buffers)


    # calculation of packets transmission and reception (from transmit_and_receive_new_packets function)
    def pktFlow(self, pkt_rate: float, buffers: Buffers, mimo_systems: list) -> (
                float, float, float, float, float, list):
        present_flow = buffers.buffer_occupancies
        available_rate = np.floor(pkt_rate).astype(int)
        t_pkts = available_rate if (np.sum(available_rate) == 0) else buffers.packets_departure(
            available_rate)  # transmission pkts
        dropped_pkts = buffers.get_dropped_last_iteration()
        dropped_pkts_sum = np.sum(dropped_pkts)
        # getting the percentual dropped of the incoming packages
        dropped_pkts_percent = buffers.get_dropped_last_iteration_percent()
        dropped_pkts_percent_mean = np.mean(dropped_pkts_percent)
        incoming_pkts = mimo_systems[0].get_current_income_packets()
        buffers.packets_arrival(incoming_pkts)  # Updating Buffer
        tx_pkts = np.sum(t_pkts)

        return (tx_pkts, dropped_pkts_sum, dropped_pkts, t_pkts, incoming_pkts, buffers, dropped_pkts_percent_mean)

    def makeMIMO(self, F: list, K: int) -> list:  # Generating MIMO system for each user
        return [MassiveMIMOSystem(K=K, frequency_index=f + 1) for f in range(len(F))]

    def loadEpMIMO(self, mimo_systems: list, F: list, tp: int) -> list:
        '''
        Load static channel data file. tp difine the running type: 0 - training; 1- validating
        '''
        # calculate the matrices to generate channel estimates
        #episode_number = np.random.randint(0, self.mimo_systems[0].num_episodes - 1)
        if tp == 0:
            episode_number = np.random.randint(self.mimo_systems[0].range_episodes_train[0],
                                               self.mimo_systems[0].range_episodes_train[1])
        else:
            episode_number = np.random.randint(self.mimo_systems[0].range_episodes_validate[0],
                                               self.mimo_systems[0].range_episodes_validate[1])
        self.episode_number = episode_number
        for f in range(len(F)):
            # same episode number for all frequencies
            self.mimo_systems[f].load_episode(episode_number)
            self.mimo_systems[f].update_intercell_interference()  # avoid interference=0 in beginning of episode
        return mimo_systems

    def reset(self):  # Required by script to initialize the observation space
        if self.running_tp == 0:
            if len(self.rws) > 0:
                print("Mean RW " + str(np.mean(self.rws)))
                self.rws_hist.append(np.mean(self.rws))
                self.rws = []
        self.curr_block = 0
        self.end_ep = True
        # Cleaning count of users
        self.c_users = np.zeros(self.K)

        if self.type != "Slave":
            # Buffer
            self.buffers.reset_buffers()  # Reseting buffers
            # get some traffic to avoid starting from empty buffers
            self.buffers.packets_arrival(self.mimo_systems[0].get_current_income_packets())

            # MIMO Interference
            self.mimo_systems = self.loadEpMIMO(self.mimo_systems, self.F, tp=self.running_tp)

            # reseting the schedulers
            for sc in self.schedulers:
                sc.reset()
                sc.buffers = copy.deepcopy(self.buffers)

        else:
            master_env = self.par_envs['Master'][0]
            self.mimo_systems = copy.deepcopy(master_env.mimo_systems)
            self.buffers = copy.deepcopy(master_env.buffers)



        self.observation_space = self.updateObsSpace(self.buffers, self.buffer_size, self.recent_spectral_eff,
                                                     self.max_spectral_eff, self.max_packet_age)
        self.ep_count += 1

        return self.observation_space

    def reset_pre(self):  # Required by script to initialize the observation space
        '''
        simplified version to only reset the buffers
        '''
        self.rws = []
        self.curr_block = 0
        #self.end_ep = True
        # Cleaning count of users
        self.c_users = np.zeros(self.K)

        if self.type != "Slave":
            # Buffer
            self.buffers.reset_buffers()  # Reseting buffers
            # get some traffic to avoid starting from empty buffers
            self.buffers.packets_arrival(self.mimo_systems[0].get_current_income_packets())

            # reseting the schedulers
            for sc in self.schedulers:
                sc.reset()
                sc.buffers = copy.deepcopy(self.buffers)

        else:
            master_env = self.par_envs['Master'][0]
            self.mimo_systems = copy.deepcopy(master_env.mimo_systems)
            self.buffers = copy.deepcopy(master_env.buffers)



        self.observation_space = self.updateObsSpace(self.buffers, self.buffer_size, self.recent_spectral_eff,
                                                     self.max_spectral_eff, self.max_packet_age)
        self.ep_count += 1

        return self.observation_space

    def compute_rate(self):
        # rate estimation for all users
        users = list(range(self.K))
        middle_index = len(users) // 2

        alloc = [users[:middle_index], users[middle_index:]]
        self.rates = self.rateEstimationUsers(self.F, alloc, self.mimo_systems, self.K,
                                              self.curr_block, self.packet_size_bits)

        self.rates_history.append(np.array(self.rates))

    def close(self):
        pass

    def render(self):
        pass

    def compute_rates(self):
        # rate estimation for all users
        # considering allocation at first frequency (with less BW)
        rates = []
        for i in range(self.K):
            r = self.rateEstimationUsersAll([self.F[0]], [[i]], self.mimo_systems, self.K,
                                            self.curr_slot, self.packet_size_bits)
            rates.append(r[i])
        return rates


    def rewardCalcSchedulers(self, mimo_systems: list, pkt_rate, buffers: Buffers) -> float:  # Calculating reward value

        (tx_pkts, dropped_pkts_sum, dropped_pkts, t_pkts, incoming_pkts, buffers,
         dropped_pkts_percent_mean) = self.pktFlow(pkt_rate, buffers, mimo_systems)

        reward = 10 * (tx_pkts / 50000)

        return reward, dropped_pkts_percent_mean

    def compute_pkt_delay(self, buffers):
        buffer_states = buffers.get_buffer_states()
        # Oldest packets
        oldest_packet_per_buffer = buffer_states[1]
        return oldest_packet_per_buffer

    def set_seed(self, seed):
        np.random.seed(seed)  # for reproducibility
