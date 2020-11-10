from typing import Tuple

import gym
from gym import spaces
import numpy as np

import consts
from akpy.MassiveMIMOSystem5 import MassiveMIMOSystem
from akpy.buffers_at_BS import Buffers

class SRAEnv(gym.Env):

    def __init__(self):
        self.__version__ = "0.1.0"
        self.ep_count = 1
        self.end_ep = False  # Indicate the end of an episode
        self.K = consts.K
        # Array with the number of times of each user was selected (it is cleaned when the episode finish)
        self.c_users = np.zeros(self.K)
        self.action_space = gym.spaces.Discrete(self.K)
        # The number of elements into the array corresponds to the frequencies available and the number of each
        # element corresponds to the max number of users allocated for each frequency
        self.F = consts.F
        # Number of slots per Block
        self.slots_block = 1
        # Number of blocks per episode
        self.blocks_ep = consts.BLOCKS_EP
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
        self.rates_pkt_per_s = np.array([])
        self.min_reward = consts.MIN_REWARD
        self.max_reward = consts.MAX_REWARD
        self.penalty = -10
        self.rws = []

        # Buffer variables
        self.packet_size_bits = consts.PACKET_SIZE_BITS  # if 1024, the number of packets per bit coincides with kbps
        self.buffer_size = consts.BUFFER_SIZE  # size in bits obtained by multiplying by self.packet_size_bits
        self.max_packet_age = consts.MAX_PACKET_AGE  # limits the packet age. Maximum value
        self.buffers = Buffers(num_buffers=self.K, buffer_size=self.buffer_size, max_packet_age=self.max_packet_age)
        self.buffers.reset_buffers()  # Initializing buffers

        # Massive MIMO System
        self.mimo_systems = self.makeMIMO(self.F, self.K)
        self.mimo_systems = self.loadEpMIMO(self.mimo_systems, self.F)

        self.history_buffers = []
        self.rates = [1.] * self.K

        self.schedulers = []
        obs = self.reset()
        self.observation_space = spaces.Box(low=0,high=1,shape=obs.shape,dtype=np.float32)

    def step(self,action):

        # incrementing slot counter
        self.curr_block += 1
        self.alloc_users = [[] for i in range(len(self.F))]
        self.alloc_users[self.curr_freq].append(action)
        self.rates_pkt_per_s = self.rateEstimationUsers(self.F, self.alloc_users, self.mimo_systems, self.K,
                                                        self.curr_slot, self.packet_size_bits)

        # rate estimation for all users
        rates = []
        for i in range(self.K):
            r = self.rateEstimationUsersAll(self.F, [[i]], self.mimo_systems, self.K,
                                            self.curr_slot, self.packet_size_bits)
            rates.append(r[i])
        self.rates = rates

        # Updating SE per user selected
        self.recent_spectral_eff = self.updateSEUsers(self.F, self.alloc_users, self.mimo_systems,
                                                      self.recent_spectral_eff, self.curr_slot)
        reward, _ = self.rewardCalc(self.alloc_users)

        # resetting the allocated users
        self.alloc_users = [[] for i in range(len(self.F))]

        self.mimo_systems = self.updateMIMO(self.mimo_systems, self.curr_slot)  # Update MIMO conditions

        # updating observation space and reward
        self.observation_space = self.updateObsSpace(self.buffers, self.buffer_size, self.recent_spectral_eff,
                                                     self.max_spectral_eff, self.max_packet_age)

        if (self.curr_block == self.blocks_ep):  # Episode has ended
            #self.reset() # aparently will be reseted by gym controller
            self.end_ep = True
        else:
            self.end_ep = False
            pass
        # Optionally we can pass additional info, we are not using that for now
        info = {}
        print("Episode " + str(self.ep_count) + " - Block " + str(self.curr_block))
        return self.observation_space, reward, self.end_ep, info

    def step_(self,action):

        reward_schedulers = []
        pkt_loss = [[] for i in range(len(self.schedulers) + 1)]
        # dealing with the schedulers
        for id, sc in enumerate(self.schedulers):
            if sc.name == 'Round Robin':
                sc.alloc_users[self.curr_freq].append(sc.policy_action())

            rates_pkt_per_s_schedulers = self.rateEstimationUsers(self.F, sc.alloc_users,
                                                                  self.mimo_systems, self.K,
                                                                  self.curr_slot, self.packet_size_bits)

            # computing the rewards for each scheduler
            rws, dpkts = self.rewardCalcSchedulers(self.mimo_systems, rates_pkt_per_s_schedulers, sc.buffers)
            reward_schedulers.append(rws)
            pkt_loss[id + 1].append(dpkts)
            # clearing alloc users
            sc.clear()


        # incrementing slot counter
        self.curr_block += 1
        self.alloc_users = [[] for i in range(len(self.F))]
        self.alloc_users[self.curr_freq].append(action)
        self.rates_pkt_per_s = self.rateEstimationUsers(self.F, self.alloc_users, self.mimo_systems, self.K,
                                                        self.curr_slot, self.packet_size_bits)

        # rate estimation for all users
        rates = []
        for i in range(self.K):
            r = self.rateEstimationUsersAll(self.F, [[i]], self.mimo_systems, self.K,
                                            self.curr_slot, self.packet_size_bits)
            rates.append(r[i])
        self.rates = rates

        # Updating SE per user selected
        self.recent_spectral_eff = self.updateSEUsers(self.F, self.alloc_users, self.mimo_systems,
                                                      self.recent_spectral_eff, self.curr_slot)
        reward, drop = self.rewardCalc(self.alloc_users)
        pkt_loss[0].append(drop)
        # resetting the allocated users
        self.alloc_users = [[] for i in range(len(self.F))]

        self.mimo_systems = self.updateMIMO(self.mimo_systems, self.curr_slot)  # Update MIMO conditions

        # updating observation space and reward
        self.observation_space = self.updateObsSpace(self.buffers, self.buffer_size, self.recent_spectral_eff,
                                                     self.max_spectral_eff, self.max_packet_age)

        if (self.curr_block == self.blocks_ep):  # Episode has ended
            self.reset()
            self.ep_count += 1
        else:
            #self.end_ep = False
            pass
        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return self.observation_space, [reward, reward_schedulers, pkt_loss], self.end_ep, info

    def rewardCalc(self, alloc_users: list) -> Tuple:  # Calculating reward value
        pkt_rate  = self.rates_pkt_per_s
        (tx_pkts, dropped_pkts_sum, dropped_pkts, t_pkts, incoming_pkts, buffers) = self.pktFlow(pkt_rate,
                                                                                                self.buffers,
                                                                                                self.mimo_systems)

        reward = 10 * (tx_pkts / 50000)

        return reward, dropped_pkts_sum

    def rateEstimationUsersAll(self, F: list, alloc_users: list, mimo_systems: list, K: int, curr_slot: int,
                               packet_size_bits: int) -> list:  # Calculates the rate per second for each user considering if it was selected to transmit in any frequency and
        rates_pkt_per_s = np.zeros((K,))  # Considering rates are per second

        for f in range(len(F)):
            for au in alloc_users[f]:
                se_freq = mimo_systems[f].SE_for_given_sample(curr_slot, [au], F[f], avoid_estimation_errors=False)
                rates = (se_freq * mimo_systems[f].BW) / packet_size_bits
                rates_pkt_per_s[au] = rates[au]
        return rates_pkt_per_s

    def rateEstimationUsers(self, F: list, alloc_users: list, mimo_systems: list, K: int, curr_slot: int,
                            packet_size_bits: int) -> list:  # Calculates the rate per second for each user considering if it was selected to transmit in any frequency and
        rates_pkt_per_s = np.zeros((K,))  # Considering rates are per second

        for f in range(len(F)):
            se_freq = mimo_systems[f].SE_for_given_sample(curr_slot, alloc_users[f], F[f],
                                                          avoid_estimation_errors=False)
            rates_pkt_per_s += (se_freq * mimo_systems[f].BW) / packet_size_bits

        return rates_pkt_per_s

    def updateSEUsers(self, F: list, alloc_users: list, mimo_systems: list, recent_spectral_eff: np.ndarray,
                      curr_slot: int):  # Function which update the Spectral efficiency value for each UE used in the last block
        for f in range(len(F)):
            SE = mimo_systems[f].SE_for_given_sample(curr_slot, alloc_users[f], F[f], avoid_estimation_errors=False)
            recent_spectral_eff[alloc_users[f], f] = SE[alloc_users[f]]
        return recent_spectral_eff

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

        # Oldest packets
        oldest_packet_per_buffer = buffer_states[1]
        oldest_packet_per_buffer[
            oldest_packet_per_buffer > max_packet_age] = max_packet_age  # All values above threshold are set to the maximum value allowed
        oldest_packet_per_buffer = oldest_packet_per_buffer / max_packet_age  # Normalization

        # Spectral efficiency
        spectral_eff = recent_spectral_eff / max_spectral_eff

        # original
        # return np.hstack((buffer_occupancies, oldest_packet_per_buffer, spectral_eff.flatten()))
        # simplificando o estado
        max_rate = np.max(self.rates)
        p_rates = self.rates / max_rate
        thr = np.minimum(self.rates, self.buffers.buffer_occupancies)
        max_rate = np.max(thr)
        p_rates = thr / self.buffer_size
        #return np.hstack((buffer_occupancies, spectral_eff.flatten(), p_rates.flatten(), oldest_packet_per_buffer))
        return np.hstack((p_rates.flatten(), buffer_occupancies, spectral_eff.flatten()))

    # calculation of packets transmission and reception (from transmit_and_receive_new_packets function)
    def pktFlow(self, pkt_rate: float, buffers: Buffers, mimo_systems: list) -> (
                float, float, float, float, float, list):
        available_rate = np.floor(pkt_rate).astype(int)
        t_pkts = available_rate if (np.sum(available_rate) == 0) else buffers.packets_departure(
            available_rate)  # transmission pkts
        dropped_pkts = buffers.get_dropped_last_iteration()
        dropped_pkts_sum = np.sum(dropped_pkts)
        incoming_pkts = mimo_systems[0].get_current_income_packets()
        buffers.packets_arrival(incoming_pkts)  # Updating Buffer
        tx_pkts = np.sum(t_pkts)

        return (tx_pkts, dropped_pkts_sum, dropped_pkts, t_pkts, incoming_pkts, buffers)

    def makeMIMO(self, F: list, K: int) -> list:  # Generating MIMO system for each user
        return [MassiveMIMOSystem(K=K, frequency_index=f + 1) for f in range(len(F))]

    def loadEpMIMO(self, mimo_systems: list, F: list) -> list:

        # calculate the matrices to generate channel estimates
        episode_number = np.random.randint(0, self.mimo_systems[0].num_episodes - 1)
        for f in range(len(F)):
            # same episode number for all frequencies
            self.mimo_systems[f].load_episode(episode_number)
            self.mimo_systems[f].update_intercell_interference()  # avoid interference=0 in beginning of episode
        return mimo_systems

    def reset(self):  # Required by script to initialize the observation space

        self.rws = []
        self.curr_block = 0
        # Buffer
        self.buffers.reset_buffers()  # Reseting buffers
        # get some traffic to avoid starting from empty buffers
        self.buffers.packets_arrival(self.mimo_systems[0].get_current_income_packets())

        # MIMO Interference
        self.mimo_systems = self.loadEpMIMO(self.mimo_systems, self.F)

        # Cleaning count of users
        self.c_users = np.zeros(self.K)

        self.end_ep = True
        self.observation_space = self.updateObsSpace(self.buffers, self.buffer_size, self.recent_spectral_eff,
                                                     self.max_spectral_eff, self.max_packet_age)

        self.ep_count += 1

        return self.observation_space

    def close(self):
        pass

    def render(self):
        pass
        # print("----------------------------")
        # print("Episode " + str(self.ep_count))
        # print("Block " + str(self.curr_block))
        # print(self.buffers.buffer_occupancies)


    def rewardCalcSchedulers(self, mimo_systems: list, pkt_rate, buffers: Buffers) -> float:  # Calculating reward value

        (tx_pkts, dropped_pkts_sum, dropped_pkts, t_pkts, incoming_pkts, buffers) = self.pktFlow(pkt_rate,
                                                                                                 buffers,
                                                                                                 mimo_systems)

        reward = 10 * (tx_pkts / 50000)

        return reward, dropped_pkts_sum