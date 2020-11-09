'''
Complete env for a massive MIMO system.
'''
import numpy as np
from random import choices, randint
from akpy.MassiveMIMOSystem3 import MassiveMIMOSystem
from akpy.buffers_at_BS import Buffers
from akpy.quantizer import OneBitUniformQuantizer
from bidict import bidict
import itertools

class EnvironmentMassiveMIMO:
    def __init__(self):
        self.__version__ = "0.1.0"

        self.F=2 #number of frequencies

        self.num_realizations_for_channel_estimation = 10
        self.num_coherence_blocks_per_episode = 2
        self.num_ttis_per_coherence_block = 600
        self.num_bs_antennas = 4

        #design quantizers for CSI (SE and margin) AK-TODO here will assume simple 1 bit
        #xx=self.massiveMIMOSystems[0].generate_SE_statistics()
        self.quantizer_SE = OneBitUniformQuantizer(threshold=0) #5 is between 1 and 9 bits

        self.K=3 #number of connected users
        self.Kmax=2 #maximum number of served users per TTI

        self.num_buffer_occupancy_possible_values = 4 #quantization values
        self.num_packet_age_possible_values = 2 #quantization values
        self.buffers = Buffers(num_buffers=self.K, buffer_size=int(np.log2(self.num_buffer_occupancy_possible_values)),
                               max_packet_age=self.num_packet_age_possible_values-1)

        #In this work, the UE $k$ obtains its estimated $\textrm{MCS}_{kf}$ for a given frequency $f$ by quantizing the maximum (theoretical) SE that could be achieved in the last TTI it was served using $f$. If there was an outage on that TTI, the estimate is the immediately lower MCS. The value $\textrm{MCS}_{kf}$ remains constant until UE $k$ is served again using $f$.
        MCS_initial_value = 1
        self.MCS_num_possible_values = 2
        self.MCS = MCS_initial_value * np.ones((self.K, self.F), dtype=np.int)

        #adopt three bidirectional maps
        self.bidict_actions = convert_list_of_possible_tuples_in_bidct(self.get_all_possible_actions())
        self.bidict_states = convert_list_of_possible_tuples_in_bidct(self.get_all_possible_states())
        #self.bidict_rewards = convert_list_of_possible_tuples_in_bidct()

        #Record if there was outage in previous TTI
        self.outages = np.zeros((self.K,), dtype=np.bool)

        self.reward = 0

        self.S = self.get_num_states()
        self.A = self.get_num_actions()

        #AK-TODO need only one
        #self.currentObservation = 0
        self.currentIteration = 0 #continuous, count time and also TTIs
        self.current_coherence_block = 0
        self.current_tti = 0 # TTI number within a block, reset at each block

        self.reset() #create MIMO systems, etc

    def incoming_traffic(self):
        x = 1
        packets_in = x * np.ones((self.K,), np.uint64)
        return packets_in

    def get_num_states(self):
        return len(self.get_all_possible_states())

    def get_num_actions(self):
        return len(self.get_all_possible_actions())

    def get_current_reward(self):
        return self.reward

    def get_all_possible_actions(self):
        '''M is the number of users and B the buffer size'''
        list_users = list()
        for i in range(self.K):
            list_users.append('u' + str(i))
        all_served_users = list(itertools.combinations(list_users, self.Kmax))

        list_frequencies = list()
        for i in range(self.F):
            list_frequencies.append('f' + str(i))
        all_frequencies = list(itertools.product(list_frequencies, repeat=self.Kmax))

        list_margins = [True, False]
        all_margins = list(itertools.product(list_margins, repeat=self.Kmax))

        all_actions = [(a,b,c) for a in all_served_users for b in all_frequencies for c in all_margins]

        #print(all_served_users)
        #print(all_frequencies)
        #print(all_actions)
        #print(len(all_actions))
        #exit(-1)
        return all_actions

    def get_all_possible_states(self):
        #CSI
        all_mcs = list(itertools.product(np.arange(self.MCS_num_possible_values), repeat=self.K * self.F))
        list_outages = [True, False]
        all_outages = list(itertools.product(list_outages, repeat=self.K))

        all_csis= [(a,b) for a in all_mcs for b in all_outages]

        #print(all_csis)
        #print(len(all_csis))

        #QSI
        all_occupancies = list(itertools.product(np.arange(self.num_buffer_occupancy_possible_values), repeat=self.K))
        #recall that -1 is used to indicate empty buffer - can change it later AK-TODO
        all_packet_ages = list(itertools.product(np.arange(self.num_packet_age_possible_values), repeat=self.K))
        all_qsis= [(a,b) for a in all_occupancies for b in all_packet_ages]
        #AK-TODO: should eliminate from list the unfeasible situations of empty lists with aged packets
        #print(all_qsis)
        #print(len(all_qsis))

        all_states = [(a,b) for a in all_csis for b in all_qsis]
        #print(all_states)
        #print(len(all_states))

        return all_states

    #I don't need a table for the rewards as in the other env. Here, I generate them as we go.
    #def get_all_possible_rewards(self):
    #    return

    def step(self, action_index):
        """
        The agent takes a step in the environment.
        Parameters
        ----------
        action :
        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
            reward (float) :
            episode_over (bool) :
            info (dict) :
        """

        #take action, convert from index to action
        #Action example (('u0', 'u1'), ('f0', 'f0'), (True, True))
        #State example (((0, 0, 0, 0, 0, 0), (True, True, True)), ((0, 0, 0), (0, 0, 0)))
        action = self.bidict_actions[action_index]
        action_users = action[0]
        action_freqs = action[1]
        action_margins = action[2]

        selected_users = np.zeros((self.Kmax,), dtype=np.int)
        selected_frequencies = np.zeros((self.Kmax,), dtype=np.int)
        users_per_frequency = list()
        for i in range(self.F):
            users_per_frequency.append(list())
        for i in range(self.Kmax):
            selected_users[i] = int(action_users[i][1:])
            selected_frequencies[i] = int(action_freqs[i][1:])
            users_per_frequency[selected_frequencies[i]].append(selected_users[i])

        #print('Users per freq')
        #for i in range(self.F):
        #    print(users_per_frequency[i])
        #exit(-1)

        #allocate SE to calculate rewards
        #each freq has its own system
        all_SEs = np.zeros((self.K,))
        for f in range(self.F):
            selected_users = users_per_frequency[f]
            if len(selected_users) < 1:
                continue #no users selected at this freq
            massiveMIMOSystem = self.massiveMIMOSystems[f]
            SE = massiveMIMOSystem.SE_for_given_range_of_channels(self.current_tti, selected_users, SE_evaluation_window = 1)
            all_SEs += SE
            #print(SE)

        previous_state = self.get_complete_state() #state used for reward (t-1)

        #AK-TODO make it right
        x = 2 #using 1 for incoming
        num_removed_packets = self.buffers.packets_departure(x * self.quantizer_SE.quantize(all_SEs))

        #self.reward = np.sum(num_removed_packets) - np.sum(self.buffers.get_dropped_last_iteration())
        self.reward = np.sum(all_SEs) - 3 * np.sum(self.buffers.get_dropped_last_iteration())
        #self.reward = np.sum(all_SEs) - np.sum(self.buffers.oldest_packet_per_buffer()) - np.sum(self.buffers.get_dropped_last_iteration())

        #update for next state TODO
        num_incoming_packets = self.incoming_traffic()
        self.buffers.packets_arrival(num_incoming_packets)

        gameOver = False
        if self.currentIteration == self.num_coherence_blocks_per_episode * self.num_ttis_per_coherence_block:
            ob = self.reset()
            gameOver = True  # game ends
        else:
            ob = self.get_state()

        history = {"time": self.currentIteration,
                   "block": self.current_coherence_block,
                   "tti": self.current_tti,
                   "action_t": action,
                   "state": previous_state,
                   "reward": self.reward}
        # history version with actions and states, not their indices
        # history = {"time": self.currentIteration, "action_t": self.actionListGivenIndex[action],
        #           "reward_tp1": reward, "observation_tp1": self.stateListGivenIndex[self.get_state()]}
        self.currentIteration += 1
        self.current_tti += 1

        if self.current_tti == self.num_ttis_per_coherence_block:
            self.current_tti = 0
            self.current_coherence_block += 1
            self.prepare_for_coherence_block()

        #print(history) #AK

        return ob, self.reward, gameOver, history

    def get_complete_state(self):
        """Get the current observation.
The channel quality for all $F$ frequencies in the current TTI is represented by an integer denoting the MCS that should be used in the downlink, as estimated by the UE.
The CSI also includes a binary value for each user
indicating if there was an outage for that user in the served frequency on previous TTI or not. If the user was not served, then this flag bit is 0.
Together with the $F$ values of MCS, this bit leads to a CSI represented by a vector with $K (F+1)$ elements.

The QSI indicates, for each of the $K$ connected users,  the number of packets in queue and the time the oldest packet has been queued.
        """
        #s = self.bidict_states[0]
        #print('State example', s, len(s))
        #i = self.bidict_states.inv[s]
        #print('State example index', i)

        CSI = (tuple(self.MCS.flatten()), tuple(self.outages))
        buffer_states = self.buffers.get_buffer_states()
        QSI = (tuple(buffer_states[0]), tuple(buffer_states[1]))
        s = (CSI, QSI)
        return s

    def get_state(self):
        """Get the current observation.
The channel quality for all $F$ frequencies in the current TTI is represented by an integer denoting the MCS that should be used in the downlink, as estimated by the UE.
The CSI also includes a binary value for each user
indicating if there was an outage for that user in the served frequency on previous TTI or not. If the user was not served, then this flag bit is 0.
Together with the $F$ values of MCS, this bit leads to a CSI represented by a vector with $K (F+1)$ elements.

The QSI indicates, for each of the $K$ connected users,  the number of packets in queue and the time the oldest packet has been queued.
        """
        #s = self.bidict_states[0]
        #print('State example', s, len(s))
        #i = self.bidict_states.inv[s]
        #print('State example index', i)
        s = self.get_complete_state()
        #print('State example', s2, len(s2))
        #state_index = self.bidict_states.inv[s2]
        self.currentObservation = self.bidict_states.inv[s]
        #need to quantize
        return self.currentObservation

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.create_mimo_systems()
        self.currentIteration = 0
        self.current_tti = 0
        self.current_coherence_block = 0
        #calculate the matrices to generate channel estimates
        return self.get_state()

    def create_mimo_systems(self):
        self.massiveMIMOSystems = list()
        for f in range(self.F):
            self.massiveMIMOSystems.append(MassiveMIMOSystem(K=self.K, Kmax=self.Kmax, frequency_index=f+1,
                                                             num_bs_antennas=self.num_bs_antennas))
            #self.massiveMIMOSystems[f].set_num_realizations(num_TTIs_per_coherence_block)
        self.prepare_for_coherence_block()

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

    def prepare_for_coherence_block(self):
        #num_realizations_for_channel_estimation = 20
        #num_ttis_per_coherence_block = 3
        #num_coherence_blocks_per_episode = 4
        for f in range(self.F):
            massiveMIMOSystem = self.massiveMIMOSystems[f]
            #initialize
            #calculate the matrices to generate channel estimates
            massiveMIMOSystem.set_num_realizations(self.num_realizations_for_channel_estimation)
            H = massiveMIMOSystem.channel_realizations()
            massiveMIMOSystem.initialize_mmse_channel_estimator(H)
            massiveMIMOSystem.set_num_realizations(self.num_ttis_per_coherence_block + 1)
            H = massiveMIMOSystem.channel_realizations()
            debugme=False
            massiveMIMOSystem.prepare_H_Hhat_for_coherence_block(H,debugme)

def convert_list_of_possible_tuples_in_bidct(list_of_tuples):
    #assume there are no repeated elements
    N = len(list_of_tuples)
    this_bidict = bidict()
    for n in range(N):
        this_bidict.put(n,list_of_tuples[n])
    return this_bidict

def test_bidct():
    x=list()
    x.append((3,5,'a'))
    x.append((3,4,'a'))
    x.append(('b'))
    bidict = convert_list_of_possible_tuples_in_bidct(x)
    print(bidict[1])
    print(bidict.inv['b'])

    #test return
    x = np.random.randn(3,4)
    print(x)
    a = tuple(x.flatten())
    print(a)
    print(len(a))
    b = np.array(a).reshape((3,4))
    print(b)
    exit(-1)

if __name__ == '__main__':
    #test_bidct()
    env = EnvironmentMassiveMIMO()
    #print(env.get_all_possible_actions())
    print('Action example', env.bidict_actions[0])

    #x = env.get_state()
    #print(x)
    #exit(-1)

    N = 1500
    Na = env.get_num_actions()
    for i in range(N):
        action_index = int(np.random.choice(Na,1)[0])
        ob, reward, gameOver, history = env.step(action_index)
        if gameOver:
            print('Game over! End of episode.')
            #env = EnvironmentMassiveMIMO()
        print(history)