import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import copy
import random

from random import choices
from akpy.buffers_at_BS_n import Buffers
from akpy.MassiveMIMOSystem8 import MassiveMIMOSystem
from schedulers.max_th import MaxTh
from schedulers.proportional_fair import ProportionalFair
from schedulers.round_robin import RoundRobin
from utils.action_space import ActionSpace


class SraEnv1(gym.Env):

  def __init__(self,**kwargs):

    self.running_tp = 0  # 0 - trainning; 1 - validating
    self.alpha = 1.0 # loss
    self.beta = 0.0 # delay
    self.gamma = 1.0 # tx pkts
    self.rotate = False
    try:
      self.se_ues_op = kwargs['se_ues_op']
      self.shuffle_se_op = kwargs['shuffle_se_op']
      self.random_op = kwargs['random_op']
    except:
      self.se_ues_op = None
      self.shuffle_se_op = False
      self.random_op = False
    try:
      self.force_error = kwargs['force_error']
    except:
      self.force_error = False
    try:
      self.use_se = kwargs['use_se']
    except:
      self.use_se = False
    try:
      self.norm_obs = kwargs['norm_obs']
    except:
      self.norm_obs = False
    try:
      self.use_mean_eff = kwargs['use_mean_eff']
    except:
      self.use_mean_eff = False
    try:
      if kwargs['rotate']:
        self.rotate = kwargs['rotate']
    except:
      pass
    try:
      self.alpha = kwargs['alpha']
      self.beta = kwargs['beta']
      self.gamma = kwargs['gamma']
      self.running_tp = kwargs['runningtp']
      if kwargs['seed']:
        np.random.seed(kwargs['seed'])
    except:
      print("Default parameters")
    self.full_obs = False # or True for full observability
    try:
      self.full_obs = kwargs['fullobs']
    except:
      print("Default observability : partial")
    self.F = [2, 2]
    self.K = 10
    self.ep_count = 1
    self.end_ep = False  # Indicate the end of an episode
    # Number of slots per Block
    self.slots_block = 2
    # Number of blocks per episode
    self.blocks_ep = 1000
    try:
      self.blocks_ep = kwargs['steps']
    except:
      print('Default Steps')
    #if self.running_tp == 1:
    #  self.blocks_ep = 500
    self.sub_blocks_ep = 50
    self.episode_number = 0
    self.curr_slot = 0
    self.curr_block = 0
    self.sub_curr_block = 0
    self.count_users = 0  # Count number of users allocated in the current frequency
    self.curr_freq = 0  # Frequency being used now
    # Allocated users per frequency, e.g. [[1,2,3],[4,5,6]] which means that user 1, 2 and 3 were
    # allocated in the first frequency and users 4, 5 and 6 were allocated in the second frequency
    self.alloc_users = [[] for i in range(len(self.F))]
    self.hist_alloc_users = np.zeros((len(self.F), self.K))
    self.max_spectral_eff = 7.8  # from dump_SE_values_on_stdout()
    # self.recent_spectral_eff = (self.max_spectral_eff / 2) * np.ones((self.K, len(self.F)))  # in bps/Hz
    self.recent_spectral_eff = (self.max_spectral_eff / len(self.F)) * np.ones((self.K, len(self.F)))  # in bps/Hz
    self.spectral_eff = (self.max_spectral_eff / len(self.F)) * np.ones((self.K, len(self.F)))  # in bps/Hz
    self.rates_pkt_per_s = np.array([])
    self.full_eff = []
    self.full_eff_hist = [] # collecting all SE history
    self.full_rate_hist = []
    self.episode_rate_hist = []

    # Buffer variables
    self.packet_size_bits = 1024  # if 1024, the number of packets per bit coincides with kbps
    self.buffer_size =  30 * 8 * self.packet_size_bits  # size in bits obtained by multiplying by self.packet_size_bits
    self.max_packet_age = 1000  # limits the packet age. Maximum value
    try:
      self.max_packet_age = kwargs['max_age']
    except:
      print('Default max age')
    self.buffers = Buffers(num_buffers=self.K, buffer_size=self.buffer_size, max_packet_age=self.max_packet_age)
    self.buffers.reset_buffers()  # Initializing buffers
    self.copy_buffers = copy.deepcopy(self.buffers)

    # individual rates history
    self.rates_history = []
    self.rates = [self.buffer_size for i in range(self.K)]
    # initialization to avoid error in the first run
    self.rates_history.append([self.buffer_size for i in range(self.K)])

    self.rws = []
    self.rws_hist = []

    #set episodes
    try:
      self.pred_eps = kwargs['eps']
      self.pred_eps_count = 0
    except:
      self.pred_eps = None

    # Massive MIMO System

    self.tf_folder = kwargs['tf_folder']
    self.tf_folder_acr = kwargs['tf_folder_acr']
    self.mimo_systems = self.makeMIMO(self.F, self.K)
    self.eps_prob = [1/self.mimo_systems[0].num_episodes for e in range(self.mimo_systems[0].num_episodes)]
    self.eps_count = [1 for e in range(self.mimo_systems[0].num_episodes)]
    self.mimo_systems = self.loadEpMIMO(self.mimo_systems, self.F, tp=self.running_tp)

    self.history_buffers = []
    self.rates = [1.] * self.K

    #self.action_space = spaces.MultiDiscrete([[self.K, self.K], [self.K, self.K]])
    ## getting combinatorial actions
    self.actions = ActionSpace.get_action_space(K=self.K, F=self.F)
    self.action_space = spaces.Discrete(len(self.actions))
    obs = self.reset()
    self.observation_space = spaces.Box(low=0,high=self.max_packet_age,shape=obs.shape,dtype=np.int32)

    self.schedulers = []
    # creating the Round Robin scheduler instance with independent buffers
    #self.schedulers.append(RoundRobin(K=self.K, F=self.F, buffers=copy.deepcopy(self.buffers)))
    # another scheduler instance, for testing with multiple schedulers
    #self.schedulers.append(ProportionalFair(K=self.K, F=self.F, buffers=copy.deepcopy(self.buffers)))
    self.schedulers.append(MaxTh(K=self.K, F=self.F, buffers=copy.deepcopy(self.buffers)))
    try:
      if not kwargs['use_sch']:
        self.schedulers = []
    except:
      print('Default Schedulers - Full')

  def step(self, action_index):

    ## allocation
    action = self.actions[action_index]

    for act in action:
      self.alloc_users[self.curr_freq] = act
      self.curr_freq += 1


    self.rates_pkt_per_s = self.rateEstimationUsers(self.F, self.alloc_users, self.mimo_systems, self.K,
                                                    self.curr_block, self.packet_size_bits)

    # Updating SE per user selected
    self.recent_spectral_eff = self.updateSEUsers(self.F, self.alloc_users, self.mimo_systems,
                                                  self.recent_spectral_eff, self.curr_block)

    reward, _, _ = self.rewardCalc(self.rates_pkt_per_s)

    self.rws.append(reward)

    # incrementing slot counter
    self.curr_block += 1
    self.sub_curr_block += 1

    # resetting the allocated users
    self.alloc_users = [[] for i in range(len(self.F))]

    self.mimo_systems = self.updateMIMO(self.mimo_systems, self.curr_block)  # Update MIMO conditions

    # resetting
    self.curr_freq = 0
    # self.curr_slot = 0

    if (self.curr_block == self.blocks_ep):  # Episode has ended
      # self.reset() # aparently will be reseted by gym controller
      self.end_ep = True
    else:
      self.end_ep = False
      #if self.sub_curr_block == self.sub_blocks_ep:
      #  self.mimo_systems = self.loadEpMIMO(self.mimo_systems, self.F, tp=self.running_tp)
      #  self.sub_curr_block = 0

    # updating observation space and reward
    self.observation_space = self.updateObsSpace(self.buffers, self.buffer_size, self.recent_spectral_eff,
                                                 self.max_spectral_eff, self.max_packet_age)

    info = {}

    return self.observation_space, reward, self.end_ep, info

  def reset(self):  # Required by script to initialize the observation space

    if self.se_ues_op and self.shuffle_se_op:
      random.shuffle(self.se_ues_op)

    if self.random_op:
      rp = [random.uniform(0.1, 2.0) for se in range(self.K)]
      self.se_ues_op = rp

    if len(self.rws) > 0:
      print("Mean RW " + str(np.mean(self.rws)))
      self.rws_hist.append(np.mean(self.rws))
      self.rws = []
    if self.rotate:
      if self.tf_folder_acr == 'nds20':
        self.tf_folder = '/traffic_interference_nds_25/'
        self.tf_folder_acr = 'nds25'
      else:
        self.tf_folder = '/traffic_interference_nds_20/'
        self.tf_folder_acr = 'nds20'
    #print(self.eps_count)
    self.full_eff = []
    self.curr_block = 0
    self.sub_curr_block = 0
    self.end_ep = True
    self.rates_history = []
    # initialization to avoid error in the first run
    self.rates_history.append([self.buffer_size for i in range(self.K)])
    # Buffer
    self.buffers.reset_buffers()  # Reseting buffers
    # get some traffic to avoid starting from empty buffers
    self.buffers.packets_arrival(self.mimo_systems[0].get_current_income_packets())

    # MIMO Interference
    self.mimo_systems = self.loadEpMIMO(self.mimo_systems, self.F, tp=self.running_tp)

    self.observation_space = self.updateObsSpace(self.buffers, self.buffer_size, self.recent_spectral_eff,
                                                 self.max_spectral_eff, self.max_packet_age)
    self.ep_count += 1

    return self.observation_space

  def close(self):
    pass

  def render(self):
    pass

  def reset_(self):  # Required by script to initialize the observation space
    self.curr_block = 0
    self.end_ep = True
    # Cleaning count of users
    self.c_users = np.zeros(self.K)
    self.full_rate_hist.append(np.mean(self.episode_rate_hist, axis=0))
    self.episode_rate_hist = []
    # computing average SE by UE/Fc
    self.full_eff_hist.append(np.mean(self.full_eff, axis=0))
    self.full_eff = []
    # Buffer
    self.buffers.reset_buffers()  # Reseting buffers
    # get some traffic to avoid starting from empty buffers
    self.buffers.packets_arrival(self.mimo_systems[0].get_current_income_packets())
    self.copy_buffers = copy.deepcopy(self.buffers)
    self.rates_history = []
    # initialization to avoid error in the first run
    self.rates_history.append([self.buffer_size for i in range(self.K)])

    # MIMO Interference
    self.mimo_systems = self.loadEpMIMO(self.mimo_systems, self.F, tp=self.running_tp)

    # reseting the schedulers
    for sc in self.schedulers:
      sc.reset()
      sc.buffers = copy.deepcopy(self.buffers)

    self.observation_space = self.updateObsSpace(self.buffers, self.buffer_size, self.recent_spectral_eff,
                                                 self.max_spectral_eff, self.max_packet_age)
    self.ep_count += 1

    return self.observation_space

  def reset_b(self):  # Required by script to initialize the observation space
    self.curr_block = 0
    self.end_ep = True
    if self.tf_folder_acr == 'nds25':
      self.tf_folder = '/traffic_interference_nds_20/'
      self.tf_folder_acr = 'nds20'
    else:
      self.tf_folder = '/traffic_interference_nds_25/'
      self.tf_folder_acr = 'nds25'

    self.mimo_systems = self.makeMIMO(self.F, self.K)

    self.mimo_systems = self.loadEpMIMO(self.mimo_systems, self.F, tp=self.running_tp)

    self.buffers.reset_buffers()  # Reseting buffers
    # get some traffic to avoid starting from empty buffers
    self.buffers.packets_arrival(self.mimo_systems[0].get_current_income_packets())

    self.observation_space = self.updateObsSpace(self.buffers, self.buffer_size, self.recent_spectral_eff,
                                                 self.max_spectral_eff, self.max_packet_age)
    self.ep_count += 1

    return self.observation_space

  def step_(self, action_index):
    reward_schedulers = []
    individual_rw_schedulers = []
    reward = 0
    pkt_loss = [[] for i in range(len(self.schedulers) + 1)]
    pkt_delay = [[] for i in range(len(self.schedulers) + 1)]
    self.compute_rate()  # computing the rates per user

    ## allocation
    action = self.actions[action_index]
    for act in action:
      self.alloc_users[self.curr_freq] = act
      self.curr_freq += 1

    ## baseline schedulers allocation
    for sc in self.schedulers:
      if sc.name == 'Proportional Fair' or sc.name == 'Max th':
        curr_f = 0
        sc.exp_thr = self.rates_history[-1]
        if self.full_obs:
          ues = sc.policy_action()
        else:
          ues = sc.policy_action_()

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

      if sc.name == 'Max th' and not self.full_obs:
        sc.update_recent_rate(rate=rates_pkt_per_s_schedulers)

      if sc.name == 'Proportional Fair':
        if not self.full_obs:
          sc.update_recent_rate(rate=rates_pkt_per_s_schedulers)
        for i, v in enumerate(rates_pkt_per_s_schedulers):
          sc.thr_count[i].append(v)
      # computing the rewards for each scheduler
      rws, dpkts, t_pkts = self.rewardCalcSchedulers(self.mimo_systems, rates_pkt_per_s_schedulers, sc.buffers)
      reward_schedulers.append(rws)
      individual_rw_schedulers.append(t_pkts)
      # pkt_loss[id + 1].append(dpkts)
      pkt_loss[id + 1] = -10.
      pkt_delay[id + 1].append(self.compute_pkt_delay(sc.buffers))
      std = np.std(pkt_delay[id + 1])  # standard deviation
      # clearing alloc users
      sc.clear()
      ####################################################

    ## Computing reward for DRL-Agent
    self.rates_pkt_per_s = self.rateEstimationUsers(self.F, self.alloc_users, self.mimo_systems, self.K,
                                                    self.curr_block, self.packet_size_bits)

    # Updating SE per user selected
    self.recent_spectral_eff = self.updateSEUsers(self.F, self.alloc_users, self.mimo_systems,
                                                  self.recent_spectral_eff, self.curr_block)

    reward, dropped_pkt, dropped_pkts_percent_mean, t_pkts_drl = self.rewardCalc_(self.F, self.alloc_users, self.mimo_systems,
                                                                                  self.K, self.curr_slot,
                                                                                  self.packet_size_bits,
                                                                                  [self.rates_pkt_per_s, self.rates],
                                                                                  self.buffers,
                                                                                  -100, self.recent_spectral_eff,
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

    # incrementing slot counter
    self.curr_block += 1

    if len(self.alloc_users[self.curr_freq]) == self.F[self.curr_freq]:
      self.curr_freq += 1
      self.curr_slot += 1

    if (self.curr_block == self.blocks_ep):  # Episode has ended up
      # before reset, compute the episode loss
      loss = np.sum(self.buffers.buffer_history_dropped) / np.sum(self.buffers.buffer_history_incoming)
      pkt_loss[0] = loss  # changing the fake value -10 by the real loss value
      ## schedulers - computing the reward
      for id, sc in enumerate(self.schedulers):
        loss = np.sum(sc.buffers.buffer_history_dropped) / np.sum(sc.buffers.buffer_history_incoming)
        pkt_loss[id + 1] = loss  # changing the fake value -10 by the real loss value
      self.reset_()
      self.ep_count += 1

    info = {}
    # print("Episode " + str(self.ep_count) + " - Block " + str(self.curr_block) + " RW " + str(reward))
    #return self.observation_space, [reward, reward_schedulers, pkt_loss, pkt_delay], self.end_ep, info

    return self.observation_space, [reward, reward_schedulers, pkt_loss, pkt_delay],\
           self.end_ep, info, individual_rw_schedulers, t_pkts_drl

  def step_2(self, action_index):
    reward_schedulers = []
    individual_rw_schedulers = []
    reward = 0
    pkt_loss = [[] for i in range(len(self.schedulers) + 1)]
    pkt_delay = [[] for i in range(len(self.schedulers) + 1)]
    se_hist = [[] for i in range(len(self.schedulers) + 1)]
    self.compute_rate()  # computing the rates per user

    ## allocation
    action = self.actions[action_index]
    for act in action:
      self.alloc_users[self.curr_freq] = act
      for a in act:
        self.hist_alloc_users[self.curr_freq][a] += 1
      self.curr_freq += 1


    ## baseline schedulers allocation
    for sc in self.schedulers:
      if sc.name == 'Proportional Fair' or sc.name == 'Max th':
        curr_f = 0
        sc.exp_thr = self.rates_history[0]
        if self.full_obs:
          ues = sc.policy_action()
        else:
          ues = sc.policy_action_()
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

      last_SE = self.estimate_SE(self.F, sc.alloc_users, self.mimo_systems, None, self.curr_block)

      if sc.name == 'Max th':
        sc.update_recent_rate(rate=rates_pkt_per_s_schedulers)

      if sc.name == 'Proportional Fair':
        sc.update_recent_rate(rate=rates_pkt_per_s_schedulers)
        for i, v in enumerate(rates_pkt_per_s_schedulers):
          sc.thr_count[i].append(v)
      # computing the rewards for each scheduler
      rws, dpkts, t_pkts = self.rewardCalcSchedulers_(self.mimo_systems, rates_pkt_per_s_schedulers, sc.buffers)
      reward_schedulers.append(rws)
      individual_rw_schedulers.append(t_pkts)
      loss = np.sum(sc.buffers.buffer_history_dropped) / np.sum(sc.buffers.buffer_history_incoming)
      #pkt_loss[id + 1].append(dpkts)
      pkt_loss[id + 1].append(loss)
      pkt_delay[id + 1].append(self.compute_pkt_delay(sc.buffers))
      std = np.std(pkt_delay[id + 1])  # standard deviation
      se_hist[id + 1].append(np.mean(last_SE[1]))
      # clearing alloc users
      sc.clear()
      ####################################################

    ## Computing reward for DRL-Agent
    self.rates_pkt_per_s = self.rateEstimationUsers(self.F, self.alloc_users, self.mimo_systems, self.K,
                                                    self.curr_block, self.packet_size_bits)

    mr = self.max_rate(self.F, [[0,1,2,3,4,5,6,7,8,9], [0,1,2,3,4,5,6,7,8,9]], self.mimo_systems, self.K,
                                                    self.curr_block, self.packet_size_bits)

    self.episode_rate_hist.append(mr)

    last_SE = self.estimate_SE(self.F, self.alloc_users, self.mimo_systems, None, self.curr_block)

    self.estimate_SE_all()

    # Updating SE per user selected
    if not self.force_error:
      self.recent_spectral_eff = self.updateSEUsers(self.F, self.alloc_users, self.mimo_systems,
                                                    self.recent_spectral_eff, self.curr_block)

    reward, dropped_pkt, dropped_pkts_percent_mean, t_pkts_drl = self.rewardCalc_(self.F, self.alloc_users, self.mimo_systems,
                                                                                  self.K, self.curr_slot,
                                                                                  self.packet_size_bits,
                                                                                  [self.rates_pkt_per_s, self.rates],
                                                                                  self.buffers,
                                                                                  -100, self.recent_spectral_eff,
                                                                                  update=True)

    #pkt_loss[0].append(dropped_pkts_percent_mean)
    loss = np.sum(self.buffers.buffer_history_dropped) / np.sum(self.buffers.buffer_history_incoming)
    pkt_loss[0].append(loss)
    pkt_delay[0].append(self.compute_pkt_delay(self.buffers))
    se_hist[0].append(np.mean(last_SE[1]))
    individual_loss = np.array(self.buffers.buffer_history_dropped) / np.array(self.buffers.buffer_history_incoming)

    # resetting the allocated users
    self.alloc_users = [[] for i in range(len(self.F))]

    self.mimo_systems = self.updateMIMO(self.mimo_systems, self.curr_block)  # Update MIMO conditions

    # updating observation space and reward
    self.observation_space = self.updateObsSpace(self.buffers, self.buffer_size, self.recent_spectral_eff,
                                                 self.max_spectral_eff, self.max_packet_age)

    # resetting
    self.curr_freq = 0
    self.curr_slot = 0

    # incrementing slot counter
    self.curr_block += 1

    if len(self.alloc_users[self.curr_freq]) == self.F[self.curr_freq]:
      self.curr_freq += 1
      self.curr_slot += 1

    if (self.curr_block == self.blocks_ep):  # Episode has ended up
      # before reset, compute the episode loss
      #loss = np.sum(self.buffers.buffer_history_dropped) / np.sum(self.buffers.buffer_history_incoming)
      #pkt_loss[0] = loss  # changing the fake value -10 by the real loss value
      ## schedulers - computing the reward
      #for id, sc in enumerate(self.schedulers):
      #  loss = np.sum(sc.buffers.buffer_history_dropped) / np.sum(sc.buffers.buffer_history_incoming)
      #  pkt_loss[id + 1] = loss  # changing the fake value -10 by the real loss value
      self.reset_()
      self.ep_count += 1

    info = {}
    # print("Episode " + str(self.ep_count) + " - Block " + str(self.curr_block) + " RW " + str(reward))
    #return self.observation_space, [reward, reward_schedulers, pkt_loss, pkt_delay], self.end_ep, info

    return self.observation_space, [reward, reward_schedulers, pkt_loss, pkt_delay, se_hist],\
           self.end_ep, info, individual_rw_schedulers, t_pkts_drl, individual_loss


  def compute_rates(self):
    # rate estimation for all users
    # considering allocation at first frequency (with less BW)
    rates = []
    for i in range(self.K):
      r = self.rateEstimationUsersAll([self.F[0]], [[i]], self.mimo_systems, self.K,
                                      self.curr_block, self.packet_size_bits)
      rates.append(r[i])
    self.rates_history.append(np.array(self.rates))
    return rates

  def compute_rate_all(self):
    # rate estimation for all users
    users = list(range(self.K))
    middle_index = len(users) // 2

    alloc = [users[:middle_index], users[middle_index:]]
    alloc = [users]
    rates1 = self.rateEstimationUsers([self.F[0]], alloc, self.mimo_systems, self.K,
                                          self.curr_block, self.packet_size_bits)
    rates2 = self.rateEstimationUsers([self.F[1]], alloc, self.mimo_systems, self.K,
                                      self.curr_block, self.packet_size_bits)

    return [rates1, rates2]

  def compute_rate(self):
    # rate estimation for all users
    users = list(range(self.K))
    middle_index = len(users) // 2

    alloc = [users[:middle_index], users[middle_index:]]
    alloc = [users]
    self.rates = self.rateEstimationUsers([self.F[0]], alloc, self.mimo_systems, self.K,
                                          self.curr_block, self.packet_size_bits)

    self.rates_history.append(np.array(self.rates))

  def rewardCalcSchedulers(self, mimo_systems: list, pkt_rate, buffers: Buffers) -> float:  # Calculating reward value

    (tx_pkts, dropped_pkts_sum, dropped_pkts, t_pkts, incoming_pkts, buffers,
     dropped_pkts_percent_mean) = self.pktFlow(pkt_rate, buffers, mimo_systems)

    reward =  (tx_pkts * self.packet_size_bits) / 1e6

    individual_thp = [((i * self.packet_size_bits) / 1e6) for i in t_pkts]

    return reward, dropped_pkts_percent_mean, individual_thp

  def rewardCalcSchedulers_(self, mimo_systems: list, pkt_rate, buffers: Buffers) -> float:  # Calculating reward value

    (tx_pkts, dropped_pkts_sum, dropped_pkts, t_pkts, incoming_pkts, buffers,
     dropped_pkts_percent_mean) = self.pktFlow(pkt_rate, buffers, mimo_systems)

    reward =  (tx_pkts * self.packet_size_bits) / 1e6

    individual_thp = [((i * self.packet_size_bits) / 1e6) for i in t_pkts]

    return reward, dropped_pkts_percent_mean, individual_thp

  def compute_pkt_delay(self, buffers):
    buffer_states = buffers.get_buffer_states()
    # Oldest packets
    oldest_packet_per_buffer = buffer_states[1]
    return oldest_packet_per_buffer

  def set_seed(self, seed):
    np.random.seed(seed)  # for reproducibility

  def rewardCalc(self, pkt_rate) -> float:  # Calculating reward value

    buffer_states = self.buffers.get_buffer_states()
    occup = np.sum(self.buffers.buffer_occupancies)

    # Oldest packets
    oldest_packet_per_buffer = buffer_states[1]
    # All values above threshold are set to the maximum value allowed
    oldest_packet_per_buffer[oldest_packet_per_buffer > self.max_packet_age] = self.max_packet_age
    # Normalization
    # oldest_packet_per_buffer = oldest_packet_per_buffer / self.max_packet_age
    (tx_pkts, dropped_pkts_sum, dropped_pkts, t_pkts, incoming_pkts, buffers, dropped_pkts_percent_mean) = self.pktFlow(
      pkt_rate, self.buffers, self.mimo_systems)

    # reward = (self.gamma * (tx_pkts * self.packet_size_bits) / 1e6) - (self.alpha * (dropped_pkts_sum * self.packet_size_bits)/ 1e6)
    #reward = (self.gamma * ((tx_pkts / occup) * 100)) - (self.alpha * ((dropped_pkts_sum / tx_pkts) * 100))
    reward = (self.gamma * ((tx_pkts / occup) * 100)) - (self.alpha * ((dropped_pkts_sum / tx_pkts) * 100))
    if np.mean(oldest_packet_per_buffer) > 3:
      reward -= self.beta * np.mean(oldest_packet_per_buffer)
    else:
      reward += self.beta * np.sum(oldest_packet_per_buffer)
    # reward = (self.gamma * (tx_pkts * self.packet_size_bits) / 1e6)

    return reward, dropped_pkts_sum, dropped_pkts_percent_mean

  def rateEstimationUsersAll(self, F: list, alloc_users: list, mimo_systems: list, K: int, curr_slot: int,
                             packet_size_bits: int) -> list:  # Calculates the rate per second for each user considering if it was selected to transmit in any frequency and
    rates_pkt_per_s = np.zeros((K,))  # Considering rates are per second

    for f in range(len(F)):
      for au in alloc_users[f]:
        se_freq = mimo_systems[f].SE_current_sample(curr_slot, [au], self.se_ues_op)
        rates = (se_freq[0] * mimo_systems[f].BW[f]) / packet_size_bits
        rates_pkt_per_s[au] = rates
    return rates_pkt_per_s

  def max_rate(self, F: list, alloc_users: list, mimo_systems: list, K: int, curr_slot: int,
                          packet_size_bits: int) -> list:  # Calculates the rate per second for each user considering if it was selected to transmit in any frequency and
    rates_pkt_per_s = np.zeros((K,len(F)))  # Considering rates are per second
    buffer_occ = self.buffers.buffer_occupancies
    for f in range(len(F)):
      se_freq = mimo_systems[f].SE_current_sample(curr_slot, alloc_users[f], self.se_ues_op)
      # se_freq = mimo_systems[f].SE_for_given_sample(curr_slot, alloc_users[f], F[f],
      #                                              avoid_estimation_errors=False)
      for iu, u in enumerate(alloc_users[f]):
        rate = (se_freq[iu] * mimo_systems[f].BW[f]) / packet_size_bits
        rates_pkt_per_s[iu][f] = np.minimum(rate, buffer_occ[iu])

    return rates_pkt_per_s

  def rateEstimationUsers(self, F: list, alloc_users: list, mimo_systems: list, K: int, curr_slot: int,
                          packet_size_bits: int) -> list:  # Calculates the rate per second for each user considering if it was selected to transmit in any frequency and
    rates_pkt_per_s = np.zeros((K,))  # Considering rates are per second

    for f in range(len(F)):
      se_freq = mimo_systems[f].SE_current_sample(curr_slot, alloc_users[f], self.se_ues_op)
      # se_freq = mimo_systems[f].SE_for_given_sample(curr_slot, alloc_users[f], F[f],
      #                                              avoid_estimation_errors=False)
      for iu, u in enumerate(alloc_users[f]):
        if rates_pkt_per_s[u] == 0:
          rates_pkt_per_s[u] += (se_freq[iu] * mimo_systems[f].BW[f]) / packet_size_bits

    return rates_pkt_per_s

  def updateSEUsers(self, F: list, alloc_users: list, mimo_systems: list, recent_spectral_eff: np.ndarray,
                    curr_slot: int):  # Function which update the Spectral efficiency value for each UE used in the last block
    for f in range(len(F)):
      SE = mimo_systems[f].SE_current_sample(curr_slot, alloc_users[f], self.se_ues_op)
      for iu, u in enumerate(alloc_users[f]):
        recent_spectral_eff[u, f] = SE[iu]
    return recent_spectral_eff

  def estimate_SE(self, F: list, alloc_users: list, mimo_systems: list, recent_spectral_eff: np.ndarray,
                    curr_slot: int):  # Function which update the Spectral efficiency value for each UE used in the last block

    sum_SE = 0.0
    SE = []

    for f in range(len(F)):
      au_SE = mimo_systems[f].SE_current_sample(curr_slot, alloc_users[f], self.se_ues_op)
      sum_SE += np.sum(au_SE)
      for iu, u in enumerate(alloc_users[f]):
        SE.append(au_SE[iu])

    return (sum_SE, SE)

  def estimate_SE_all(self):
    users = list(range(self.K))
    for f in range(len(self.F)):
      SE = self.mimo_systems[f].SE_current_sample(self.curr_block, users, self.se_ues_op)
      for iu, u in enumerate(users):
        self.spectral_eff[u, f] = SE[iu]
    self.full_eff.append(copy.deepcopy(self.spectral_eff))


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
    buffer_occupancies_norm = buffer_occupancies
    buffer_occupancies = np.array(buffer_occupancies * 100).astype(int)


    # Oldest packets
    oldest_packet_per_buffer = buffer_states[1]
    oldest_packet_per_buffer[
      oldest_packet_per_buffer > max_packet_age] = max_packet_age  # All values above threshold are set to the maximum value allowed
    oldest_packet_per_buffer_norm = oldest_packet_per_buffer / np.max(oldest_packet_per_buffer)  # Normalization

    if self.full_obs:
      # estimate individual SE per frequency
      self.estimate_SE_all()
      # spectral_eff = self.spectral_eff / np.max(self.spectral_eff)
      if self.use_mean_eff:
        #mean_eff = (self.max_spectral_eff / len(self.F)) * np.ones((self.K, len(self.F)))
        mean_eff = [[[],[]] for u in range(self.K)]
        me = [[[],[]] for u in range(self.K)]
        for fe in self.full_eff:
          for fek, vfek in enumerate(fe):
            for fekf, vfekf in enumerate(vfek):
              mean_eff[fek][fekf].append(vfekf)
        for ifk, fk in enumerate(mean_eff):
          for ifkf, fkf in enumerate(fk):
            me[ifk][ifkf] = np.mean(fkf)
        spectral_eff = np.array(me)
      else:
        spectral_eff = np.array(self.spectral_eff)
      se_flat = spectral_eff.flatten()
      se_flat_norm = se_flat / np.max(se_flat)
    else:
      spectral_eff = np.array(self.recent_spectral_eff)
      se_flat = spectral_eff.flatten()
      se_flat_norm = se_flat / np.max(se_flat)

    #return np.hstack(
    #  (buffer_occupancies, spectral_eff.flatten(), oldest_packet_per_buffer))  # oldest without normalization
    if self.use_se:
      if self.norm_obs:
        return np.hstack(
          (buffer_occupancies_norm, se_flat_norm, oldest_packet_per_buffer_norm))  # oldest with normalization
      return np.hstack((buffer_occupancies, se_flat, oldest_packet_per_buffer_norm))  # oldest with normalization
    else:
      return np.hstack((buffer_occupancies, oldest_packet_per_buffer_norm))  # oldest without normalization
    #return np.hstack(
    #  (buffer_occupancies_norm, se_flat_norm, oldest_packet_per_buffer_norm))  # normalized


  # calculation of packets transmission and reception (from transmit_and_receive_new_packets function)
  def pktFlow_(self, pkt_rate: float, buffers: Buffers, mimo_systems: list) -> (
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
    #self.last_incoming = np.sum(incoming_pkts)
    buffers.packets_arrival(incoming_pkts)  # Updating Buffer
    tx_pkts = np.sum(t_pkts)

    return (tx_pkts, dropped_pkts_sum, dropped_pkts, t_pkts, incoming_pkts, buffers, dropped_pkts_percent_mean)

  def pktFlow(self, pkt_rate: float, buffers: Buffers, mimo_systems: list) -> (
          float, float, float, float, float, list):
    present_flow = buffers.buffer_occupancies
    available_rate = np.floor(pkt_rate).astype(int)
    t_pkts = available_rate if (np.sum(available_rate) == 0) else buffers.packets_departure(
      available_rate)  # transmission pkts
    #incoming_pkts = mimo_systems[0].get_current_income_packets(pck_op=self.se_ues_op) # using the same SE operator factor to incomming traffic
    incoming_pkts = mimo_systems[0].get_current_income_packets()
    buffers.packets_arrival(incoming_pkts)  # Updating Buffer
    dropped_pkts = buffers.get_dropped_last_iteration()
    dropped_pkts_sum = np.sum(dropped_pkts)
    # getting the percentual dropped of the incoming packages
    dropped_pkts_percent = buffers.get_dropped_last_iteration_percent()
    dropped_pkts_percent_mean = np.mean(dropped_pkts_percent)

    #self.last_incoming = np.sum(incoming_pkts)

    tx_pkts = np.sum(t_pkts)

    return (tx_pkts, dropped_pkts_sum, dropped_pkts, t_pkts, incoming_pkts, buffers, dropped_pkts_percent_mean)

  def makeMIMO(self, F: list, K: int) -> list:  # Generating MIMO system for each user
    return [MassiveMIMOSystem(K=K, frequency_index=f + 1, tf_folder=self.tf_folder) for f in range(len(F))]

  def loadEpMIMO(self, mimo_systems: list, F: list, tp: int) -> list:
    '''
    Load static channel data file. tp define the running type: 0 - training; 1- validating
    '''
    # calculate the matrices to generate channel estimates
    # episode_number = np.random.randint(0, self.mimo_systems[0].num_episodes - 1)
    if self.running_tp == 0:
      self.episode_number = np.random.randint(self.mimo_systems[0].range_episodes_train[0],
                                         self.mimo_systems[0].range_episodes_validate[1])
    else:
      self.episode_number = np.random.randint(self.mimo_systems[0].range_episodes_validate[0],
                                         self.mimo_systems[0].range_episodes_validate[1])
    if self.pred_eps:
      self.episode_number = self.pred_eps[self.pred_eps_count]
      self.pred_eps_count += 1
      if self.pred_eps_count > len(self.pred_eps)-1:
        self.pred_eps_count = 0
    #if self.running_tp == 1:
    #  print("Episode {}".format(self.episode_number))
    #if self.running_tp == 0:

    #  self.episode_number = choices(range(self.mimo_systems[0].num_episodes), self.eps_prob)[0]

    #  for e in range(self.mimo_systems[0].num_episodes):
    #    if e != self.episode_number:
    #      self.eps_count[e] += 1
    #  s = sum(self.eps_count)
    #  self.eps_prob = [self.eps_count[e]/s for e in range(self.mimo_systems[0].num_episodes)]

    #else:
    #  self.episode_number = np.random.randint(self.mimo_systems[0].range_episodes_validate[0],
    #                                       self.mimo_systems[0].range_episodes_validate[1])

    #self.episode_number = episode_number
    # if tp == 0:
    #  if (self.episode_number + 1) > (self.mimo_systems[0].num_episodes - 1):
    #    self.episode_number = np.random.randint(0, self.mimo_systems[0].num_episodes - 1)
    #  else:
    #    self.episode_number += 1
    # else:
    #  self.episode_number = np.random.randint(self.mimo_systems[0].range_episodes_validate[0],
    #                                        self.mimo_systems[0].range_episodes_validate[1])

    for f in range(len(F)):
      self.mimo_systems[f].set_tf_folder(self.tf_folder)
      # same episode number for all frequencies
      self.mimo_systems[f].load_episode(self.episode_number)
    return mimo_systems

  def rewardCalc_(self, F: list, alloc_users: list, mimo_systems: list, K: int, curr_slot: int, packet_size_bits: int,
                  pkt_rate, buffers: Buffers, min_reward: int, SE: list, update) -> float:  # Calculating reward value

    (tx_pkts, dropped_pkts_sum, dropped_pkts, t_pkts, incoming_pkts, buffers,
     dropped_pkts_percent_mean) = self.pktFlow(pkt_rate[0],
                                               buffers,
                                               mimo_systems)
    reward = (tx_pkts * self.packet_size_bits) / 1e6

    individual_thp = [((i * self.packet_size_bits) / 1e6) for i in t_pkts]

    return reward, dropped_pkts_sum, dropped_pkts_percent_mean, individual_thp