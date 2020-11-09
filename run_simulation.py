import numpy as np
import matplotlib.pyplot as plt
from stable_baselines import ACKTR, A2C
from stable_baselines.common.policies import MlpPolicy

import consts
import copy

from schedulers.max_th import MaxTh
from schedulers.proportional_fair import ProportionalFair
from schedulers.round_robin import RoundRobin
from sra_env.sra_env import SRAEnv

env = SRAEnv()

#model = ACKTR(MlpPolicy, env, verbose=1, n_steps=100)
#model.load('acktr_drl')
#model = ACKTR.load('acktr_drl')
model = A2C.load("a2c_drl_300k")

env.reset()
obs, rw, endep, info = env.step_(0)
obs = env.reset()
# Creating the schedulers
# creating the Round Robin scheduler instance with independent buffers
env.schedulers.append(RoundRobin(K=consts.K, F=consts.F, buffers=copy.deepcopy(env.buffers)))
# another scheduler instance, for testing with multiple schedulers
#schedulers.append(ProportionalFair(K=consts.K, F=consts.F, buffers=copy.deepcopy(env.buffers)))
#schedulers.append(MaxTh(K=consts.K, F=consts.F, buffers=copy.deepcopy(env.buffers)))

rewards_drl_agent = []
reward_schedulers = [[] for i in range(len(env.schedulers))]
pkt_loss = []
actions = []
#self.observation_space, [reward, reward_schedulers, pkt_loss], self.end_ep, info

for i in range(100):
    rw_drl_a = []
    rw_sh = []
    pkt_l = []
    for ii in range(env.blocks_ep):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step_(action)
        rw_drl_a.append(rewards[0])
        rw_sh.append(rewards[1])
        pkt_l.append(rewards[2])
    #env.render()
    rewards_drl_agent.append(np.mean(rw_drl_a))
    support_rw = [[] for i in range(len(env.schedulers))]
    for i,v in enumerate(env.schedulers):
        for ii in rw_sh:
            support_rw[i].append(ii[i])
    for i, v in enumerate(env.schedulers):
        for ii,vv in enumerate(support_rw):
            reward_schedulers[i].append(np.mean(vv))
    pkt_loss.append(pkt_l)
print(rewards_drl_agent)

figure, axis = plt.subplots(2, 2)
axis[0,0].plot(rewards_drl_agent, label='DRL-SRA')
for i,v in enumerate(env.schedulers):
    axis[0,0].plot(reward_schedulers[i], label=v.name)
axis[0,0].set_xlabel('Episodes')
axis[0,0].set_ylabel('Mean Sum-rate')
axis[0,0].set_title('Sum-rate per episode')
axis[0,0].legend()

plt.show()

