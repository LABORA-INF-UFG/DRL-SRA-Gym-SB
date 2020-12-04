import numpy as np
import matplotlib.pyplot as plt

import csv
import json
import consts
from schedulers.max_th import MaxTh
from schedulers.proportional_fair import ProportionalFair
from schedulers.round_robin import RoundRobin
from utils.save_results import SaveResults


tqdm_ = "0-200"
t = 100
simulation_type = "stationary"
simulation_type = "n-stationary"


# loading the results

f = 'history/'+simulation_type+'_history_'+ tqdm_ +'_'+str(t)+'_rounds_'+str(consts.BLOCKS_EP)+'_bloks_eps.json'
history = SaveResults.load_history(f)


rewards_drl_agent_1 = history['rewards_drl_agents']['A2C']
rewards_drl_agent_2 = history['rewards_drl_agents']['ACKTR']
rewards_drl_agent_3 = history['rewards_drl_agents']['TRPO']
rewards_drl_agent_4 = history['rewards_drl_agents']['ACER']
rewards_drl_agent_5 = history['rewards_drl_agents']['DQN']
rewards_drl_agent_6 = history['rewards_drl_agents']['PPO1']
rewards_drl_agent_7 = history['rewards_drl_agents']['PPO2']

reward_schedulers = history['rewards_schedulers']
pkt_loss_1 = history['pkt_loss_agents']['A2C']
pkt_loss_2 = history['pkt_loss_agents']['ACKTR']
pkt_loss_3 = history['pkt_loss_agents']['TRPO']
pkt_loss_4 = history['pkt_loss_agents']['ACER']
pkt_loss_5 = history['pkt_loss_agents']['DQN']
pkt_loss_6 = history['pkt_loss_agents']['PPO1']
pkt_loss_7 = history['pkt_loss_agents']['PPO2']

pkt_loss_sch = history['pkt_loss_schedulers']

tss = history['tss']
#############################

schedulers =[]
schedulers.append(RoundRobin(K=0, F=[0], buffers=None))
schedulers.append(ProportionalFair(K=0, F=[0], buffers=None))
schedulers.append(MaxTh(K=0, F=[0], buffers=None))

figure, axis = plt.subplots(1)
axis.plot(tss, rewards_drl_agent_1, label=r'DRL-SRA$_{A2C}$')
axis.plot(tss, rewards_drl_agent_2, label=r'DRL-SRA$_{ACKTR}$')
axis.plot(tss, rewards_drl_agent_3, label=r'DRL-SRA$_{TRPO}$')
axis.plot(tss, rewards_drl_agent_4, label=r'DRL-SRA$_{ACER}$')
axis.plot(tss, rewards_drl_agent_5, label=r'DRL-SRA$_{DQN}$')
axis.plot(tss, rewards_drl_agent_6, label=r'DRL-SRA$_{PPO1}$')
axis.plot(tss, rewards_drl_agent_7, label=r'DRL-SRA$_{PPO2}$')

for i,v in enumerate(schedulers):
    vrs = []
    for rs in reward_schedulers:
        vrs.append(rs[i])
    axis.plot(tss, vrs, linestyle='dashdot', label=v.name)

axis.set_xlabel('Episodes')
axis.set_ylabel('Mean Sum-rate')
axis.set_title('Sum-rate per episode')
axis.legend()

figure, axis2 = plt.subplots(1)

axis2.plot(tss, pkt_loss_1, label=r'DRL-SRA$_{A2C}$')
axis2.plot(tss, pkt_loss_2, label=r'DRL-SRA$_{ACKTR}$')
axis2.plot(tss, pkt_loss_3, label=r'DRL-SRA$_{TRPO}$')
axis2.plot(tss, pkt_loss_4, label=r'DRL-SRA$_{ACER}$')
axis2.plot(tss, pkt_loss_5, label=r'DRL-SRA$_{DQN}$')
axis2.plot(tss, pkt_loss_6, label=r'DRL-SRA$_{PPO1}$')
axis2.plot(tss, pkt_loss_7, label=r'DRL-SRA$_{PPO2}$')

for i,v in enumerate(schedulers):
    axis2.plot(tss, pkt_loss_sch[i], linestyle='dashdot', label=v.name)

axis2.set_xlabel('Episodes')
axis2.set_ylabel('Mean pkt loss')
axis2.set_title('Packet loss per episode')
axis2.legend()

plt.show()