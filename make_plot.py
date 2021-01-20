import numpy as np
from numpy import trapz
import matplotlib.pyplot as plt

import math
import csv
import json
import consts
from schedulers.max_th import MaxTh
from schedulers.proportional_fair import ProportionalFair
from schedulers.round_robin import RoundRobin
from utils.compute_area import ComputeArea
from utils.save_results import SaveResults

LR = 0.007
LR_D= '007'
tqdm_ = "10-100000"
t = 500
simulation_type = "stationary"
simulation_type = "n-stationary"
#F = "_F_3-3_NR"
F = "_F_2-2_ME" # LE = Less Training Episode data = 30 episodes
F = "_F_1-1_ME" # LE = Less Training Episode data = 30 episodes - ME = More TE = 100
#F = ""
#F = "_F_3-3_ME_TI_2" # LE = Less Training Episode data = 30 episodes - ME = More TE = 100 - TI traffic int
# loading the results

f = 'history_final/'+simulation_type+ F + '_history_'+ tqdm_ +'_'+str(t)+'_rounds_'+str(consts.BLOCKS_EP)+\
    '_bloks_eps_lr_'+ LR_D +'.json'
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

pkt_delay_sch = history['pkt_delay_schedulers']

tss = history['tss']
#############################

schedulers =[]
schedulers.append(RoundRobin(K=0, F=[0], buffers=None))
schedulers.append(ProportionalFair(K=0, F=[0], buffers=None))
schedulers.append(MaxTh(K=0, F=[0], buffers=None))

plot_data = {}

figure, axis = plt.subplots(1)
if len(rewards_drl_agent_1) > 0 and ~math.isnan(rewards_drl_agent_1[0]):
    #axis.plot(tss, rewards_drl_agent_1, label=r'DRL-SRA$_{A2C}$')
    plot_data[ComputeArea.auc(x=tss,y=rewards_drl_agent_1)] = [tss, rewards_drl_agent_1, r'DRL-SRA$_{A2C}$', '-', '^', 'r']
    #print(ComputeArea.auc(x=tss,y=rewards_drl_agent_1))
if len(rewards_drl_agent_2) > 0 and not math.isnan(rewards_drl_agent_2[0]):
    #axis.plot(tss, rewards_drl_agent_2, label=r'DRL-SRA$_{ACKTR}$')
    plot_data[ComputeArea.auc(x=tss, y=rewards_drl_agent_2)] = [tss, rewards_drl_agent_2, r'DRL-SRA$_{ACKTR}$', '--', '^', 'b']
if len(rewards_drl_agent_3) > 0 and not math.isnan(rewards_drl_agent_3[0]):
    #axis.plot(tss, rewards_drl_agent_3, label=r'DRL-SRA$_{TRPO}$')
    plot_data[ComputeArea.auc(x=tss, y=rewards_drl_agent_3)] = [tss, rewards_drl_agent_3, r'DRL-SRA$_{TRPO}$', '--', '^', 'b']
if len(rewards_drl_agent_4) > 0 and not math.isnan(rewards_drl_agent_4[0]):
    #axis.plot(tss, rewards_drl_agent_4, label=r'DRL-SRA$_{ACER}$')
    plot_data[ComputeArea.auc(x=tss, y=rewards_drl_agent_4)] = [tss, rewards_drl_agent_4, r'DRL-SRA$_{ACER}$', '--', '^', 'b']
if len(rewards_drl_agent_5) > 0 and not math.isnan(rewards_drl_agent_5[0]):
    #axis.plot(tss, rewards_drl_agent_5, label=r'DRL-SRA$_{DQN}$')
    plot_data[ComputeArea.auc(x=tss, y=rewards_drl_agent_5)] = [tss, rewards_drl_agent_5, r'DRL-SRA$_{DQN}$', '-', 'x', 'g']
    #print(ComputeArea.auc(x=tss, y=rewards_drl_agent_5))
if len(rewards_drl_agent_6) > 0 and not math.isnan(rewards_drl_agent_6[0]):
    #axis.plot(tss, rewards_drl_agent_6, label=r'DRL-SRA$_{PPO1}$')
    plot_data[ComputeArea.auc(x=tss, y=rewards_drl_agent_6)] = [tss, rewards_drl_agent_6, r'DRL-SRA$_{PPO1}$', '-', 's', 'm']
    #print(ComputeArea.auc(x=tss, y=rewards_drl_agent_6))
if len(rewards_drl_agent_7) > 0 and not math.isnan(rewards_drl_agent_7[0]):
    #axis.plot(tss, rewards_drl_agent_7, label=r'DRL-SRA$_{PPO2}$')
    plot_data[ComputeArea.auc(x=tss, y=rewards_drl_agent_7)] = [tss, rewards_drl_agent_7, r'DRL-SRA$_{PPO2}$', '--', '^', 'b']

plot_colors = ['k', 'c', 'b']
plot_marks = ['o', 'p', '+']
plot_lines = ['--', '-.', '--']

for i,v in enumerate(schedulers):
    vrs = []
    for rs in reward_schedulers:
        vrs.append(rs[i][0])
    #axis.plot(tss, vrs, linestyle='dashdot', label=v.name)
    plot_data[ComputeArea.auc(x=tss, y=vrs)] = [tss, vrs, v.name, plot_lines[i], plot_marks[i], plot_colors[i]]

# sortind plot data in order to get a plot with desc ordering
plot_data_sorted = sorted(plot_data.items(),key=lambda x: x[0], reverse=True)
# plotting sum-rate plot
for p in plot_data_sorted:
    axis.plot(tss, p[1][1], label=p[1][2], linestyle=p[1][3], marker=p[1][4], color=p[1][5])

axis.set_xlabel('Training episodes')
axis.set_ylabel('Mean sum-rate (Mbps)')
#axis.set_title('Sum-rate per episode')
axis.legend(loc=7)

plot_data = {}
figure, axis2 = plt.subplots(1)

if len(pkt_loss_1) > 0 and not math.isnan(pkt_loss_1[0]):
    #axis2.plot(tss, pkt_loss_1, label=r'DRL-SRA$_{A2C}$')
    plot_data[ComputeArea.auc(x=tss, y=pkt_loss_1)] = [tss, pkt_loss_1, r'DRL-SRA$_{A2C}$', '-', '^','r']
if len(pkt_loss_2) > 0 and not math.isnan(pkt_loss_2[0]):
    #axis2.plot(tss, pkt_loss_2, label=r'DRL-SRA$_{ACKTR}$')
    plot_data[ComputeArea.auc(x=tss, y=pkt_loss_2)] = [tss, pkt_loss_2, r'DRL-SRA$_{ACKTR}$', '--','^', 'b']
if len(pkt_loss_3) > 0 and not math.isnan(pkt_loss_3[0]):
    #axis2.plot(tss, pkt_loss_3, label=r'DRL-SRA$_{TRPO}$')
    plot_data[ComputeArea.auc(x=tss, y=pkt_loss_3)] = [tss, pkt_loss_3, r'DRL-SRA$_{TRPO}$', '--','^', 'b']
if len(pkt_loss_4) > 0 and not math.isnan(pkt_loss_4[0]):
    #axis2.plot(tss, pkt_loss_4, label=r'DRL-SRA$_{ACER}$')
    plot_data[ComputeArea.auc(x=tss, y=pkt_loss_4)] = [tss, pkt_loss_4, r'DRL-SRA$_{ACER}$', '--', '^', 'b']
if len(pkt_loss_5) > 0 and not math.isnan(pkt_loss_5[0]):
    #axis2.plot(tss, pkt_loss_5, label=r'DRL-SRA$_{DQN}$')
    plot_data[ComputeArea.auc(x=tss, y=pkt_loss_5)] = [tss, pkt_loss_5, r'DRL-SRA$_{DQN}$', '-', 'x', 'g']
if len(pkt_loss_6) > 0 and not math.isnan(pkt_loss_6[0]):
    #axis2.plot(tss, pkt_loss_6, label=r'DRL-SRA$_{PPO1}$')
    plot_data[ComputeArea.auc(x=tss, y=pkt_loss_6)] = [tss, pkt_loss_6, r'DRL-SRA$_{PPO1}$', '-', 's', 'm']
if len(pkt_loss_7) > 0 and not math.isnan(pkt_loss_7[0]):
    #axis2.plot(tss, pkt_loss_7, label=r'DRL-SRA$_{PPO2}$')
    plot_data[ComputeArea.auc(x=tss, y=pkt_loss_7)] = [tss, pkt_loss_7, r'DRL-SRA$_{PPO2}$', '--', '^', 'b']

for i,v in enumerate(schedulers):
    #axis2.plot(tss, pkt_loss_sch[i], linestyle='dashdot', label=v.name)
    plot_data[ComputeArea.auc(x=tss, y=pkt_loss_sch[i])] = [tss, pkt_loss_sch[i], v.name, plot_lines[i], plot_marks[i], plot_colors[i]]

# sortind plot data in order to get a plot with desc ordering
plot_data_sorted = sorted(plot_data.items(),key=lambda x: x[0], reverse=True)
# plotting sum-rate plot
for p in plot_data_sorted:
    axis2.plot(tss, p[1][1], label=p[1][2], linestyle=p[1][3], marker=p[1][4], color=p[1][5])

axis2.set_xlabel('Training episodes')
axis2.set_ylabel('Mean packet loss')
#axis2.set_title('Packet loss per episode')
axis2.legend(loc=7)

figure, axis3 = plt.subplots(1)

pkt_d_1 = history['pkt_delay_agents']['A2C']
pkt_d_2 = history['pkt_delay_agents']['ACKTR']
pkt_d_3 = history['pkt_delay_agents']['TRPO']
pkt_d_4 = history['pkt_delay_agents']['ACER']
pkt_d_5 = history['pkt_delay_agents']['DQN']
pkt_d_6 = history['pkt_delay_agents']['PPO1']
pkt_d_7 = history['pkt_delay_agents']['PPO2']

if len(pkt_d_1) > 0 and not math.isnan(pkt_d_1[0]):
    axis3.plot(tss, pkt_d_1, label=r'DRL-SRA$_{A2C}$')
if len(pkt_d_2) > 0 and not math.isnan(pkt_d_2[0]):
    axis3.plot(tss, pkt_d_2, label=r'DRL-SRA$_{ACKTR}$')
if len(pkt_d_3) > 0 and not math.isnan(pkt_d_3[0]):
    axis3.plot(tss, pkt_d_3, label=r'DRL-SRA$_{TRPO}$')
if len(pkt_d_4) > 0 and not math.isnan(pkt_d_4[0]):
    axis3.plot(tss, pkt_d_4, label=r'DRL-SRA$_{ACER}$')
if len(pkt_d_5) > 0 and not math.isnan(pkt_d_5[0]):
    axis3.plot(tss, pkt_d_5, label=r'DRL-SRA$_{DQN}$')
if len(pkt_d_6) > 0 and not math.isnan(pkt_d_6[0]):
    axis3.plot(tss, pkt_d_6, label=r'DRL-SRA$_{PPO1}$')
if len(pkt_d_7) > 0 and not math.isnan(pkt_d_7[0]):
    axis3.plot(tss, pkt_d_7, label=r'DRL-SRA$_{PPO2}$')

for i,v in enumerate(schedulers):
    axis3.plot(tss, pkt_delay_sch[i], linestyle='dashdot', label=v.name)

axis3.set_xlabel('Training episodes')
axis3.set_ylabel('Mean packet delay')
#axis3.set_title('Packet delay per episode')
axis3.legend(loc=7)

plt.show()