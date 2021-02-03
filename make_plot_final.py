from tqdm import tqdm
import consts
import matplotlib.pyplot as plt

from schedulers.max_th import MaxTh
from schedulers.proportional_fair import ProportionalFair
from schedulers.round_robin import RoundRobin
from utils.compute_area import ComputeArea
from utils.save_results import SaveResults
import numpy as np
from utils.tools import Tools

'''
Make the plots for run_simulation_final.py computing mean by UE, computing the Interval Confidence (disabled)
'''
def empty(seq):
    try:
        return all(map(empty, seq))
    except TypeError:
        return False

schedulers =[]
schedulers.append(RoundRobin(K=0, F=[0], buffers=None))
schedulers.append(ProportionalFair(K=0, F=[0], buffers=None))
schedulers.append(MaxTh(K=0, F=[0], buffers=None))

plot_colors = ['k', 'c', 'b']
plot_marks = ['o', 'p', '+']
plot_lines = ['--', '-.', '--']
sch_names = ['Round robin', 'Proportional fair', 'Max th']

tqdm_ = "10-101000"
#tqdm_ = "10-26000"
t = 50 # avaliation rounds/episodes
#simulation_type = "stationary"
simulation_type = "n-stationary"
F = "_F_" + consts.F_D + "_ME_TI_low1"
F = "_F_" + consts.F_D + "_high"
F = "_F_" + consts.F_D + "_low"
#number of schedulers
n_sch = 3
tss = []

rewards_drl = []
rewards_sch = [[] for i in range(n_sch)]
pkt_loss_drl = []
pkt_loss_sch = [[] for i in range(n_sch)]
pkt_d_drl = []
pkt_d_sch = [[] for i in range(n_sch)]

#n-stationary_F_2-2_ME_history_full_10-100000_2_rounds_100_bloks_eps_lr_007
#n-stationary_F_2-2_ME_history_full_10-100000_2_rounds_100_bloks_eps_lr_007.json
#f = 'history_final/' + simulation_type + F + "_history_full_10-101000_" + str(t) + '_rounds_' + str(consts.BLOCKS_EP) + '_bloks_eps_lr_' + consts.LR_D + '.json'
f = 'history_final/' + simulation_type + F + "_history_full_10-30000_" + str(t) + '_rounds_' + str(consts.BLOCKS_EP) + '_bloks_eps_lr_' + consts.LR_D + '.json'
tqdm_ = "10-30000"

history = SaveResults.load_history(f)

## getting the plot data
## rewards for DRL agents
p_rewards_drl_agent_1 = history['rewards_drl_agents']['A2C']
p_rewards_drl_agent_2 = history['rewards_drl_agents']['ACKTR']
p_rewards_drl_agent_3 = history['rewards_drl_agents']['TRPO']
p_rewards_drl_agent_4 = history['rewards_drl_agents']['ACER']
p_rewards_drl_agent_5 = history['rewards_drl_agents']['DQN']
p_rewards_drl_agent_6 = history['rewards_drl_agents']['PPO1']
p_rewards_drl_agent_7 = history['rewards_drl_agents']['PPO2']

tss = history['tss']

#############################
plot_data = {}
plot_data_ci = {}

for i,v in enumerate(schedulers):
    for vi in history['rewards_schedulers']:
        rewards_sch[i].append(np.mean(vi[i])/consts.K)
    plot_data[ComputeArea.auc(x=tss, y=rewards_sch[i])] = [tss, rewards_sch[i], v.name, plot_lines[i], plot_marks[i], plot_colors[i]]

if not empty(p_rewards_drl_agent_1):
    rewards_drl_agent_1 = []
    rewards_drl_ci_1 = []
    for i,v in enumerate(p_rewards_drl_agent_1):
        rewards_drl_agent_1.append(np.mean(v)/consts.K)
        vv = np.mean(v,axis=1)
        # computing the confidence interval
        rewards_drl_ci_1.append(Tools.mean_confidence_interval(vv))
    plot_data[ComputeArea.auc(x=tss, y=rewards_drl_agent_1)] = [tss, rewards_drl_agent_1, r'DRL-SRA$_{A2C}$', '-', '^', 'r']

if not empty(p_rewards_drl_agent_2):
    rewards_drl_agent_2 = []
    rewards_drl_ci_2 = []
    for i,v in enumerate(p_rewards_drl_agent_2):
        rewards_drl_agent_2.append(np.mean(v)/consts.K)
        vv = np.mean(v,axis=1)
        # computing the confidence interval
        rewards_drl_ci_2.append(Tools.mean_confidence_interval(vv))
    plot_data[ComputeArea.auc(x=tss, y=rewards_drl_agent_2)] = [tss, rewards_drl_agent_2, r'DRL-SRA$_{ACKTR}$', '--','^', 'b']

if not empty(p_rewards_drl_agent_3):
    rewards_drl_agent_3 = []
    rewards_drl_ci_3 = []
    for i,v in enumerate(p_rewards_drl_agent_3):
        rewards_drl_agent_3.append(np.mean(v)/consts.K)
        vv = np.mean(v,axis=1)
        # computing the confidence interval
        rewards_drl_ci_3.append(Tools.mean_confidence_interval(vv))
    plot_data[ComputeArea.auc(x=tss, y=rewards_drl_agent_3)] = [tss, rewards_drl_agent_3, r'DRL-SRA$_{TRPO}$', '--','^', 'b']

if not empty(p_rewards_drl_agent_3):
    rewards_drl_agent_3 = []
    rewards_drl_ci_3 = []
    for i,v in enumerate(p_rewards_drl_agent_3):
        rewards_drl_agent_3.append(np.mean(v)/consts.K)
        vv = np.mean(v,axis=1)
        # computing the confidence interval
        rewards_drl_ci_3.append(Tools.mean_confidence_interval(vv))
    plot_data[ComputeArea.auc(x=tss, y=rewards_drl_agent_3)] = [tss, rewards_drl_agent_3, r'DRL-SRA$_{TRPO}$', '--','^', 'b']

if not empty(p_rewards_drl_agent_4):
    rewards_drl_agent_4 = []
    rewards_drl_ci_4 = []
    for i,v in enumerate(p_rewards_drl_agent_4):
        rewards_drl_agent_4.append(np.mean(v)/consts.K)
        vv = np.mean(v,axis=1)
        # computing the confidence interval
        rewards_drl_ci_4.append(Tools.mean_confidence_interval(vv))
    plot_data[ComputeArea.auc(x=tss, y=rewards_drl_agent_4)] = [tss, rewards_drl_agent_4, r'DRL-SRA$_{ACER}$', '--','^', 'b']

if not empty(p_rewards_drl_agent_5):
    rewards_drl_agent_5 = []
    rewards_drl_ci_5 = []
    for i,v in enumerate(p_rewards_drl_agent_5):
        rewards_drl_agent_5.append(np.mean(v)/consts.K)
        vv = np.mean(v,axis=1)
        # computing the confidence interval
        rewards_drl_ci_5.append(Tools.mean_confidence_interval(vv))
    plot_data[ComputeArea.auc(x=tss, y=rewards_drl_agent_5)] = [tss, rewards_drl_agent_5, r'DRL-SRA$_{DQN}$', '-', 'x', 'g']

if not empty(p_rewards_drl_agent_6):
    rewards_drl_agent_6 = []
    rewards_drl_ci_6 = []
    for i,v in enumerate(p_rewards_drl_agent_6):
        rewards_drl_agent_6.append(np.mean(v)/consts.K)
        vv = np.mean(v,axis=1)
        # computing the confidence interval
        rewards_drl_ci_6.append(Tools.mean_confidence_interval(vv))
    plot_data[ComputeArea.auc(x=tss, y=rewards_drl_agent_6)] = [tss, rewards_drl_agent_6, r'DRL-SRA$_{PPO1}$', '-', 's', 'm']

if not empty(p_rewards_drl_agent_7):
    rewards_drl_agent_7 = []
    rewards_drl_ci_7 = []
    for i,v in enumerate(p_rewards_drl_agent_7):
        rewards_drl_agent_7.append(np.mean(v)/consts.K)
        vv = np.mean(v,axis=1)
        # computing the confidence interval
        rewards_drl_ci_7.append(Tools.mean_confidence_interval(vv))
    plot_data[ComputeArea.auc(x=tss, y=rewards_drl_agent_7)] = [tss, rewards_drl_agent_7, r'DRL-SRA$_{PPO2}$', '--','^', 'b']

# sorting plot data in order to get a plot with desc ordering
plot_data_sorted = sorted(plot_data.items(),key=lambda x: x[0], reverse=True)

# plotting

# first: throughput
figure, axis = plt.subplots(1)

# plotting sum-rate plot
for p in plot_data_sorted:
    axis.plot(tss, p[1][1], label=p[1][2], linestyle=p[1][3], marker=p[1][4], color=p[1][5])

# plotting confidence interval
# if rewards_drl_ci_1:
#     axis.fill_between(tss, np.array(rewards_drl_ci_1)[:,1] / 10, np.array(rewards_drl_ci_1)[:,2] / 10, color='r', alpha=0.2)
# if not empty(p_rewards_drl_agent_5):
#     if rewards_drl_ci_5:
#         axis.fill_between(tss, np.array(rewards_drl_ci_5)[:,1], np.array(rewards_drl_ci_5)[:,2], color='g', alpha=0.2)
# if not empty(p_rewards_drl_agent_6):
#     if rewards_drl_ci_6:
#         axis.fill_between(tss, np.array(rewards_drl_ci_6)[:, 1], np.array(rewards_drl_ci_6)[:, 2], color='m', alpha=0.2)

axis.set_xlabel('Training episodes')
axis.set_ylabel('Mean user throughput (Mbps)')
#axis.set_title('Sum-rate per episode')
axis.legend(loc=7)

## plotting packet loss
plot_data = {}
plot_data_ci = {}

for i,v in enumerate(schedulers):
    for vi in history['pkt_loss_schedulers']:
        pkt_loss_sch[i].append(np.mean(vi[i]))
    plot_data[ComputeArea.auc(x=tss, y=pkt_loss_sch[i])] = [tss, pkt_loss_sch[i], v.name, plot_lines[i], plot_marks[i], plot_colors[i]]



p_pkt_loss_1 = history['pkt_loss_agents']['A2C']
p_pkt_loss_5 = history['pkt_loss_agents']['DQN']
p_pkt_loss_6 = history['pkt_loss_agents']['PPO1']

if not empty(p_pkt_loss_1):
    pkt_loss_1 = []
    p_pkt_loss_ci_1 = []
    for i,v in enumerate(p_pkt_loss_1):
        pkt_loss_1.append(np.mean(v))
        vv = np.mean(Tools.matrix_to_vector(v),axis=1)
        # computing the confidence interval
        p_pkt_loss_ci_1.append(Tools.mean_confidence_interval(vv))
    plot_data[ComputeArea.auc(x=tss, y=pkt_loss_1)] = [tss, pkt_loss_1, r'DRL-SRA$_{A2C}$', '-', '^', 'r']

if not empty(p_pkt_loss_5):
    pkt_loss_5 = []
    p_pkt_loss_ci_5 = []
    for i,v in enumerate(p_pkt_loss_5):
        pkt_loss_5.append(np.mean(v))
        vv = np.mean(Tools.matrix_to_vector(v),axis=1)
        # computing the confidence interval
        p_pkt_loss_ci_5.append(Tools.mean_confidence_interval(vv))
    plot_data[ComputeArea.auc(x=tss, y=pkt_loss_5)] = [tss, pkt_loss_5, r'DRL-SRA$_{DQN}$', '-', 'x', 'g']

if not empty(p_pkt_loss_6):
    pkt_loss_6 = []
    p_pkt_loss_ci_6 = []
    for i,v in enumerate(p_pkt_loss_6):
        pkt_loss_6.append(np.mean(v))
        vv = np.mean(Tools.matrix_to_vector(v),axis=1)
        # computing the confidence interval
        p_pkt_loss_ci_6.append(Tools.mean_confidence_interval(vv))
    plot_data[ComputeArea.auc(x=tss, y=pkt_loss_6)] = [tss, pkt_loss_6, r'DRL-SRA$_{PPO1}$', '-', 's', 'm']

# sorting plot data in order to get a plot with desc ordering
plot_data_sorted = sorted(plot_data.items(),key=lambda x: x[0], reverse=True)

figure, axis2 = plt.subplots(1)

# plotting packet loss
for p in plot_data_sorted:
    axis2.plot(tss, p[1][1], label=p[1][2], linestyle=p[1][3], marker=p[1][4], color=p[1][5])

# plotting confidence interval
#if p_pkt_loss_ci_1:
#    axis2.fill_between(tss, np.array(p_pkt_loss_ci_1)[:,1], np.array(p_pkt_loss_ci_1)[:,2], color='r', alpha=0.2)

axis2.set_xlabel('Training episodes')
axis2.set_ylabel('Mean user packet loss (%)')
#axis2.set_title('Packet loss per episode')
axis2.legend(loc=7)

plt.show()
