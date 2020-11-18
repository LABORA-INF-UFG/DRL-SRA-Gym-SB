import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines import ACKTR, A2C
from stable_baselines.common.policies import MlpPolicy
from tqdm import tqdm

import consts
import copy

from schedulers.max_th import MaxTh
from schedulers.proportional_fair import ProportionalFair
from schedulers.round_robin import RoundRobin
from sra_env.sra_env import SRAEnv

env1 = SRAEnv()
env1.type = "Master"



tqdm_e = tqdm(range(0,20,5), desc='Time Steps', leave=True, unit=" time steps")
obs, rw, endep, info = env1.step_(0)
obs2 = copy.deepcopy(obs)
env2 = copy.deepcopy(env1)
env2.type = "Slave"

env2.par_envs['Master'].append(env1)

rewards_drl_agent_1, rewards_drl_agent_2 = [], []
reward_schedulers_1 = [[] for i in range(len(env1.schedulers))]
reward_schedulers_2 = copy.deepcopy(reward_schedulers_1)
pkt_loss_1, pkt_loss_2 = [], []
actions_1, actions_2 = [], []
c_act_1, c_act_2 = [0] * len(env1.schedulers), [0] * len(env1.schedulers)
tss = []

for i in tqdm_e:
    #i = 1 if i == 0 else i
    #ts = consts.BLOCKS_EP * i
    ts = consts.BLOCKS_EP * (i + 1)
    tss.append(ts)
    ## Loading the trained model for A2C
    model1 = A2C.load("models/a2c_drl_"+str(ts))
    ## Loading the trained model for A2C
    model2 = ACKTR.load("models/acktr_"+str(ts))

    p_rewards_drl_agent_1, p_rewards_drl_agent_2 = [], []
    p_reward_schedulers_1 = [[] for i in range(len(env1.schedulers))]
    p_reward_schedulers_2 = copy.deepcopy(p_reward_schedulers_1)
    p_pkt_loss_1, p_pkt_loss_2 = [], []

    for i in range(2):
        rw_drl_a_1 = []
        rw_drl_a_2 = []
        rw_sh_1 = []
        rw_sh_2 = []
        pkt_l_drl_a_1 = []
        pkt_l_drl_a_2 = []
        pkt_l_sh_1 = []
        pkt_l_sh_2 = []
        acts_1 = []
        acts_2 = []
        for ii in range(env1.blocks_ep):
            action1, _ = model1.predict(obs)
            acts_1.append(action1)
            obs, rewards_1, _, _ = env1.step_(action1)
            action2, _ = model2.predict(obs2)
            acts_2.append(action2)
            obs2, rewards_2, _, _ = env2.step_(action2)

            # for model1
            rw_drl_a_1.append(rewards_1[0])
            rw_sh_1.append(rewards_1[1])
            # the last one is the drl agent pkt loss
            pkt_l_drl_a_1.append(rewards_1[2][len(rewards_1[2]) - 1])
            # for model2
            rw_drl_a_2.append(rewards_2[0])
            rw_sh_2.append(rewards_2[1])
            # the last one is the drl agent pkt loss
            pkt_l_drl_a_2.append(rewards_2[2][len(rewards_2[2]) - 1])

        # env.render()
        p_rewards_drl_agent_1.append(np.mean(rw_drl_a_1))
        p_rewards_drl_agent_2.append(np.mean(rw_drl_a_2))
        p_support_rw_1 = [[] for i in range(len(env1.schedulers))]
        p_support_rw_2 = [[] for i in range(len(env1.schedulers))]
        #for i, v in enumerate(env1.schedulers):
        #    for ii in rw_sh_1:
        #        support_rw_1[i].append(ii[i])
        #    for ii in rw_sh_2:
        #        support_rw_2[i].append(ii[i])
        #for i, v in enumerate(env1.schedulers):
        #    for ii, vv in enumerate(support_rw_1):
        #        reward_schedulers_1[i].append(np.mean(vv))
        #    for ii, vv in enumerate(support_rw_2):
        #        reward_schedulers_2[i].append(np.mean(vv))

        p_pkt_loss_1.append(np.mean(pkt_l_drl_a_1))
        p_pkt_loss_2.append(np.mean(pkt_l_drl_a_2))

        actions_1.append(acts_1)
        actions_2.append(acts_2)

    rewards_drl_agent_1.append(np.mean(p_rewards_drl_agent_1))
    rewards_drl_agent_2.append(np.mean(p_rewards_drl_agent_2))

    pkt_loss_1.append(np.mean(p_pkt_loss_1))
    pkt_loss_2.append(np.mean(p_pkt_loss_2))

figure, axis = plt.subplots(2, 2)
axis[0,0].plot(tss, rewards_drl_agent_1, label=r'DRL-SRA$_{A2C}$')
axis[0,0].plot(tss, rewards_drl_agent_2, label=r'DRL-SRA$_{ACKTR}$')

for i,v in enumerate(env1.schedulers):
    axis[0,0].plot(reward_schedulers_1[i], label=v.name+r'$_{A2C}$')

axis[0,0].set_xlabel('Episodes')
axis[0,0].set_ylabel('Mean Sum-rate')
axis[0,0].set_title('Sum-rate per episode')
axis[0,0].legend()

axis[1,0].plot(tss, pkt_loss_1, label=r'DRL-SRA$_{A2C}$')
axis[1,0].plot(tss, pkt_loss_2, label=r'DRL-SRA$_{ACKTR}$')
axis[1,0].set_xlabel('Episodes')
axis[1,0].set_ylabel('Mean pkt loss')
axis[1,0].set_title('Packet loss per episode')
axis[1,0].legend()

plt.show()