import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines import ACKTR, A2C, TRPO, DQN, PPO1, PPO2
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



tqdm_e = tqdm(range(0,140,5), desc='Time Steps', leave=True, unit=" time steps")
obs, rw, endep, info = env1.step_(0)
obs2 = copy.deepcopy(obs)
env2 = copy.deepcopy(env1)
env2.type = "Slave"
env2.par_envs['Master'].append(env1)

obs3 = copy.deepcopy(obs)
env3 = copy.deepcopy(env1)
env3.type = "Slave"
env3.par_envs['Master'].append(env1)

obs5 = copy.deepcopy(obs)
env5 = copy.deepcopy(env1)
env5.type = "Slave"
env5.par_envs['Master'].append(env1)

obs6 = copy.deepcopy(obs)
env6 = copy.deepcopy(env1)
env6.type = "Slave"
env6.par_envs['Master'].append(env1)

obs7 = copy.deepcopy(obs)
env7 = copy.deepcopy(env1)
env7.type = "Slave"
env7.par_envs['Master'].append(env1)

rewards_drl_agent_1, rewards_drl_agent_2, rewards_drl_agent_3 = [], [], []
rewards_drl_agent_5, rewards_drl_agent_6, rewards_drl_agent_7 = [], [], []
reward_schedulers_1 = [[] for i in range(len(env1.schedulers))]
reward_schedulers_2 = copy.deepcopy(reward_schedulers_1)
reward_schedulers_3 = copy.deepcopy(reward_schedulers_1)
reward_schedulers_5 = copy.deepcopy(reward_schedulers_1)
reward_schedulers_6 = copy.deepcopy(reward_schedulers_1)
reward_schedulers_7 = copy.deepcopy(reward_schedulers_1)
pkt_loss_1, pkt_loss_2, pkt_loss_3 = [], [], []
pkt_loss_5, pkt_loss_6, pkt_loss_7 = [], [], []
actions_1, actions_2, actions_3 = [], [], []
actions_5, actions_6, actions_7 = [], [], []
c_act_1, c_act_2, c_act_3 = [0] * len(env1.schedulers), [0] * len(env1.schedulers), [0] * len(env1.schedulers)
c_act_5, c_act_6, c_act_7 = [0] * len(env1.schedulers), [0] * len(env1.schedulers), [0] * len(env1.schedulers)
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
    ## Loading the trained model for TRPO
    model3 = TRPO.load("models/trpo_" + str(ts))
    ## Loading the trained model for TRPO
    model5 = DQN.load("models/dqn_" + str(ts))
    ## Loading the trained model for TRPO
    model6 = PPO1.load("models/ppo1_" + str(ts))
    ## Loading the trained model for TRPO
    model7 = PPO2.load("models/ppo2_" + str(ts))

    p_rewards_drl_agent_1, p_rewards_drl_agent_2, p_rewards_drl_agent_3= [], [], []
    p_rewards_drl_agent_5, p_rewards_drl_agent_6, p_rewards_drl_agent_7 = [], [], []
    p_reward_schedulers_1 = [[] for i in range(len(env1.schedulers))]
    p_reward_schedulers_2 = copy.deepcopy(p_reward_schedulers_1)
    p_reward_schedulers_3 = copy.deepcopy(p_reward_schedulers_1)
    p_reward_schedulers_5 = copy.deepcopy(p_reward_schedulers_1)
    p_reward_schedulers_6 = copy.deepcopy(p_reward_schedulers_1)
    p_reward_schedulers_7 = copy.deepcopy(p_reward_schedulers_1)
    p_pkt_loss_1, p_pkt_loss_2, p_pkt_loss_3 = [], [], []
    p_pkt_loss_5, p_pkt_loss_6, p_pkt_loss_7 = [], [], []

    for i in range(10):
        rw_drl_a_1 = []
        rw_drl_a_2 = []
        rw_drl_a_3 = []
        rw_drl_a_5 = []
        rw_drl_a_6 = []
        rw_drl_a_7 = []
        rw_sh_1 = []
        rw_sh_2 = []
        rw_sh_3 = []
        rw_sh_5 = []
        rw_sh_6 = []
        rw_sh_7 = []
        pkt_l_drl_a_1 = []
        pkt_l_drl_a_2 = []
        pkt_l_drl_a_3 = []
        pkt_l_drl_a_5 = []
        pkt_l_drl_a_6 = []
        pkt_l_drl_a_7 = []
        pkt_l_sh_1 = []
        pkt_l_sh_2 = []
        pkt_l_sh_3 = []
        pkt_l_sh_5 = []
        pkt_l_sh_6 = []
        pkt_l_sh_7 = []
        acts_1 = []
        acts_2 = []
        acts_3 = []
        acts_5 = []
        acts_6 = []
        acts_7 = []
        for ii in range(env1.blocks_ep):
            action1, _ = model1.predict(obs)
            acts_1.append(action1)
            obs, rewards_1, _, _ = env1.step_(action1)

            action2, _ = model2.predict(obs2)
            acts_2.append(action2)
            obs2, rewards_2, _, _ = env2.step_(action2)

            action3, _ = model3.predict(obs3)
            acts_3.append(action3)
            obs3, rewards_3, _, _ = env3.step_(action3)

            action5, _ = model5.predict(obs5)
            acts_5.append(action5)
            obs5, rewards_5, _, _ = env5.step_(action5)

            action6, _ = model6.predict(obs6)
            acts_6.append(action6)
            obs6, rewards_6, _, _ = env6.step_(action6)

            action7, _ = model7.predict(obs7)
            acts_7.append(action7)
            obs7, rewards_7, _, _ = env7.step_(action7)

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
            # for model3
            rw_drl_a_3.append(rewards_3[0])
            rw_sh_3.append(rewards_3[1])
            # the last one is the drl agent pkt loss
            pkt_l_drl_a_3.append(rewards_3[2][len(rewards_3[2]) - 1])
            # for model5
            rw_drl_a_5.append(rewards_5[0])
            rw_sh_5.append(rewards_5[1])
            # the last one is the drl agent pkt loss
            pkt_l_drl_a_5.append(rewards_5[2][len(rewards_5[2]) - 1])
            # for model6
            rw_drl_a_6.append(rewards_6[0])
            rw_sh_6.append(rewards_6[1])
            # the last one is the drl agent pkt loss
            pkt_l_drl_a_6.append(rewards_6[2][len(rewards_6[2]) - 1])
            # for model7
            rw_drl_a_7.append(rewards_7[0])
            rw_sh_7.append(rewards_7[1])
            # the last one is the drl agent pkt loss
            pkt_l_drl_a_7.append(rewards_7[2][len(rewards_7[2]) - 1])

        # env.render()
        p_rewards_drl_agent_1.append(np.mean(rw_drl_a_1))
        p_rewards_drl_agent_2.append(np.mean(rw_drl_a_2))
        p_rewards_drl_agent_3.append(np.mean(rw_drl_a_3))
        p_rewards_drl_agent_5.append(np.mean(rw_drl_a_5))
        p_rewards_drl_agent_6.append(np.mean(rw_drl_a_6))
        p_rewards_drl_agent_7.append(np.mean(rw_drl_a_7))
        p_support_rw_1 = [[] for i in range(len(env1.schedulers))]
        p_support_rw_2 = [[] for i in range(len(env1.schedulers))]
        p_support_rw_3 = [[] for i in range(len(env1.schedulers))]
        p_support_rw_5 = [[] for i in range(len(env1.schedulers))]
        p_support_rw_6 = [[] for i in range(len(env1.schedulers))]
        p_support_rw_7 = [[] for i in range(len(env1.schedulers))]
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
        p_pkt_loss_3.append(np.mean(pkt_l_drl_a_3))
        p_pkt_loss_5.append(np.mean(pkt_l_drl_a_5))
        p_pkt_loss_6.append(np.mean(pkt_l_drl_a_6))
        p_pkt_loss_7.append(np.mean(pkt_l_drl_a_7))

        actions_1.append(acts_1)
        actions_2.append(acts_2)
        actions_3.append(acts_3)
        actions_5.append(acts_5)
        actions_6.append(acts_6)
        actions_7.append(acts_7)

    rewards_drl_agent_1.append(np.mean(p_rewards_drl_agent_1))
    rewards_drl_agent_2.append(np.mean(p_rewards_drl_agent_2))
    rewards_drl_agent_3.append(np.mean(p_rewards_drl_agent_3))
    rewards_drl_agent_5.append(np.mean(p_rewards_drl_agent_5))
    rewards_drl_agent_6.append(np.mean(p_rewards_drl_agent_6))
    rewards_drl_agent_7.append(np.mean(p_rewards_drl_agent_7))

    pkt_loss_1.append(np.mean(p_pkt_loss_1))
    pkt_loss_2.append(np.mean(p_pkt_loss_2))
    pkt_loss_3.append(np.mean(p_pkt_loss_3))
    pkt_loss_5.append(np.mean(p_pkt_loss_5))
    pkt_loss_6.append(np.mean(p_pkt_loss_6))
    pkt_loss_7.append(np.mean(p_pkt_loss_7))

figure, axis = plt.subplots(2, 2)
axis[0,0].plot(tss, rewards_drl_agent_1, label=r'DRL-SRA$_{A2C}$')
axis[0,0].plot(tss, rewards_drl_agent_2, label=r'DRL-SRA$_{ACKTR}$')
axis[0,0].plot(tss, rewards_drl_agent_3, label=r'DRL-SRA$_{TRPO}$')
axis[0,0].plot(tss, rewards_drl_agent_5, label=r'DRL-SRA$_{DQN}$')
axis[0,0].plot(tss, rewards_drl_agent_6, label=r'DRL-SRA$_{PPO1}$')
axis[0,0].plot(tss, rewards_drl_agent_7, label=r'DRL-SRA$_{PPO2}$')


for i,v in enumerate(env1.schedulers):
    axis[0,0].plot(reward_schedulers_1[i], label=v.name+r'$_{A2C}$')

axis[0,0].set_xlabel('Episodes')
axis[0,0].set_ylabel('Mean Sum-rate')
axis[0,0].set_title('Sum-rate per episode')
axis[0,0].legend()

axis[1,0].plot(tss, pkt_loss_1, label=r'DRL-SRA$_{A2C}$')
axis[1,0].plot(tss, pkt_loss_2, label=r'DRL-SRA$_{ACKTR}$')
axis[1,0].plot(tss, pkt_loss_3, label=r'DRL-SRA$_{TRPO}$')
axis[1,0].plot(tss, pkt_loss_5, label=r'DRL-SRA$_{DQN}$')
axis[1,0].plot(tss, pkt_loss_6, label=r'DRL-SRA$_{PPO1}$')
axis[1,0].plot(tss, pkt_loss_7, label=r'DRL-SRA$_{PPO2}$')
axis[1,0].set_xlabel('Episodes')
axis[1,0].set_ylabel('Mean pkt loss')
axis[1,0].set_title('Packet loss per episode')
axis[1,0].legend()

plt.show()