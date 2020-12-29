import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines import ACKTR, A2C, TRPO, DQN, PPO1, PPO2, ACER
from stable_baselines.common.policies import MlpPolicy
from tqdm import tqdm
import csv
import json
from json import JSONEncoder

import consts
import copy

### Simulando com F[3,3]

## env dealling with combinatorial action space
from sra_env.sra_env2 import SRAEnv


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

env1 = SRAEnv()
env1.type = "Master"

#simulation_type = "stationary"
simulation_type = "n-stationary"
#F = "_F_3-3_NR" # NR - No repetition - combinatorial action space
F = "_F_3-3_NR_ND" # ND - New Data

# number of executions per trained models
t = 100

tqdm_ = "0-200"
rr = list(range(0,200,5))
rr.append(1810000)
tqdm_e = tqdm(rr, desc='Time Steps', leave=True, unit=" time steps")
env1.reset()
obs, rw, endep, info = env1.step_(0)
obs2 = copy.deepcopy(obs)
env2 = copy.deepcopy(env1)
env2.type = "Slave"
env2.par_envs['Master'].append(env1)

obs3 = copy.deepcopy(obs)
env3 = copy.deepcopy(env1)
env3.type = "Slave"
env3.par_envs['Master'].append(env1)

obs4 = copy.deepcopy(obs)
env4 = copy.deepcopy(env1)
env4.type = "Slave"
env4.par_envs['Master'].append(env1)

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
rewards_drl_agent_4 = []
rewards_drl_agent_5, rewards_drl_agent_6, rewards_drl_agent_7 = [], [], []
reward_schedulers = []

pkt_loss_1, pkt_loss_2, pkt_loss_3 = [], [], []
pkt_loss_4 = []
pkt_loss_5, pkt_loss_6, pkt_loss_7 = [], [], []

pkt_d_1, pkt_d_2, pkt_d_3 = [], [], []
pkt_d_4 = []
pkt_d_5, pkt_d_6, pkt_d_7 = [], [], []

pkt_loss_sch = [[] for i in range(len(env1.schedulers))]
pkt_delay_sch = [[] for i in range(len(env1.schedulers))]
actions_1, actions_2, actions_3 = [], [], []
actions_4 = []
actions_5, actions_6, actions_7 = [], [], []
c_act_1, c_act_2, c_act_3 = [0] * len(env1.schedulers), [0] * len(env1.schedulers), [0] * len(env1.schedulers)
c_act_5, c_act_6, c_act_7 = [0] * len(env1.schedulers), [0] * len(env1.schedulers), [0] * len(env1.schedulers)
tss = []
#folder = consts.MODELS_FOLDER_STATIONARY
folder = consts.MODELS_FOLDER

for i in tqdm_e:
    #i = 1 if i == 0 else i
    #ts = consts.BLOCKS_EP * i
    ts = consts.BLOCKS_EP * (i + 1)
    if i > 200:
        ts = i
    tss.append(ts)
    base_file = F + "_gamma_" + consts.GAMMA_D + "_lr_" + consts.LR_D + '_epsilon_' + consts.EPSILON_D
    ## Loading the trained model for A2C
    model1 = A2C.load(folder+"a2c_drl_"+str(ts)+base_file)
    ## Loading the trained model for A2C
    model2 = ACKTR.load(folder+"acktr_"+str(ts)+base_file)
    ## Loading the trained model for TRPO
    model3 = TRPO.load(folder+"trpo_" + str(ts)+base_file)
    model4 = ACER.load(folder+"acer_"+str(ts)+base_file)
    ## Loading the trained model for TRPO
    model5 = DQN.load(folder+"dqn_" + str(ts)+base_file)
    ## Loading the trained model for TRPO
    model6 = PPO1.load(folder+"ppo1_" + str(ts)+base_file)
    ## Loading the trained model for TRPO
    model7 = PPO2.load(folder+"ppo2_" + str(ts)+base_file)

    p_rewards_drl_agent_1, p_rewards_drl_agent_2, p_rewards_drl_agent_3= [], [], []
    p_rewards_drl_agent_4 = []
    p_rewards_drl_agent_5, p_rewards_drl_agent_6, p_rewards_drl_agent_7 = [], [], []
    p_reward_schedulers = [[0.] for i in range(len(env1.schedulers))]

    p_pkt_loss_1, p_pkt_loss_2, p_pkt_loss_3 = [], [], []
    p_pkt_loss_4 = []
    p_pkt_loss_5, p_pkt_loss_6, p_pkt_loss_7 = [], [], []

    p_pkt_d_1, p_pkt_d_2, p_pkt_d_3 = [], [], []
    p_pkt_d_4 = []
    p_pkt_d_5, p_pkt_d_6, p_pkt_d_7 = [], [], []

    #p_pkt_schedulers = [[0.] for i in range(len(env1.schedulers))]
    p_pkt_schedulers = [[] for i in range(len(env1.schedulers))]
    p_pkt_d_schedulers = [[] for i in range(len(env1.schedulers))]

    for ti in range(t):
        rw_drl_a_1, rw_drl_a_2, rw_drl_a_3, rw_drl_a_4, rw_drl_a_5, rw_drl_a_6, rw_drl_a_7 = [], [], [], [], [], [], []
        # rewards for schedulers
        rw_sh = [[0.] for i in range(len(env1.schedulers))]
        pkt_l_sh = [[] for i in range(len(env1.schedulers))]
        pkt_d_sh = [[] for i in range(len(env1.schedulers))]
        pkt_l_drl_a_1, pkt_l_drl_a_2, pkt_l_drl_a_3, pkt_l_drl_a_4, pkt_l_drl_a_5, pkt_l_drl_a_6, pkt_l_drl_a_7 = [], [], [], [], [], [], []
        pkt_d_drl_a_1, pkt_d_drl_a_2, pkt_d_drl_a_3, pkt_d_drl_a_4, pkt_d_drl_a_5, pkt_d_drl_a_6, pkt_d_drl_a_7 = [], [], [], [], [], [], []

        acts_1, acts_2, acts_3, acts_4, acts_5, acts_6, acts_7 = [], [], [], [], [], [], []

        for ii in range(env1.blocks_ep):

            while True:
                action1, _ = model1.predict(obs)
                acts_1.append(action1)
                obs, rewards_1, _, _ = env1.step_(action1)
                if rewards_1[0] != 0:
                    break
            while True:
                action2, _ = model2.predict(obs2)
                acts_2.append(action2)
                obs2, rewards_2, _, _ = env2.step_(action2)
                if rewards_2[0] != 0:
                    break

            while True:
                action3, _ = model3.predict(obs3)
                acts_3.append(action3)
                obs3, rewards_3, _, _ = env3.step_(action3)
                if rewards_3[0] != 0:
                    break

            while True:
                action4, _ = model4.predict(obs4)
                acts_4.append(action4)
                obs4, rewards_4, _, _ = env4.step_(action4)
                if rewards_4[0] != 0:
                    break

            while True:
                action5, _ = model5.predict(obs5)
                acts_5.append(action5)
                obs5, rewards_5, _, _ = env5.step_(action5)
                if rewards_5[0] != 0:
                    break

            while True:
                action6, _ = model6.predict(obs6)
                acts_6.append(action6)
                obs6, rewards_6, _, _ = env6.step_(action6)
                if rewards_6[0] != 0:
                    break

            while True:
                action7, _ = model7.predict(obs7)
                acts_7.append(action7)
                obs7, rewards_7, _, _ = env7.step_(action7)
                if rewards_7[0] != 0:
                    break

            # for model1
            rw_drl_a_1.append(rewards_1[0])
            # index 2 is the drl agent pkt loss
            pkt_l_drl_a_1.append(rewards_1[2][0])
            pkt_d_drl_a_1.append(np.mean(rewards_1[3][0]))
            # for model2
            rw_drl_a_2.append(rewards_2[0])
            # the last one is the drl agent pkt loss
            pkt_l_drl_a_2.append(rewards_2[2][0])
            pkt_d_drl_a_2.append(np.mean(rewards_2[3][0]))
            # for model3
            rw_drl_a_3.append(rewards_3[0])
            # the last one is the drl agent pkt loss
            pkt_l_drl_a_3.append(rewards_3[2][0])
            pkt_d_drl_a_3.append(np.mean(rewards_3[3][0]))
            # for model4
            rw_drl_a_4.append(rewards_4[0])
            # the last one is the drl agent pkt loss
            pkt_l_drl_a_4.append(rewards_4[2][0])
            pkt_d_drl_a_4.append(np.mean(rewards_4[3][0]))
            # for model5
            rw_drl_a_5.append(rewards_5[0])
            # the last one is the drl agent pkt loss
            pkt_l_drl_a_5.append(rewards_5[2][0])
            pkt_d_drl_a_5.append(np.mean(rewards_5[3][0]))
            # for model6
            rw_drl_a_6.append(rewards_6[0])
            # the last one is the drl agent pkt loss
            pkt_l_drl_a_6.append(rewards_6[2][0])
            pkt_d_drl_a_6.append(np.mean(rewards_6[3][0]))
            # for model7
            rw_drl_a_7.append(rewards_7[0])
            # the last one is the drl agent pkt loss
            pkt_l_drl_a_7.append(rewards_7[2][0])
            pkt_d_drl_a_7.append(np.mean(rewards_7[3][0]))

            ## schedulers
            for u,v in enumerate(env1.schedulers):
                rw_sh[u] += rewards_1[1][u]
                pkt_l_sh[u].append(rewards_1[2][u+1])
                pkt_d_sh[u].append(np.mean(rewards_1[3][u + 1]))

        # env.render()
        p_rewards_drl_agent_1.append(np.mean(rw_drl_a_1))
        p_rewards_drl_agent_2.append(np.mean(rw_drl_a_2))
        p_rewards_drl_agent_3.append(np.mean(rw_drl_a_3))
        p_rewards_drl_agent_4.append(np.mean(rw_drl_a_4))
        p_rewards_drl_agent_5.append(np.mean(rw_drl_a_5))
        p_rewards_drl_agent_6.append(np.mean(rw_drl_a_6))
        p_rewards_drl_agent_7.append(np.mean(rw_drl_a_7))

        # schedulers
        p_sc = []
        for i,v in enumerate(env1.schedulers):
            #p_sc.append(rw_sh[i] / len(rw_drl_a_1))
            p_reward_schedulers[i] += rw_sh[i] / len(rw_drl_a_1)
            p_pkt_schedulers[i].append(np.mean(pkt_l_sh[i]))
            p_pkt_d_schedulers[i].append(np.mean(pkt_d_sh[i]))
        #p_reward_schedulers.append(p_sc)

        p_pkt_loss_1.append(np.mean(pkt_l_drl_a_1))
        p_pkt_loss_2.append(np.mean(pkt_l_drl_a_2))
        p_pkt_loss_3.append(np.mean(pkt_l_drl_a_3))
        p_pkt_loss_4.append(np.mean(pkt_l_drl_a_4))
        p_pkt_loss_5.append(np.mean(pkt_l_drl_a_5))
        p_pkt_loss_6.append(np.mean(pkt_l_drl_a_6))
        p_pkt_loss_7.append(np.mean(pkt_l_drl_a_7))

        p_pkt_d_1.append(np.mean(np.mean(pkt_d_drl_a_1)))
        p_pkt_d_2.append(np.mean(np.mean(pkt_d_drl_a_2)))
        p_pkt_d_3.append(np.mean(np.mean(pkt_d_drl_a_3)))
        p_pkt_d_4.append(np.mean(np.mean(pkt_d_drl_a_4)))
        p_pkt_d_5.append(np.mean(np.mean(pkt_d_drl_a_5)))
        p_pkt_d_6.append(np.mean(np.mean(pkt_d_drl_a_6)))
        p_pkt_d_7.append(np.mean(np.mean(pkt_d_drl_a_7)))

        actions_1.append(acts_1)
        actions_2.append(acts_2)
        actions_3.append(acts_3)
        actions_4.append(acts_4)
        actions_5.append(acts_5)
        actions_6.append(acts_6)
        actions_7.append(acts_7)

    rewards_drl_agent_1.append(np.mean(p_rewards_drl_agent_1))
    rewards_drl_agent_2.append(np.mean(p_rewards_drl_agent_2))
    rewards_drl_agent_3.append(np.mean(p_rewards_drl_agent_3))
    rewards_drl_agent_4.append(np.mean(p_rewards_drl_agent_4))
    rewards_drl_agent_5.append(np.mean(p_rewards_drl_agent_5))
    rewards_drl_agent_6.append(np.mean(p_rewards_drl_agent_6))
    rewards_drl_agent_7.append(np.mean(p_rewards_drl_agent_7))

    p_rs = []
    for u,v in enumerate(env1.schedulers):
        p_rs.append(p_reward_schedulers[u] / len(p_rewards_drl_agent_1))
        pkt_loss_sch[u].append(np.mean(p_pkt_schedulers[u]))
        pkt_delay_sch[u].append(np.mean(p_pkt_d_schedulers[u]))

    reward_schedulers.append(p_rs)

    pkt_loss_1.append(np.mean(p_pkt_loss_1))
    pkt_loss_2.append(np.mean(p_pkt_loss_2))
    pkt_loss_3.append(np.mean(p_pkt_loss_3))
    pkt_loss_4.append(np.mean(p_pkt_loss_4))
    pkt_loss_5.append(np.mean(p_pkt_loss_5))
    pkt_loss_6.append(np.mean(p_pkt_loss_6))
    pkt_loss_7.append(np.mean(p_pkt_loss_7))

    pkt_d_1.append(np.mean(p_pkt_d_1))
    pkt_d_2.append(np.mean(p_pkt_d_2))
    pkt_d_3.append(np.mean(p_pkt_d_3))
    pkt_d_4.append(np.mean(p_pkt_d_4))
    pkt_d_5.append(np.mean(p_pkt_d_5))
    pkt_d_6.append(np.mean(p_pkt_d_6))
    pkt_d_7.append(np.mean(p_pkt_d_7))

## savind results
history={
    "rewards_drl_agents":{
        "A2C": rewards_drl_agent_1,
        "ACKTR": rewards_drl_agent_2,
        "TRPO": rewards_drl_agent_3,
        "ACER": rewards_drl_agent_4,
        "DQN": rewards_drl_agent_5,
        "PPO1": rewards_drl_agent_6,
        "PPO2": rewards_drl_agent_7,
    },
    "rewards_schedulers": reward_schedulers,
    "pkt_loss_agents":{
        "A2C": pkt_loss_1,
        "ACKTR": pkt_loss_2,
        "TRPO": pkt_loss_3,
        "ACER": pkt_loss_4,
        "DQN": pkt_loss_5,
        "PPO1": pkt_loss_6,
        "PPO2": pkt_loss_7,
    },
    "pkt_loss_schedulers": pkt_loss_sch,
    "pkt_delay_schedulers": pkt_delay_sch,
    "pkt_delay_agents":{
        "A2C": pkt_d_1,
        "ACKTR": pkt_d_2,
        "TRPO": pkt_d_3,
        "ACER": pkt_d_4,
        "DQN": pkt_d_5,
        "PPO1": pkt_d_6,
        "PPO2": pkt_d_7,
    },
    "tss": tss
}

with open('history/'+simulation_type+ F +'b_history_'+str(tqdm_)+'_'+str(t)+'_rounds_'+str(env1.blocks_ep)+'_bloks_eps.json', 'w') as outfile:
    json.dump(history, outfile, cls=NumpyArrayEncoder)
#############################

figure, axis = plt.subplots(2)
axis[0].plot(tss, rewards_drl_agent_1, label=r'DRL-SRA$_{A2C}$')
axis[0].plot(tss, rewards_drl_agent_2, label=r'DRL-SRA$_{ACKTR}$')
axis[0].plot(tss, rewards_drl_agent_3, label=r'DRL-SRA$_{TRPO}$')
axis[0].plot(tss, rewards_drl_agent_4, label=r'DRL-SRA$_{ACER}$')
axis[0].plot(tss, rewards_drl_agent_5, label=r'DRL-SRA$_{DQN}$')
axis[0].plot(tss, rewards_drl_agent_6, label=r'DRL-SRA$_{PPO1}$')
axis[0].plot(tss, rewards_drl_agent_7, label=r'DRL-SRA$_{PPO2}$')

for i,v in enumerate(env1.schedulers):
    vrs = []
    for rs in reward_schedulers:
        vrs.append(rs[i])
    axis[0].plot(tss, vrs, linestyle='dashdot', label=v.name)

axis[0].set_xlabel('Episodes')
axis[0].set_ylabel('Mean Sum-rate')
axis[0].set_title('Sum-rate per episode')
axis[0].legend()

axis[1].plot(tss, pkt_loss_1, label=r'DRL-SRA$_{A2C}$')
axis[1].plot(tss, pkt_loss_2, label=r'DRL-SRA$_{ACKTR}$')
axis[1].plot(tss, pkt_loss_3, label=r'DRL-SRA$_{TRPO}$')
axis[1].plot(tss, pkt_loss_4, label=r'DRL-SRA$_{ACER}$')
axis[1].plot(tss, pkt_loss_5, label=r'DRL-SRA$_{DQN}$')
axis[1].plot(tss, pkt_loss_6, label=r'DRL-SRA$_{PPO1}$')
axis[1].plot(tss, pkt_loss_7, label=r'DRL-SRA$_{PPO2}$')

for i,v in enumerate(env1.schedulers):
    axis[1].plot(tss, pkt_loss_sch[i], linestyle='dashdot', label=v.name)

axis[1].set_xlabel('Episodes')
axis[1].set_ylabel('Mean pkt loss')
axis[1].set_title('Packet loss per episode')
axis[1].legend()

plt.show()


