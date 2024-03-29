import os
import sys

LR = 0.007
LR_D= '007'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines import ACKTR, A2C, TRPO, DQN, PPO1, PPO2, ACER
from tqdm import tqdm
import json
from json import JSONEncoder

import consts
import copy

## env dealling with combinatorial action space
## computing percentual of the packet loss per allocation
from sra_env.sra_env6 import SRAEnv


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

env1 = SRAEnv(type="Master", running_tp=1, desc="A2C")
#env1.running_tp = 1 # validation with different dataset
#env1.type = "Master"

#simulation_type = "stationary"
simulation_type = "n-stationary"
#F = "_F_3-3_NR" # NR - No repetition - combinatorial action space
#F = "_F_3-3_NR_ND" # ND - New Data
#F = "_F_3-3_LE" # LE = Less Training Episode data = 30 episodes
#F = "_F_3-3_ME" # LE = Less Training Episode data = 30 episodes - ME = More TE = 100
#F = "_F_3-3_ME_TI_mixed" # LE = Less Training Episode data = 30 episodes - ME = More TE = 100 - TI traffic int
#F = "_F_2-2_ME"
#F = "_F_1-1_ME_TI_low1"
F = "_F_" + consts.F_D + "_all_mixed_ti11"

# number of executions/episodes per trained models
t = 50

tqdm_ = "10-30000"
rr = list(range(1000,30010,1000))
#rr = list(range(10000,50001,10000))
rr.append(10)
#rr.append(30000)
#rr.append(40000)
#rr.append(50000)
#rr.append(60000)
rr.sort()
tqdm_e = tqdm(rr, desc='Time Steps', leave=True, unit=" time steps")

obs, obs2, obs3, obs4, obs5, obs6, obs7 = None, None, None, None, None, None, None

env1.reset()
obs, rw, endep, info = env1.step_(0)

if 'ACKTR' in sys.argv:
    obs2 = copy.deepcopy(obs)
    env2 = copy.deepcopy(env1)
    env2.type = "Slave"
    env2.par_envs['Master'].append(env1)

if 'TRPO' in sys.argv:
    obs3 = copy.deepcopy(obs)
    env3 = copy.deepcopy(env1)
    env3.type = "Slave"
    env3.par_envs['Master'].append(env1)

if 'ACER' in sys.argv:
    obs4 = copy.deepcopy(obs)
    env4 = copy.deepcopy(env1)
    env4.type = "Slave"
    env4.par_envs['Master'].append(env1)

if 'DQN' in sys.argv:
    obs5 = copy.deepcopy(obs)
    env5 = copy.deepcopy(env1)
    env5.type = "Slave"
    env5.par_envs['Master'].append(env1)

if 'PPO1' in sys.argv:
    obs6 = copy.deepcopy(obs)
    env6 = copy.deepcopy(env1)
    env6.type = "Slave"
    env6.par_envs['Master'].append(env1)

if 'PPO2' in sys.argv:
    obs7 = copy.deepcopy(obs)
    env7 = copy.deepcopy(env1)
    env7.type = "Slave"
    env7.par_envs['Master'].append(env1)

rewards_drl_agent_1, rewards_drl_agent_2, rewards_drl_agent_3, rewards_drl_agent_4, rewards_drl_agent_5, \
rewards_drl_agent_6, rewards_drl_agent_7 = [], [], [], [], [], [], []

reward_schedulers = []

pkt_loss_1, pkt_loss_2, pkt_loss_3, pkt_loss_4, pkt_loss_5, pkt_loss_6, pkt_loss_7 = [], [], [], [], [], [], []

pkt_d_1, pkt_d_2, pkt_d_3, pkt_d_4, pkt_d_5, pkt_d_6, pkt_d_7 = [], [], [], [], [], [], []

pkt_loss_sch = []
pkt_delay_sch = []
actions_1, actions_2, actions_3 = [], [], []
actions_4 = []
actions_5, actions_6, actions_7 = [], [], []
c_act_1, c_act_2, c_act_3 = [0] * len(env1.schedulers), [0] * len(env1.schedulers), [0] * len(env1.schedulers)
c_act_5, c_act_6, c_act_7 = [0] * len(env1.schedulers), [0] * len(env1.schedulers), [0] * len(env1.schedulers)
tss = []
#folder = consts.MODELS_FOLDER_STATIONARY
#folder = consts.MODELS_FOLDER
#folder = consts.MODELS_FINAL
folder = 'trained_models_4/v3/'



for i in tqdm_e:
    ts = i
    tss.append(ts)
    base_file = F + "_gamma_" + consts.GAMMA_D + "_lr_" + LR_D + '_epsilon_' + consts.EPSILON_D
    if obs is not None:
        ## Loading the trained model for A2C
        model1 = A2C.load(folder+"a2c_drl_"+str(ts)+base_file)
    if obs2 is not None:
        ## Loading the trained model for A2C
        model2 = ACKTR.load(folder+"acktr_"+str(ts)+base_file)
    if obs3 is not None:
        ## Loading the trained model for TRPO
        model3 = TRPO.load(folder+"trpo_" + str(ts)+base_file)
    if obs4 is not None:
        model4 = ACER.load(folder+"acer_"+str(ts)+base_file)
    if obs5 is not None:
        ## Loading the trained model for TRPO
        model5 = DQN.load(folder+"dqn_" + str(ts)+base_file)
    if obs6 is not None:
        ## Loading the trained model for TRPO
        model6 = PPO1.load(folder+"ppo1_" + str(ts)+base_file)
    if obs7 is not None:
        ## Loading the trained model for TRPO
        model7 = PPO2.load(folder+"ppo2_" + str(ts)+base_file)

    p_rewards_drl_agent_1, p_rewards_drl_agent_2, p_rewards_drl_agent_3, p_rewards_drl_agent_4, \
    p_rewards_drl_agent_5, p_rewards_drl_agent_6, p_rewards_drl_agent_7= [], [], [], [], [], [], []

    p_reward_schedulers = [[] for i in range(len(env1.schedulers))]

    p_pkt_loss_1, p_pkt_loss_2, p_pkt_loss_3, p_pkt_loss_4, p_pkt_loss_5, p_pkt_loss_6, p_pkt_loss_7 = [], [], [], [], [], [], []

    p_pkt_d_1, p_pkt_d_2, p_pkt_d_3, p_pkt_d_4, p_pkt_d_5, p_pkt_d_6, p_pkt_d_7 = [], [], [], [], [], [], []

    p_pkt_schedulers = [[] for i in range(len(env1.schedulers))]
    p_pkt_d_schedulers = [[] for i in range(len(env1.schedulers))]

    # loop for "Monte Carlo" validation
    for ti in range(t):
        rw_drl_a_1, rw_drl_a_2, rw_drl_a_3, rw_drl_a_4, rw_drl_a_5, rw_drl_a_6, rw_drl_a_7 = [], [], [], [], [], [], []
        # rewards for schedulers
        rw_sh = [[] for i in range(len(env1.schedulers))]
        pkt_l_sh = [[] for i in range(len(env1.schedulers))]
        pkt_d_sh = [[] for i in range(len(env1.schedulers))]
        pkt_l_drl_a_1, pkt_l_drl_a_2, pkt_l_drl_a_3, pkt_l_drl_a_4, pkt_l_drl_a_5, pkt_l_drl_a_6, pkt_l_drl_a_7 = [], [], [], [], [], [], []
        pkt_d_drl_a_1, pkt_d_drl_a_2, pkt_d_drl_a_3, pkt_d_drl_a_4, pkt_d_drl_a_5, pkt_d_drl_a_6, pkt_d_drl_a_7 = [], [], [], [], [], [], []

        acts_1, acts_2, acts_3, acts_4, acts_5, acts_6, acts_7 = [], [], [], [], [], [], []

        # running a entire episode
        for ii in range(env1.blocks_ep):
            if obs is not None:
                while True:
                    action1, _ = model1.predict(obs,deterministic=True)
                    acts_1.append(action1)
                    obs, rewards_1, _, _ = env1.step_(action1)
                    if rewards_1[0] != 0:
                        break
                # for model1
                rw_drl_a_1.append(rewards_1[0])
                # index 2 is the drl agent pkt loss
                # -10 is a fake packet loss. The real value only comes at the episode end.
                if rewards_1[2][0] > -10.:
                    pkt_l_drl_a_1.append(rewards_1[2][0])
                pkt_d_drl_a_1.append(np.mean(rewards_1[3][0]))
            if obs2 is not None:
                while True:
                    action2, _ = model2.predict(obs2,deterministic=True)
                    acts_2.append(action2)
                    obs2, rewards_2, _, _ = env2.step_(action2)
                    if rewards_2[0] != 0:
                        break
                        # for model2
                rw_drl_a_2.append(rewards_2[0])
                # the last one is the drl agent pkt loss
                # -10 is a fake packet loss. The real value only comes at the episode end.
                if rewards_2[2][0] > -10.:
                    pkt_l_drl_a_2.append(rewards_2[2][0])
                pkt_d_drl_a_2.append(np.mean(rewards_2[3][0]))
            if obs3 is not None:
                while True:
                    action3, _ = model3.predict(obs3,deterministic=True)
                    acts_3.append(action3)
                    obs3, rewards_3, _, _ = env3.step_(action3)
                    if rewards_3[0] != 0:
                        break
                # for model3
                rw_drl_a_3.append(rewards_3[0])
                # the last one is the drl agent pkt loss
                # -10 is a fake packet loss. The real value only comes at the episode end.
                if rewards_3[2][0] > -10.:
                    pkt_l_drl_a_3.append(rewards_3[2][0])
                pkt_d_drl_a_3.append(np.mean(rewards_3[3][0]))
            if obs4 is not None:
                while True:
                    action4, _ = model4.predict(obs4,deterministic=True)
                    acts_4.append(action4)
                    obs4, rewards_4, _, _ = env4.step_(action4)
                    if rewards_4[0] != 0:
                        break
                # for model4
                rw_drl_a_4.append(rewards_4[0])
                # the last one is the drl agent pkt loss
                # -10 is a fake packet loss. The real value only comes at the episode end.
                if rewards_4[2][0] > -10.:
                    pkt_l_drl_a_4.append(rewards_4[2][0])
                pkt_d_drl_a_4.append(np.mean(rewards_4[3][0]))
            if obs5 is not None:
                while True:
                    action5, _ = model5.predict(obs5,deterministic=True)
                    acts_5.append(action5)
                    obs5, rewards_5, _, _ = env5.step_(action5)
                    if rewards_5[0] != 0:
                        break
                # for model5
                rw_drl_a_5.append(rewards_5[0])
                # the last one is the drl agent pkt loss
                # -10 is a fake packet loss. The real value only comes at the episode end.
                if rewards_5[2][0] > -10.:
                    pkt_l_drl_a_5.append(rewards_5[2][0])
                pkt_d_drl_a_5.append(np.mean(rewards_5[3][0]))
            if obs6 is not None:
                while True:
                    action6, _ = model6.predict(obs6,deterministic=True)
                    acts_6.append(action6)
                    obs6, rewards_6, _, _ = env6.step_(action6)
                    if rewards_6[0] != 0:
                        break
                # for model6
                rw_drl_a_6.append(rewards_6[0])
                # the last one is the drl agent pkt loss
                # -10 is a fake packet loss. The real value only comes at the episode end.
                if rewards_6[2][0] > -10.:
                    pkt_l_drl_a_6.append(rewards_6[2][0])
                pkt_d_drl_a_6.append(np.mean(rewards_6[3][0]))
            if obs7 is not None:
                while True:
                    action7, _ = model7.predict(obs7,deterministic=True)
                    acts_7.append(action7)
                    obs7, rewards_7, _, _ = env7.step_(action7)
                    if rewards_7[0] != 0:
                        break
                # for model7
                rw_drl_a_7.append(rewards_7[0])
                # the last one is the drl agent pkt loss
                # -10 is a fake packet loss. The real value only comes at the episode end.
                if rewards_7[2][0] > -10.:
                    pkt_l_drl_a_7.append(rewards_7[2][0])
                pkt_d_drl_a_7.append(np.mean(rewards_7[3][0]))

            ## running for each scheduler
            for u,v in enumerate(env1.schedulers):
                rw_sh[u].append(rewards_1[1][u])
                # -10 is a fake packet loss. The real value only comes at the episode end.
                if rewards_1[2][u + 1] > -10.:
                    pkt_l_sh[u].append(rewards_1[2][u+1])
                pkt_d_sh[u].append(np.mean(rewards_1[3][u + 1]))

        # appending the episode allocations results #################

        p_rewards_drl_agent_1.append(rw_drl_a_1)
        p_rewards_drl_agent_2.append(rw_drl_a_2)
        p_rewards_drl_agent_3.append(rw_drl_a_3)
        p_rewards_drl_agent_4.append(rw_drl_a_4)
        p_rewards_drl_agent_5.append(rw_drl_a_5)
        p_rewards_drl_agent_6.append(rw_drl_a_6)
        p_rewards_drl_agent_7.append(rw_drl_a_7)

        # schedulers
        p_sc = []
        for i,v in enumerate(env1.schedulers):
            #p_reward_schedulers[i] += rw_sh[i] / len(rw_drl_a_1)
            p_reward_schedulers[i].append(rw_sh[i])
            p_pkt_schedulers[i].append(pkt_l_sh[i])
            p_pkt_d_schedulers[i].append(pkt_d_sh[i])
        #p_reward_schedulers.append(p_sc)

        p_pkt_loss_1.append(pkt_l_drl_a_1)
        p_pkt_loss_2.append(pkt_l_drl_a_2)
        p_pkt_loss_3.append(pkt_l_drl_a_3)
        p_pkt_loss_4.append(pkt_l_drl_a_4)
        p_pkt_loss_5.append(pkt_l_drl_a_5)
        p_pkt_loss_6.append(pkt_l_drl_a_6)
        p_pkt_loss_7.append(pkt_l_drl_a_7)

        p_pkt_d_1.append(pkt_d_drl_a_1)
        p_pkt_d_2.append(pkt_d_drl_a_2)
        p_pkt_d_3.append(pkt_d_drl_a_3)
        p_pkt_d_4.append(pkt_d_drl_a_4)
        p_pkt_d_5.append(pkt_d_drl_a_5)
        p_pkt_d_6.append(pkt_d_drl_a_6)
        p_pkt_d_7.append(pkt_d_drl_a_7)

        actions_1.append(acts_1)
        actions_2.append(acts_2)
        actions_3.append(acts_3)
        actions_4.append(acts_4)
        actions_5.append(acts_5)
        actions_6.append(acts_6)
        actions_7.append(acts_7)

    rewards_drl_agent_1.append(p_rewards_drl_agent_1)
    rewards_drl_agent_2.append(p_rewards_drl_agent_2)
    rewards_drl_agent_3.append(p_rewards_drl_agent_3)
    rewards_drl_agent_4.append(p_rewards_drl_agent_4)
    rewards_drl_agent_5.append(p_rewards_drl_agent_5)
    rewards_drl_agent_6.append(p_rewards_drl_agent_6)
    rewards_drl_agent_7.append(p_rewards_drl_agent_7)

    # p_rs = []
    # for u,v in enumerate(env1.schedulers):
    #     p_rs.append(p_reward_schedulers[u])
    #     pkt_loss_sch[u].append(p_pkt_schedulers[u])
    #     pkt_delay_sch[u].append(p_pkt_d_schedulers[u])

    reward_schedulers.append(p_reward_schedulers)
    pkt_loss_sch.append(p_pkt_schedulers)
    pkt_delay_sch.append(p_pkt_d_schedulers)

    pkt_loss_1.append(p_pkt_loss_1)
    pkt_loss_2.append(p_pkt_loss_2)
    pkt_loss_3.append(p_pkt_loss_3)
    pkt_loss_4.append(p_pkt_loss_4)
    pkt_loss_5.append(p_pkt_loss_5)
    pkt_loss_6.append(p_pkt_loss_6)
    pkt_loss_7.append(p_pkt_loss_7)

    pkt_d_1.append(p_pkt_d_1)
    pkt_d_2.append(p_pkt_d_2)
    pkt_d_3.append(p_pkt_d_3)
    pkt_d_4.append(p_pkt_d_4)
    pkt_d_5.append(p_pkt_d_5)
    pkt_d_6.append(p_pkt_d_6)
    pkt_d_7.append(p_pkt_d_7)

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


# saving the simulation history

with open('history_final/'+simulation_type+ F +'_history_full_'+str(tqdm_)+'_'+str(t)+'_rounds_'
          +str(env1.blocks_ep)+'_bloks_eps_lr_'+ LR_D +'.json', 'w') as outfile:
    json.dump(history, outfile, cls=NumpyArrayEncoder)
#############################




