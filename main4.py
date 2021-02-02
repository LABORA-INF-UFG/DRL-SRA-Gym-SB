import gym
from stable_baselines.common import make_vec_env
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.deepq.policies import MlpPolicy as mlpp
from stable_baselines import A2C, ACKTR, TRPO, ACER, DQN, PPO1, PPO2
from sra_env.sra_env2 import SRAEnv
from tqdm import tqdm
import consts
import sys
import os

'''
run "traditional" training. Main4 > run_simulation_final > make_plot_final
'''

## training models using combinatorial action space

env = SRAEnv()
env.running_tp = 0
model1, model2, model3, model4, model5, model6, model7 = None, None, None, None, None, None, None
if 'A2C' in sys.argv:
    model1 = A2C(MlpPolicy, env, verbose=0, gamma=consts.GAMMA, learning_rate=consts.LR, epsilon=consts.EPSILON)
if 'ACKTR' in sys.argv:
    model2 = ACKTR(MlpPolicy, env, verbose=0, gamma=consts.GAMMA, learning_rate=consts.LR)
if 'TRPO' in sys.argv:
    model3 = TRPO(MlpPolicy, env, verbose=0, gamma=consts.GAMMA, vf_stepsize=consts.LR)
if 'ACER' in sys.argv:
    model4 = ACER(MlpPolicy, env, verbose=0, gamma=consts.GAMMA, learning_rate=consts.LR)
if 'DQN' in sys.argv:
    model5 = DQN(mlpp, env, verbose=0, gamma=consts.GAMMA, learning_rate=consts.LR)
if 'PPO1' in sys.argv:
    model6 = PPO1(MlpPolicy, env, verbose=0, gamma=consts.GAMMA, adam_epsilon=consts.EPSILON)
if 'PPO2' in sys.argv:
    model7 = PPO2(MlpPolicy, env, verbose=0, gamma=consts.GAMMA, learning_rate=consts.LR)

#tqdm_e = tqdm(range(10,100011,50000), desc='Time Steps', leave=True, unit=" time steps")
#folder = consts.MODELS_FOLDER_STATIONARY
#folder = consts.MODELS_FOLDER

rr = list(range(1000,100001,1000))
rr.append(10)
rr.sort()
tqdm_e = tqdm(rr, desc='Time Steps', leave=True, unit=" time steps")

#folder = consts.MODELS_FINAL
folder = 'trained_models/'
if not os.path.exists(folder):
    os.makedirs(folder)
#F = "_F_3-3_ME_TI_2" # LE = Less Training Episode data = 30 episodes - ME = More TE = 100 - TI traffic int
# low using load factor = 2
# low 2 using load factor = 8
#F = "_F_1-1_ME_TI_low1" # LE = Less Training Episode data = 30 episodes - ME = More TE = 100 - TI traffic interference
F = "_F_" + consts.F_D + "_high"

for i in tqdm_e:
    #i = 1 if i == 0 else i
    #ts = consts.BLOCKS_EP * i
    ts = i
    #ts = consts.BLOCKS_EP * (i + 1)
    base_file = F + "_gamma_"+consts.GAMMA_D+"_lr_"+consts.LR_D+'_epsilon_'+consts.EPSILON_D

    if model1:
        model1.learn(total_timesteps=ts) # time steps
        model1.save(folder+"a2c_drl_"+str(ts)+base_file)

    if model2:
        model2.learn(total_timesteps=ts)
        model2.save(folder+"acktr_"+str(ts)+base_file)

    if model3:
        model3.learn(total_timesteps=ts)
        model3.save(folder+"trpo_"+str(ts)+base_file)

    if model4:
        model4.learn(total_timesteps=ts)
        model4.save(folder+"acer_"+str(ts)+base_file)

    if model5:
        model5.learn(total_timesteps=ts)
        model5.save(folder+"dqn_" + str(ts)+base_file)

    if model6:
        model6.learn(total_timesteps=ts)
        model6.save(folder+"ppo1_" + str(ts)+base_file)

    if model7:
        model7.learn(total_timesteps=ts)
        model7.save(folder+"ppo2_" + str(ts)+base_file)
