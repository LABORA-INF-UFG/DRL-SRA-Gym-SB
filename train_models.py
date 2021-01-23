import sys
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.deepq.policies import MlpPolicy as mlpp
from stable_baselines.common.callbacks import EvalCallback
from training_callback import TrainingCallback
from model_eval import ModelEval
from stable_baselines import A2C, ACKTR, TRPO, ACER, DQN, PPO1, PPO2
from sra_env.sra_env3 import SRAEnv
from tqdm import tqdm
import consts
import sys
import os

## training models using combinatorial action space
env = SRAEnv()
env.running_tp = 0 # 0 for training; 1 for validation

model1_eval = ModelEval()

training_cbk_md1 = TrainingCallback(eval_interval=10,
                                    episode_length=consts.BLOCKS_EP,
                                    model_eval=model1_eval, name='a2c_drl')

model1, model2, model3, model4, model5, model6, model7 = None, None, None, None, None, None, None
#model1 = A2C(MlpPolicy, env, verbose=0, gamma=consts.GAMMA, learning_rate=LR, epsilon=consts.EPSILON)
# model5 = DQN(mlpp, env, verbose=0, gamma=consts.GAMMA, learning_rate=LR)
# model6 = PPO1(MlpPolicy, env, verbose=0, gamma=consts.GAMMA, adam_epsilon=consts.EPSILON)

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

if model1:
    model1.learn(total_timesteps=2000, callback=training_cbk_md1)  # time steps
    # model1.save(folder+"a2c_drl_300"+base_file)

