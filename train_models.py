import gym
from stable_baselines import DQN, ACKTR
from policies.dqn_policies import CustomLnMlpPolicy

env = gym.make('sra_env:sra-v0')

#model = DQN(CustomLnMlpPolicy, env, verbose=1, gamma=0.99, learning_rate=0.007)
model = ACKTR("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=80000)
folder = 'E:/Docs_Doutorado/models_new_dataset3/'
model.save(folder + "new_acktr_80k")