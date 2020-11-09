import gym
from stable_baselines.common import make_vec_env
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import A2C, ACKTR
from sra_env.sra_env import SRAEnv

env = SRAEnv()

model = A2C(MlpPolicy, env, verbose=1, gamma=0.99, learning_rate=0.0001, epsilon=1e-05)
model.learn(total_timesteps=300000) # time steps
model.save("a2c_drl_300k")
# Train the agent
#model = ACKTR('MlpPolicy', env, verbose=1).learn(5000)

# training and saving the model
#model = ACKTR(MlpPolicy, env, verbose=1)
#model.learn(total_timesteps=1000)
#model.save('acktr_drl')


# testing
# obs = env.reset()
# for i in range(1000):
#     action, _states = model.predict(obs)
#     print(action)
#     obs, rewards, dones, info = env.step(action)
#     env.render()