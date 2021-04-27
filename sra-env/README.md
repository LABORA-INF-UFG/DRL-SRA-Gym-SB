## New environment

* This is a Gym environment for scheduling and resource allocation problems, applied to PHY and MAC allocation problems, where the agent must take action to accommodate UEs on frequencies, so that the system reaches its maximum sum rate while controlling packet loss and delay.

* This environment was built according to https://github.com/openai/gym/blob/master/docs/creating-environments.md

* With this environment you can use with an instance of the SraEnv class, or direct from Gym make method, allowing the use of multiprocessing.

``
env = gym.make('sra_env:sra-v0') # for 1 env
env = make_vec_env("sra_env:sra-v0", n_envs=4) # for 4 envs - multiprocessing
``

#### Installing

```sh
pip install -e sra-env
```