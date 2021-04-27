from gym.envs.registration import register

register(
    id='sra-v0',
    entry_point='sra_env.envs:SraEnv',
)