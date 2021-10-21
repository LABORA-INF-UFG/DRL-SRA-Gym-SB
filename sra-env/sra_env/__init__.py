from gym.envs.registration import register

register(
    id='sra-v0',
    entry_point='sra_env.envs:SraEnv',
)
register(
    id='sra-v1',
    entry_point='sra_env.envs:SraEnv1',
)
register(
    id='sra-v2',
    entry_point='sra_env.envs:SraEnv2',
)