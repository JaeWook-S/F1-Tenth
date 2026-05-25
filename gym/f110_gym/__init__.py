from gym.envs.registration import register
register(
	id='f110-v0',
	entry_point='f110_gym.envs:F110Env',
	)

register(
    id='baseline-v0',
    entry_point='f110_gym.envs:F110BaselineEnv',
	)

register(
    id='f110-v1',
    entry_point='f110_gym.envs:DDQN_Best_Env',
	)