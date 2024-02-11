from enviroments import PortfolioEnv
import gym

env = gym.envs.spec('TradingEnv-v1').make()
observation = env.reset()
print("shape =", observation["history"].shape)