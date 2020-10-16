from Agent import Agent
import gym
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    n_games = 400
    agent = Agent(lr=0.001, n_actions=4, gamma=0.99, epsilon=1, epsilon_dec=2e-4,input_dims=[8], batch_size=32, lstm=True, replace=10, normalize=True)
    scores, eps_history = [], []
    
    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()   
        while not done:
            env.render()
            action = agent.choose_action(observation)
            obs_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, obs_, done)
            observation = obs_
            agent.learn()
        eps_history.append(agent.epsilon)
        scores.append(score)
        avg_score = np.mean(scores[-100:])
        
        print("Episode: ", i, " Score: %.1f " % score, 
              "Avg Score: %.1f" % avg_score,
              "Epislon: %.2f" % agent.epsilon)
    
         