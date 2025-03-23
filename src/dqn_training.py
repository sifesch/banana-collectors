import gym
import random
import torch
import numpy as np
from collections import deque
from unityagents import UnityEnvironment
from dqn_agent import Agent

def dqn(brain_name, env: UnityEnvironment, agent: Agent, n_episodes: int = 2000, max_t: int = 1000, eps_start: float = 1.0, eps_end:float = 0.01, eps_decay: float = 0.995, training_trial_number: str = '03') -> list:
    """Deep Q-Learning.
    
    Params
    ======
        brain_name (str): name of the brain in the unity env.
        env (UnityEnvironment): The Unity env instance in which the agent interacts. Provides states, rewards, etc.
        agent (Agent): rl agent that interacts with env
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        training_trial_number: (string): Indicates the name of the training trial for the sake of saving models with the respective name
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        # Reset env, extract the data  and extract the state.
        env_info = env.reset(train_mode=True)[brain_name] 
        state = env_info.vector_observations[0]  
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            # Retrive step result, extract next state and the reward
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0] 
            done = env_info.local_done[0] 
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), f'results/model_weights/checkpoint_{training_trial_number}.pth')
            break
    return scores