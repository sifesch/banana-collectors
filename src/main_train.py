from unityagents import UnityEnvironment
import numpy as np
from utils import create_training_plot
import numpy as np
from dqn_agent import Agent
from dqn_training import dqn

if __name__ == '__main__':
    ####### Define the Trial Number and the Parameters for Training the DQN Agent #######
    trial_number = '05'
    n_episodes = 1800 # At least 101 Episodes are necessary for the plot function to work
    max_t = 1200
    eps_start = 1.0
    eps_end = 0.02
    eps_decay = 0.8
    
    # Adjust to your Banana Collector
    file_name_banana = "../Banana_Linux/Banana.x86_64" 

    env = UnityEnvironment(file_name=file_name_banana)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    agent = Agent(state_size=37, action_size=4, seed=0)

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    # number of agents in the environment
    print('Number of agents:', len(env_info.agents))
    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)
    # examine the state space 
    state = env_info.vector_observations[0]
    print('States look like:', state)
    state_size = len(state)
    print('States have length:', state_size)

    env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
    state = env_info.vector_observations[0]            # get the current state 
    scores = dqn(brain_name=brain_name, env=env, agent=agent, n_episodes=n_episodes, max_t = max_t, eps_start=eps_start, eps_end = eps_end, eps_decay = eps_decay, training_trial_number= trial_number)
    env.close()

    # Save results
    np.save(f"results/training_scores/scores_trial_{trial_number}.npy", np.array(scores))
    create_training_plot(scores = scores, trial_num= trial_number)

