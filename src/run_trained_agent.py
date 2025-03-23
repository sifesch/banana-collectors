import torch
from unityagents import UnityEnvironment
import numpy as np
from dqn_agent import Agent

class Trained_Agent:
    def __init__(self,file_name):
        self.env = UnityEnvironment(file_name=file_name)
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]

        state_size = self.brain.vector_observation_space_size
        action_size = self.brain.vector_action_space_size
        self.agent = Agent(state_size=state_size, action_size=action_size, seed=0)

    def run_trained_agent(self, trial_number: str):
        self.agent.qnetwork_local.load_state_dict(torch.load(f'results/model_weights/checkpoint_{trial_number}.pth'))
        env_info = self.env.reset(train_mode=False)[self.brain_name] 
        state = env_info.vector_observations[0]
        score = 0
        
        while True:
            action = self.agent.act(state, eps=0.0)  
            env_info = self.env.step(action)[self.brain_name]  
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            score += reward
            state = next_state 

            if done:
                break  
        print(f"Score: {score}")
        self.env.close()

if __name__ == "__main__":
    file_name = "../Banana_Linux/Banana.x86_64"
    agent = Trained_Agent(file_name=file_name)
    agent.run_trained_agent(trial_number='05')