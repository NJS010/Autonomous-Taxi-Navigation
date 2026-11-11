import gymnasium as gym
import numpy as np
import random
import time
import os

# PHASE 1: TRAINING
env = gym.make("Taxi-v4") # Load env

# Initialize Q-table with zeros (500 states x 6 actions)
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparameters
alpha = 0.1      # Learning rate
gamma = 0.6      # Discount factor (future rewards)
epsilon = 0.1    # Exploration rate

print("Training the AI Agent for 10,000 episodes... please wait.")

for episode in range(1, 10001):
    state, info = env.reset()
    done = False
    
    while not done:
        # 10% chance to explore, 90% chance to exploit best move
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() 
        else:
            action = np.argmax(q_table[state]) 
            
        # Take action
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Bellman Equation: Update Q-table
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        
        q_table[state, action] = new_value
        state = next_state

print("Training complete!")
env.close()

# PHASE 2: SIMULATION
# Reload env in text mode for the terminal
env = gym.make("Taxi-v4", render_mode="ansi")
state, info = env.reset()
done = False

print("\nStarting simulation in 3 seconds...")
time.sleep(3)

steps = 0
while not done:
    os.system('clear') # Clear screen for animation
    
    print(env.render())
    print(f"Step: {steps}")
    
    # Pick the best action from the trained Q-table
    action = np.argmax(q_table[state])
    state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    steps += 1
    time.sleep(0.5) # Pause to see the movement

os.system('clear')
print(env.render())
print(f"Goal reached in {steps} steps!")
env.close()