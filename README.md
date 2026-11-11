# Autonomous Taxi RL Agent 🚕

My first Machine Learning project using Reinforcement Learning! This AI agent learns how to navigate a city grid to pick up a passenger and drop them off at their destination without breaking the rules.

## How it Works
The project uses **Q-Learning** and the `Taxi-v4` environment from the Gymnasium library. 
* **Phase 1 (Training):** The agent trains for 10,000 trips in the background, learning from its mistakes through trial and error to build a Q-Table.
* **Phase 2 (Simulation):** The trained agent uses its Q-Table to navigate the grid flawlessly, which you can watch live in the terminal.

## Setup & Installation

1. Install the required libraries:
```bash
pip install gymnasium numpy