# Doudizhu PPO Implementation
### Deep Reinforcement Learning Semester Project @ UOS

This folder holds the implementation of our PPO Agent for the game of Dou Dizhu.
The agent is located in ``agents/``. To allow the the play of an on-policy agent, 
changes to the original rlcard library have been made, that mostly deal with the returned values of
the agents' step function, since we now are returning ``action, log_prob, value, entropy`` of each step taken, 
compared to the previous ``action, log_prob``. 

### Usage

To run the PPO agents, call 
``
python doudizhu_ppo.py.
``
from this directory.

### Disclaimer
Our PPO agent is not able to outperform the random agent when trained on a comparable
amount of games played like the DQN agent.



