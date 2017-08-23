Data won't be uploaded here for now because of the size.

# Experiment - 001 ('exp1') - 6/16/2017
20 episodes  
100 steps max  
Human agent - collecting data for imitation learning
2 hz or .5sec of dt

# Experiment - 002 ('exp2') - 6/17/2017
20 episodes  
750 steps max  
Human agent - collecting data for imitation learning
20 hz or .05sec of dt

# Experiment - 003 ('dqn0') - 7/24/2017
60 episodes  
200 steps max  
forward = 2.5, z = -5
DQN agent guided by human
(Human would still pilot the quad, but states and actions will be used to train
a deep q-network)
20 hz or .05sec of dt

# Experiment - 004 (dqn0_compare_imit_0) - 7/28/2017
10 episodes  
70 steps max  
Comparing performance of DQN and imitation network
DQN net: dqn_imit_99
IMIT net: eval_imit_19
Want to compare their performance based on how often the imitation suggests actions.
Runs:
1. dqn0_compare_imit_0 (0% of imitation learning interventions)
1. dqn0_compare_imit_30 (30% of imitation learning interventions)
1. dqn0_compare_imit_50 (50% of imitation learning interventions)
1. dqn0_compare_imit_80 (80% of imitation learning interventions)
1. dqn0_compare_imit_100 (100% of imitation learning interventions)
