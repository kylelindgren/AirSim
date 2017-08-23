#!/usr/bin/env python
"""Test Universe

Run test script from OpenAI Universe Webpage.
"""

__author__ = "Vinicius Guimaraes Goecks"
__version__ = "0.0.0"
__status__ = "Prototype"
__date__ = "May 16, 2017"

# import
import gym
import universe  # register the universe environments

env = gym.make('flashgames.DuskDrive-v0')
env.configure(remotes=1)  # automatically creates a local docker container
observation_n = env.reset()

while True:
  action_n = [[('KeyEvent', 'ArrowUp', True)] for ob in observation_n]  # your agent here
  observation_n, reward_n, done_n, info = env.step(action_n)
  env.render()
