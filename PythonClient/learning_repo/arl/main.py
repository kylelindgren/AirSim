#!/usr/bin/env python
""" main.py
OpenAI-Gym-like wrap on custom AirSim class so past written Deep RL algorithm
can be easily applied.
"""

__author__ = "Vinicius Guimaraes Goecks"
__version__ = "0.0.0"
__status__ = "Prototype"
__date__ = "June 12, 2017"

# import
import numpy as np
import gym
import time
import sys

import argparse
from gym import spaces

from PythonClient import *

from support import CustomAirSim, GroundAirSim, train_model_multi, train_model, train_deep_rl, train_human
from learn import RandomAgent, HumanAgent, ImitationAgent, InterventionAgent
from learn import InterventionAgentMulti, DQN_AirSim, HumanAgentXBox, HumanAgentXBoxMulti

import configparser

def main(inf_mode=False, use_gui=False):
    """
    Testing multiple inheritance and wrapping functions.
    """
    # start config parser
    config = configparser.ConfigParser()
    config.read('config_main.ini')

    main_setup = config['DEFAULT']

    # parameters and modes
    n_episodes = main_setup.getint('n_episodes')
    n_steps = main_setup.getint('n_steps')

    # initial setup
    inf_mode = main_setup.getint('inf_mode')
    use_gui = main_setup.getint('use_gui')
    if main_setup['ground']:
        drone = GroundAirSim(n_steps, inf_mode=inf_mode, use_gui=use_gui)
    else:
        drone = CustomAirSim(n_steps, inf_mode=inf_mode, use_gui=use_gui)

    # start learning agents
    select_agent = main_setup['agent']
    print('AGENT: {}'.format(select_agent))
    if select_agent == 'human':
        agent = HumanAgent(drone, n_episodes)  # changed to use kb - KL
        # agent = HumanAgentXBoxMulti(drone, n_episodes)
        # drone.cycle_start = True
        drone.ran_start = True
    elif select_agent == 'imitation':
        # drone.cycle_start = True  # cycle through 3 start locations
        drone.ran_start = True  # random start
        agent = ImitationAgent(drone, n_episodes)
    elif select_agent == 'interv':
        agent = InterventionAgentMulti(drone, n_episodes)
    elif select_agent == 'dqn':
        # parameters
        gamma = 0.9
        eps_max = 1
        eps_min = 1
        lbd = 0.001
        batch_size = 64
        buffer_size = 10000
        target_update_freq = 100
        dqn_agent = main_setup['dqn_agent']

        # learning agent
        agent = DQN_AirSim(drone,
                    n_steps=n_steps,
                    n_episodes=n_episodes,
                    gamma=gamma,
                    eps_min=eps_min,
                    eps_max=eps_max,
                    lbd=lbd,
                    batch_size=batch_size,
                    buffer_size=buffer_size,
                    target_update_freq=target_update_freq,
                    pre_fill_buffer=True,
                    target=False,
                    eval_factor=20,
                    dqn_agent=dqn_agent)
    else:
        print('Invalid agent. Please check main.ini file')
        sys.exit(1)

    # turtle! - KL
    if main_setup['ground'] == 'True':
        agent.ground = True

    # ros = main_setup['ros']
    if main_setup['ros'] == 'True':
        agent.init_ros()

    # test a few iterations
    run_id = main_setup['run_id']

    # select experiment mode
    exp_mode = main_setup.getint('exp')
    print('Experiment # {}:'.format(exp_mode))
    if exp_mode == 0:
        print('Imitation Learning with only Convolutional layers.')
        train_model(run_id, drone, agent, n_episodes, n_steps)
    elif exp_mode == 1:
        print('Imitation Learning with Convolutional and Recurrent layers.')
        train_model_multi(run_id, drone, agent, n_episodes, n_steps)
    elif exp_mode == 2:
        print('Training Deep RL algorithms based on human demonstration.')
        train_deep_rl(run_id, drone, agent, n_episodes, n_steps)
    elif exp_mode == 3:
        print('Validating the depth sensor.')
        drone.validate_depth()
    elif exp_mode == 4:
        print('Training human and getting data.')
        train_human(run_id, drone, agent, n_episodes, n_steps)


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--autoland", type=bool, help="lands drone if code crashes.")
    parser.add_argument("--gui", type=bool, help="activates live gui.")
    parser.add_argument("--inf", type=bool, help="runs ad infinitum.")
    args = parser.parse_args()

    # check autoland and run main
    if args.autoland:
        try:
            main()
        except:
            print("* CODE CRASHED *")
            drone = CustomAirSim()
            # land
            drone.drone_land()
            drone.disarm()

    elif args.inf:
        if args.gui:
            print('Using GUI.')
            main(inf_mode=True, use_gui=True)

        else:
            main(inf_mode=True, use_gui=False)
    else:
        main()
