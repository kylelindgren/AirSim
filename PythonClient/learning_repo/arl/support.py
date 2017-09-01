#!/usr/bin/env python
""" support.py
Support functions to connect to AirSim and others.
"""

__author__ = "Vinicius Guimaraes Goecks and Kyle Martin Lindgren"
__version__ = "0.0.0"
__status__ = "Prototype"
__date__ = "June 7, 2017"

# import
import math
import sys
import time
import cv2
import numpy as np
import resource
import gym
from gym import spaces
import pygame

import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy import signal

from PythonClient import *

from plotting import plot_history, process_avg
from gui import MachineGUI
from learn import ReplayBuffer
from neural import save_neural


def train_model(run_id, env, agent, n_episodes=1, n_steps=50):
    """
    Funtion to train a model or environment using a specific learning agent.

    Inputs
    ----------
    env: defined environment/model (should follow base.py).
    agent: learning agent
    n_episodes: number of episodes
    n_steps: number of steps per episode

    Outputs
    ----------
    total_s: all states tested.
    total_a: all actions applied.
    total_r: all rewards received.
    """

    # reset environment/model
    data_folder = '../data/'
    best_reward = -1e6
    total_done = 0.0

    # run for a given number of episodes
    for i_episode in range(n_episodes):
        # get initial states after reseting environment
        print('****************************')
        print('Episode %i/%i' % (i_episode+1, n_episodes))
        # print('Memory usage: %s (kb)' %
        #       resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

        # print('TESTING: took pic')
        # img = env.grab_depth()

        total_reward = 0
        start_reset_time = time.time()
        observation = env.reset()

        print('Total reset time: %.2f seconds.' %(time.time()-start_reset_time))

        # # TESTING MOVING TO A RANDOM POSITION
        # env.test_pos()

        # create object to store state-action pairs and rew-act pairs
        state_act = np.zeros((n_steps,observation.flatten().shape[0]+1)) # +1 for action, +1 for t
        rew_act = np.zeros((n_steps,3)) # step, rew, act
        recording_already = False # flag to help identifying human intervention
        start_interv = 0

        if agent.name == 'interv':
            x = state_act[0:1,:-1]
            y = state_act[0:1,-1]
            agent.update_net(x, y)

        start_epi_time = time.time()
        for t in range(n_steps):
            if t % 50 == 0:
                print 'Step ' + str(t)

            start_step_time = time.time()
            # select action based on current observation
            action = agent.act(observation)
            # print action

            # record past observation
            past_observation = np.copy(observation)

            # save past observation and action taken
            state_act[t,:] = np.hstack((past_observation.flatten(),action))

            # # check if human is intervening
            # if agent.name == 'interv':
            #     if agent.interv:
            #
            #         # update agent network
            #         x = state_act[t:t+1,:-1]
            #         y = state_act[t:t+1,-1]
            #         agent.update_net(x, y)

            # # check if human is intervening
            # if agent.name == 'interv':
            #     if agent.interv:
            #         if not recording_already:
            #             print('** START recording ...')
            #             start_interv = t
            #             recording_already = True
            #
            #     elif agent.interv == False:
            #         if recording_already:
            #             print('** END recording ...')
            #             recording_already = False
            #
            #     # update net when have a given number of samples
            #     if recording_already:
            #         batch_count = t - start_interv
            #         # print("Batch count = ", batch_count)
            #         if (batch_count) >= 31:
            #
            #             # update agent network
            #             x = state_act[start_interv:t+1,:-1]
            #             y = state_act[start_interv:t+1,-1]
            #             agent.update_net(x, y)
            #
            #             # reset times
            #             start_interv = t

            # execute selected action, get new observation and reward
            observation, reward, done, collision = env.step(action)
            total_reward += reward

            # stream to gui
            if env.use_gui:
                # pipe image and action to gui
                env.gui.display(past_observation,action)

            # save rew and action taken
            rew_act[t,:] = np.hstack((t,reward,action))

            if collision:
                print "Collision detected --> Ending episode."
                time.sleep(2)
                break

            # check if goal or if reached any other simulation limit
            if done:
                total_done = total_done + 1.0
                print("Episode finished after {} steps.".format(t + 1))
                break

            # # report time/frequency of steps
            # print('Running at %i Hz.' %(1/(time.time()-start_step_time)))

        # end of episode
        print '* End of episode *'
        print('Total episode time: %.2f seconds.' %(time.time()-start_epi_time))
        print('Total reward: %.2f' % total_reward)

        if total_reward > best_reward:
            # better episode so far, keep data
            print('Found best reward: %.2f' % total_reward)
            best_reward = total_reward

        # brake drone
        if env.inf_mode == False:
            env.drone_brake()

        # save total rewards
        agent.history[i_episode,:] = [i_episode, total_reward]

        # dump zero rows and save collected data
        state_act = state_act[~(state_act==0).all(1)]
        rew_act = rew_act[~(rew_act==0).all(1)]
        np.savetxt(data_folder+run_id+'_imit_'+str(i_episode)+'.csv', state_act, delimiter=',')
        np.savetxt(data_folder+run_id+'_avg_'+str(i_episode)+'.csv', rew_act, delimiter=',')

    # REPORT
    print('\nGoal achieved in %i out of %i tries.' % (total_done, n_episodes))
    print('Success rate = ', total_done / n_episodes)

    # save and plot reward results
    np.savetxt(data_folder+run_id+'_rew.csv', agent.history, delimiter=',')
    plot_history(agent.history)
    process_avg(run_id, n_episodes)

    # go home
    print 'Going HOME...' 
    env.drone_gohome()
    print 'DONE! Thank you! :)'

def train_model_multi(run_id, env, agent, n_episodes=1, n_steps=50):
    """
    Funtion to train a model or environment using a specific learning agent.
    Updated to work with multiple actions.

    Inputs
    ----------
    env: defined environment/model (should follow base.py).
    agent: learning agent
    n_episodes: number of episodes
    n_steps: number of steps per episode

    Outputs
    ----------
    total_s: all states tested.
    total_a: all actions applied.
    total_r: all rewards received.
    """

    # reset environment/model
    data_folder = '../data/'
    best_reward = -1e6
    total_done = 0
    n_act = 2
    updating_net = False

    # run for a given number of episodes
    for i_episode in range(n_episodes):
        # get initial states after reseting environment
        print('****************************')
        print('Episode %i/%i' % (i_episode+1, n_episodes))
        # print('Memory usage: %s (kb)' %
        #       resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

        total_reward = 0
        start_reset_time = time.time()
        observation = env.reset()
        print('Total reset time: %.2f seconds.' %(time.time()-start_reset_time))

        # create object to store state-action pairs and rew-act pairs
        state_act = np.zeros((n_steps,observation.flatten().shape[0]+n_act)) # +1 for action
        rew_act = np.zeros((n_steps,2+n_act)) # step, rew, act
        recording_already = False # flag to help identifying human intervention
        start_interv = 0

        # step required to start tensorflow library before flight
        if agent.name == 'interv':
            x = state_act[0:1,:-agent.n_act]
            y = state_act[0:1,-agent.n_act:]
            agent.update_net(x, y)

        start_epi_time = time.time()
        for t in range(n_steps):
            # report time steps
            # start_step_time = time.time()
            if t % 200 == 0:
                print('Time Step %i/%i' % (t, n_steps))

            # select action based on current observation
            env.t_step = t
            action = agent.act(observation)

            # record past observation
            past_observation = np.copy(observation)

            # save past observation and action taken
            state_act[t,:] = np.hstack((past_observation.flatten(),action.flatten()))

            # check if human is intervening
            if updating_net:
                if agent.name == 'interv':
                    if agent.interv:
                        if not recording_already:
                            print('** START recording ...')
                            start_interv = t
                            recording_already = True

                    elif agent.interv == False:
                        if recording_already:
                            print('** END recording ...')
                            recording_already = False

                    # update net when have a given number of samples
                    if recording_already:
                        batch_count = t - start_interv
                        # print("Batch count = ", batch_count)
                        # print("Step count = ", t)
                        if (batch_count) >= 31:

                            # update agent network
                            x = state_act[0:1,:-agent.n_act]
                            y = state_act[0:1,-agent.n_act:]
                            agent.update_net(x, y)

                            # reset times
                            start_interv = t

            # execute selected action, get new observation and reward
            observation, reward, done, _ = env.step_multi(action)
            total_reward += reward

            # work on replay when recording human data
            if recording_already:
                agent.replay.add(past_observation, action, reward, done,observation)

            # stream to gui
            if env.use_gui:
                # pipe image and action to gui
                env.gui.display(past_observation,action)

            # save rew and action taken
            rew_act[t,:] = np.hstack((t,reward,action))

            # check if goal or if reached any other simulation limit
            if done:
                print("Episode finished after {} steps.".format(t + 1))
                break

            # # report time/frequency of steps
            # print('Running at %i Hz.' %(1/(time.time()-start_step_time)))

        # end of episode
        print('* End of episode *')
        print('Total episode time: %.2f seconds.' %(time.time()-start_epi_time))

        if total_reward > best_reward:
            # better episode so far, keep data
            print('Found best reward: %.2f' % total_reward)
            best_reward = total_reward

        # brake drone
        if env.inf_mode == False:
            env.drone_brake()

        # save total rewards
        agent.history[i_episode,:] = [i_episode, total_reward]

        # dump zero rows and save collected data
        state_act = state_act[~(state_act==0).all(1)]
        rew_act = rew_act[~(rew_act==0).all(1)]
        env.hist_attitude = env.hist_attitude[~(env.hist_attitude==0).all(1)]
        np.savetxt(data_folder+run_id+'_imit_'+str(i_episode)+'.csv', state_act, delimiter=',')
        np.savetxt(data_folder+run_id+'_avg_'+str(i_episode)+'.csv', rew_act, delimiter=',')
        np.savetxt(data_folder+run_id+'_att_'+str(i_episode)+'.csv', env.hist_attitude, delimiter=',')
        # save_neural(agent.model, name=run_id)

    # REPORT
    print('\nGoal achieved in %i out of %i tries.' % (total_done, n_episodes))
    print('Success rate = ', total_done / n_episodes)

    # save and plot reward results
    print('Hold on! Saving data and plotting stuff!')
    save_neural(agent.model, name=run_id)
    np.savetxt(data_folder+run_id+'_rew.csv', agent.history, delimiter=',')
    plot_history(agent.history)
    process_avg(run_id, n_episodes)
    print('Done! Feel free to kill the process if stuck GOING HOME.')

    # go home
    print('Going HOME...')
    env.drone_gohome()
    print('DONE! Thank you! :)')

def train_deep_rl(run_id, env, agent, n_episodes=1, n_steps=50):
    """
    Funtion to train a model or environment using a specific learning agent.
    Updated to work with multiple actions.

    It integrates human demonstration to train the deep rl algorithm during the
    initial state-space exploration.

    Inputs
    ----------
    env: defined environment/model (should follow base.py).
    agent: learning agent
    n_episodes: number of episodes
    n_steps: number of steps per episode

    """

    # reset environment/model
    data_folder = '../data/'
    best_reward = -1e6
    total_done = 0
    n_act = env.act_n
    updating_net = False

    # run for a given number of episodes
    for i_episode in range(n_episodes):
        # get initial states after reseting environment
        print('****************************')
        print('Episode %i/%i' % (i_episode+1, n_episodes))
        # print('Memory usage: %s (kb)' %
        #       resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

        total_reward = 0
        start_reset_time = time.time()
        observation = env.reset()

        print('Total reset time: %.2f seconds.' %(time.time()-start_reset_time))

        # create object to store state-action pairs and rew-act pairs
        state_act = np.zeros((n_steps,observation.flatten().shape[0]+n_act)) # +1 for action
        rew_act = np.zeros((n_steps,2+n_act)) # step, rew, act
        recording_already = False # flag to help identifying human intervention
        start_interv = 0

        # step required to start tensorflow library before flight
        if agent.name == 'interv':
            x = state_act[0:1,:-agent.n_act]
            y = state_act[0:1,-agent.n_act:]
            agent.update_net(x, y)

        start_epi_time = time.time()
        for t in range(n_steps):
            # report time steps
            # start_step_time = time.time()
            if t % 200 == 0:
                print('Time Step %i/%i' % (t, n_steps))

            # select action based on current observation
            env.t_step = t
            agent.t = t
            action = agent.act(observation)

            # record past observation
            past_observation = np.copy(observation)

            # save past observation and action taken
            state_act[t,:] = np.hstack((past_observation.flatten(),action))

            # execute selected action, get new observation and reward
            observation, reward, done, _ = env.step_dqn(action)

            # return max reward if the agent is intervening
            if agent.interv:
                reward = 1
                # print('Human reward = ',reward)
            total_reward += reward

            # # work on replay
            if agent.has_replay:
                agent.use_replay(past_observation, action, reward, done,
                                 observation)

            # stream to gui
            if env.use_gui:
                # pipe image and action to gui
                env.gui.display(past_observation,action)

            # save rew and action taken
            rew_act[t,:] = np.hstack((t,reward,action))

            # check if goal or if reached any other simulation limit
            # print(done)
            if done:
                print("Episode finished after {} steps.".format(t + 1))
                break

            # # report time/frequency of steps
            # print('Running at %i Hz.' %(1/(time.time()-start_step_time)))

        # end of episode
        print('* End of episode *')
        print('Total episode time: %.2f seconds.' %(time.time()-start_epi_time))

        if total_reward > best_reward:
            # better episode so far, keep data
            print('Found best reward: %.2f' % total_reward)
            best_reward = total_reward

        # brake drone
        if env.inf_mode == False:
            env.drone_brake()

        # save total rewards
        agent.history[i_episode,:] = [i_episode, total_reward]

        # dump zero rows and save collected data
        state_act = state_act[~(state_act==0).all(1)]
        rew_act = rew_act[~(rew_act==0).all(1)]
        env.hist_attitude = env.hist_attitude[~(env.hist_attitude==0).all(1)]
        np.savetxt(data_folder+run_id+'_imit_'+str(i_episode)+'.csv', state_act, delimiter=',')
        np.savetxt(data_folder+run_id+'_avg_'+str(i_episode)+'.csv', rew_act, delimiter=',')
        np.savetxt(data_folder+run_id+'_att_'+str(i_episode)+'.csv', env.hist_attitude, delimiter=',')
        save_neural(agent.model, name=run_id+'_'+str(i_episode))

    # REPORT
    print('\nGoal achieved in %i out of %i tries.' % (total_done, n_episodes))
    print('Success rate = ', total_done / n_episodes)

    # save and plot reward results
    print('Hold on! Saving data and plotting stuff!')
    save_neural(agent.model, name=run_id)
    np.savetxt(data_folder+run_id+'_rew.csv', agent.history, delimiter=',')
    plot_history(agent.history)
    process_avg(run_id, n_episodes)
    print('Done! Feel free to kill the process if stuck GOING HOME.')

    # go home
    print('Going HOME...')
    env.drone_gohome()
    print('DONE! Thank you! :)')

def train_human(run_id, env, agent, n_episodes=1, n_steps=50):
    """
    Funtion to train a model or environment using a specific learning agent.
    Updated to work with multiple actions.

    It integrates human demonstration to train the deep rl algorithm during the
    initial state-space exploration.

    Inputs
    ----------
    env: defined environment/model (should follow base.py).
    agent: learning agent
    n_episodes: number of episodes
    n_steps: number of steps per episode

    """

    # reset environment/model
    data_folder = '../data/'
    best_reward = -1e6
    total_done = 0
    n_act = 2
    updating_net = False

    # run for a given number of episodes
    for i_episode in range(n_episodes):
        # get initial states after reseting environment
        print('****************************')
        print('Episode %i/%i' % (i_episode+1, n_episodes))
        # print('Memory usage: %s (kb)' %
        #       resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

        total_reward = 0
        start_reset_time = time.time()
        observation = env.reset()

        print('Total reset time: %.2f seconds.' %(time.time()-start_reset_time))

        # create object to store state-action pairs and rew-act pairs
        state_act = np.zeros((n_steps,observation.flatten().shape[0]+n_act)) # +1 for action
        rew_act = np.zeros((n_steps,2+n_act)) # step, rew, act
        recording_already = False # flag to help identifying human intervention
        start_interv = 0


        start_epi_time = time.time()
        for t in range(n_steps):
            # report time steps
            # start_step_time = time.time()
            if t % 200 == 0:
                print('Time Step %i/%i' % (t, n_steps))

            # select action based on current observation
            env.t_step = t
            action = agent.act(observation)

            # record past observation
            past_observation = np.copy(observation)

            # save past observation and action taken
            state_act[t,:] = np.hstack((past_observation.flatten(),action.flatten()))

            # execute selected action, get new observation and reward
            observation, reward, done, _ = env.step_multi(action)
            total_reward += reward

            # work on replay when recording human data
            if recording_already:
                agent.replay.add(past_observation, action, reward, done,observation)

            # stream to gui
            if env.use_gui:
                # pipe image and action to gui
                env.gui.display(past_observation,action)

            # save rew and action taken
            rew_act[t,:] = np.hstack((t,reward,action))

            # check if goal or if reached any other simulation limit
            if done:
                print("Episode finished after {} steps.".format(t + 1))
                break

            # # report time/frequency of steps
            # print('Running at %i Hz.' %(1/(time.time()-start_step_time)))

        # end of episode
        print('* End of episode *')
        print('Total episode time: %.2f seconds.' %(time.time()-start_epi_time))

        if total_reward > best_reward:
            # better episode so far, keep data
            print('Found best reward: %.2f' % total_reward)
            best_reward = total_reward

        # brake drone
        if env.inf_mode == False:
            env.drone_brake()

        # save total rewards
        agent.history[i_episode,:] = [i_episode, total_reward]

        # dump zero rows and save collected data
        state_act = state_act[~(state_act==0).all(1)]
        rew_act = rew_act[~(rew_act==0).all(1)]
        env.hist_attitude = env.hist_attitude[~(env.hist_attitude==0).all(1)]
        np.savetxt(data_folder+run_id+'_imit_'+str(i_episode)+'.csv', state_act, delimiter=',')
        np.savetxt(data_folder+run_id+'_avg_'+str(i_episode)+'.csv', rew_act, delimiter=',')
        np.savetxt(data_folder+run_id+'_att_'+str(i_episode)+'.csv', env.hist_attitude, delimiter=',')
        # save_neural(agent.model, name=run_id)

    # REPORT
    print('\nGoal achieved in %i out of %i tries.' % (total_done, n_episodes))
    print('Success rate = ', total_done / n_episodes)

    # save and plot reward results
    print('Hold on! Saving data and plotting stuff!')
    save_neural(agent.model, name=run_id)
    np.savetxt(data_folder+run_id+'_rew.csv', agent.history, delimiter=',')
    plot_history(agent.history)
    process_avg(run_id, n_episodes)
    print('Done! Feel free to kill the process if stuck GOING HOME.')

    # go home
    print('Going HOME...')
    env.drone_gohome()
    print('DONE! Thank you! :)')


class CustomAirSim(AirSimClient, gym.Env):
    """
    Custom class for handling AirSim commands like connect, takeoff, land, etc.
    """
    def __init__(self,n_steps, inf_mode=False, use_gui=False):
        AirSimClient.__init__(self, "127.0.0.1") # connect to ip
        self.max_timeout = 20 # seconds
        # self.connect_AirSim()  # moved to reset() - KL
        self.inf_mode = inf_mode

        # possible bank actions (m/s)
        self.actions = [.5, 0, -.5]
        self.dt = .05 # seconds, interval between actions
        self.forward_vel = 2
        self.vy_scale = 2  
        self.n_steps = n_steps

        # parameters for turn maneuver
        self.old_yaw = 0
        self.old_roll = 0
        self.set_z = -5

        # action limits
        self.act_low = -1
        self.act_high = 1
        self.act_n = 1 # one action
        self.map_length = 110 # 380

        # depth parameters and compression factors
        # depth 1/4: 36 x 64 pics
        # depth 1/8: 18 x 32 pics
        depth_width = 256
        depth_height = 144
        self.reduc_factor = 0.25  # 1/4 KL # multiply original size by this factor
        self.depth_w = int(depth_width*self.reduc_factor)
        self.depth_h = int(depth_height*self.reduc_factor)

        # gui parameters
        self.use_gui = use_gui
        if self.use_gui:
            # create gui object
            self.gui = MachineGUI(self.depth_w, self.depth_h, start_gui=True)


        # store current and reference states
        # step, roll, pitch, yaw (current), roll, pitch, yaw (ref)
        self.t_step = 0
        # step, 3 pos, 3 vel, 3 att, 3 ref att
        self.hist_attitude = np.zeros((n_steps,13))

    # CUSTOM ARL COMMANDS
    def setOffboardModeTrue(self):
        return self.client.call('setOffboardMode', True)
    #####################

    def connect_AirSim(self):
        """
        Establish initial connection to AirSim client and set GPS.
        """
        # get GPS
        print("Waiting for home GPS location to be set...")
        # home = self.getHomePoint()
        home = self.getPosition()
        while ((home[0] == 0 and home[1] == 0 and home[2] == 0) or
               math.isnan(home[0]) or  math.isnan(home[1]) or  math.isnan(home[2])):
            time.sleep(1)
            # home = self.getHomePoint()
            home = self.getPosition()

        print("Home lat=%g, lon=%g, alt=%g" % tuple(home))

        # save home position and gps
        self.home = home
        self.home_pos = self.getPosition()

    def drone_takeoff(self):
        """
        Takeoff function.
        """
        # # arm drone
        # if (not self.arm()):
        #     print("Failed to arm the drone.")
        #     sys.exit(1)
        #
        # # takeoff
        # if (self.getLandedState() == LandedState.Landed):
        #     print("Taking off...")
        #     if (not self.takeoff(20)):
        #         print("Failed to reach takeoff altitude after 20 seconds.")
        #         sys.exit(1)
        #     print("Should now be flying...")
        # else:
        #     print("It appears the drone is already flying")

        # stabilize
        # for i in range(np.abs(self.set_z)):
        #     self.drone_altitude(-i)
        # try:
        #     self.drone_altitude(self.set_z)
        #     self.takeoff(1)
        # except:
        #     self.drone_altitude(self.set_z)
        # self.drone_altitude(self.set_z)

        self.arm()
        self.setOffboardModeTrue()
        time.sleep(3)
        self.takeoff(5)
        time.sleep(3)
        # self.hover()
        # time.sleep(6)
        # self.drone_altitude(self.set_z)
        # time.sleep(6)

    def drone_land(self):
        """
        Land function.
        """
        if (self.getLandedState() != LandedState.Landed):
            print("Landing...")
            if (not self.land(20)):
                print("Failed to reach takeoff altitude after 60 seconds.")
                sys.exit(1)
            print("Landed.")
        else:
            print("It appears the drone is already landed.")

    def drone_forward(self, vx, angle=0):
        """
        Go forward on camera frame with vx speed (m/s).
        """
        # define move parameters
        drivetrain = DrivetrainType.MaxDegreeOfFreedom
        yaw_mode = YawMode(False, angle)
        duration = self.dt # seconds
        self.moveByVelocityZ(vx, 0, 0, duration, drivetrain, yaw_mode)

        return duration

    def drone_bank(self, vy):
        """
        Bank on camera frame with vy speed (m/s).
        """
        # define move parameters
        drivetrain = DrivetrainType.MaxDegreeOfFreedom
        yaw_mode = YawMode(False,0)
        duration = self.dt # seconds
        command = np.clip(vy*self.vy_scale,-self.vy_scale,self.vy_scale)

        self.moveByVelocityZ(self.forward_vel, command, self.set_z, duration, drivetrain, yaw_mode)

        return duration

    def drone_turn(self, vy):
        """
        Combines pitch, roll, and yaw for turn maneuvers.
        """
        # parse commands
        if vy == 2: # pitch forward
            # define angular motion
            pitch = -10
            roll = 0
            yaw = self.old_yaw

        elif vy == 3: # break
            # define angular motion
            pitch = 10
            roll = 0
            yaw = self.old_yaw

        else:
            # define angular motion
            pitch = -1
            roll = 15*vy
            yaw = self.old_yaw + vy*2

        # send commands
        self.client.call('moveByAngle', pitch, roll, self.set_z, yaw, self.dt)

        # store applied yaw so we can send cumulative commands to change attitude
        self.old_yaw = yaw

        return self.dt

    def drone_turn_multi(self, act):
        """
        Combines pitch, roll, and yaw for turn maneuvers.
        Updated to work with multiple actions.
        """
        # parse commands
        lat = act[0]
        lon = act[1]

        # # scale actions to adequate control inputs
        # pitch = .5*lon
        # yaw = self.old_yaw + lat*1
        # if lat == 0:
        #     roll = 0
        # else:
        #     # roll = self.old_roll + lat*2
        #     # roll = np.clip(roll,-20,20)
        #     roll = lat*2
        #
        # # send commands
        # self.client.call('moveByAngle', pitch, roll, self.set_z, yaw, self.dt)
        #
        # # store applied yaw so we can send cumulative commands to change attitude
        # self.old_yaw = yaw
        # self.old_roll = roll

        # define move parameters
        drivetrain = DrivetrainType.MaxDegreeOfFreedom
        yaw_mode = YawMode(False,0)
        duration = self.dt # seconds
        command = np.clip(lat*self.vy_scale,-self.vy_scale,self.vy_scale)

        self.moveByVelocityZ(-self.forward_vel*lon, command, self.set_z, duration, drivetrain, yaw_mode)

        # # save ref controls and attitude
        # curr_pos = self.getPosition()
        # curr_vel = self.getVelocity()
        # curr_att = np.rad2deg(self.getRollPitchYaw())
        #
        # self.hist_attitude[self.t_step,:] = [self.t_step,
        #                                      curr_pos[0],
        #                                      curr_pos[1],
        #                                      curr_pos[2],
        #                                      curr_vel[0],
        #                                      curr_vel[1],
        #                                      curr_vel[2],
        #                                      curr_att[1],
        #                                      -curr_att[0],
        #                                      curr_att[2],
        #                                      roll,
        #                                      pitch,
        #                                      yaw]

        return self.dt


    def test_pos(self):
        """
        Testing moveToPosition. Going to a random Y position.
        """
        # define move parameters
        z = -5 # altitude
        velocity = 5
        drivetrain = DrivetrainType.MaxDegreeOfFreedom
        yaw_mode = YawMode(False, 0)
        self.moveToPosition(0, -30, z, velocity, 30, drivetrain, yaw_mode, 0, 1)

    def take_pic(self):
        """
        Return rgb image.
        """
        # get rgb image
        result = self.setImageTypeForCamera(0, AirSimImageType.Scene)
        result = self.getImageForCamera(0, AirSimImageType.Scene)
        if (result != "\0"):
            # rgb
            rawImage = np.fromstring(result, np.int8)
            png = cv2.imdecode(rawImage, cv2.IMREAD_UNCHANGED)
            cv2.imwrite('rgb.png',png)

    def grab_rgb(self):
        """
        Get camera rgb image and return array of pixel values.
        Returns numpy ndarray.
        """
        # get rgb image
        result = self.setImageTypeForCamera(0, AirSimImageType.Scene)
        result = self.getImageForCamera(0, AirSimImageType.Scene)
        if (result != "\0"):
            # rgb
            rawImage = np.fromstring(result, np.int8)
            png = cv2.imdecode(rawImage, cv2.IMREAD_UNCHANGED)

            return png

    def simGetImage(self, camera_id, image_type):
        # because this method returns std::vector<uint8>, msgpack decides to encode it as a string unfortunately.
        result = self.client.call('simGetImage', camera_id, image_type)
        if (result == "" or result == "\0"):
            return None
        return np.fromstring(result, np.int8)

    def grab_depth(self):
        """
        Get camera depth image and return array of pixel values.
        Returns numpy ndarray.
        """
        # get depth image
        # result = self.setImageTypeForCamera(0, AirSimImageType.Depth)
        result = self.simGetImage(0, AirSimImageType.Depth)
        if (result is not None):
            # depth
            rawImage = np.fromstring(result, np.int8)
            png = cv2.imdecode(rawImage, cv2.IMREAD_UNCHANGED)
            if png is not None:
                # return pic, only first channel is enough for depth
                # apply threshold
                # png[:,:,0] = self.tsh_distance(100,png[:,:,0])
                # cv2.imshow('depth', png[:, :, 2])
                # cv2.waitKey(1)
                # airsim update changes depth image output - KL
                return png[:, :, 2]
            else:
                print('Couldnt take one depth pic.')
                return np.zeros((144,256)) # empty picture

    def take_depth_pic(self):
        """
        Return depth image.
        """
        # get depth image
        result = self.setImageTypeForCamera(0, AirSimImageType.Depth)
        result = self.getImageForCamera(0, AirSimImageType.Depth)
        if (result != "\0"):
            # depth
            rawImage = np.fromstring(result, np.int8)
            png = cv2.imdecode(rawImage, cv2.IMREAD_UNCHANGED)
            if png is not None:
                print('Depth pic taken.')
                png[:,:,0] = self.tsh_distance(100,png[:,:,0])
                cv2.imwrite('depth_tsh.png',png[:,:,0])
            else:
                print('Couldnt take one depth pic.')

    def tsh_distance(self, tsh_val, img):
        """
        Threshold pixel values on image. Set to zero if less than tsh_val.
        img vals < 100 indicate 20+ meters depth 
            (figures->validate_depth->pixel_distance)
        """
        low_val_idx = img < tsh_val
        img[low_val_idx] = 0

        return img

    def validate_depth(self):
        """
        Validate depth image, trying to make the ground thruth more similar
        to real sensors.
        For example, ZED Camera Depth only sees between 0.5-20 meters, blurred.
        """
        # takeoff if landed
        if (self.getLandedState() == LandedState.Landed):
            print("Landed.")
            try:
                self.drone_takeoff()
            except:
                print("Takeoff failed. Trying again...")
                # might need to reconnect first
                CustomAirSim.__init__(self)
                self.drone_takeoff()

            # arm drone
            if (not self.arm()):
                print("Failed to arm the drone.")
                sys.exit(1)

        # take a few pics
        # time.sleep(25)
        for i in range(5):
            self.take_depth_pic()

    def drone_brake(self):
        """
        Brake the drone and maintain altitude.
        """
        # break drone
        z = -3 # altitude
        drivetrain = DrivetrainType.MaxDegreeOfFreedom
        yaw_mode = YawMode(False, 0)
        duration = 1 # seconds
        self.moveByVelocityZ(0, 0, z, duration, drivetrain, yaw_mode)
        time.sleep(duration)
        self.moveByVelocityZ(0, 0, z, duration, drivetrain, yaw_mode)
        time.sleep(duration)

    def drone_altitude(self, alt):
        """
        Changes drone's altitude.
        """
        max_wait_seconds = 5
        drivetrain = DrivetrainType.MaxDegreeOfFreedom
        yaw_mode = YawMode(False,0)

        print('Changing altitude to %i meters...' %(alt*(-1)))
        self.moveToZ(alt, 3, max_wait_seconds, yaw_mode, 0, 1)
        # lookahead distance not 0 in moveToZ() makes unreal engine crash
        # instances: drone_altitude(), correct_altitude(), drone_gohome()
        # self.moveToZ(alt, 3, max_wait_seconds, yaw_mode, 1, 1)
        time.sleep(-alt)

    def correct_altitude(self):
        """
        Changes drone's altitude.
        """
        drivetrain = DrivetrainType.MaxDegreeOfFreedom
        yaw_mode = YawMode(False,0)
        self.moveToZ(self.set_z, 3, 0, yaw_mode, 0, 1)
        # self.moveToZ(self.set_z, 3, 0, yaw_mode, 1, 1)        

    def drone_gohome(self):
        """
        Climb high, go home, and land.
        """
        # make sure offboard mode is on
        self.setOffboardModeTrue()

        # move home
        print('Moving to HOME position...')

        # compute distance from home
        dist = self.dist_home()

        # while (self.getLandedState() != LandedState.Landed):
        while dist > 10:
            self.goHome()
            time.sleep(5)

            # compute distance from home
            dist = self.dist_home()

        print('Close enough.')
        z = -3
        max_wait_seconds = 30
        drivetrain = DrivetrainType.MaxDegreeOfFreedom
        yaw_mode = YawMode(False,0)

        print('Descending to %i meters...' %(z*(-1)))
        self.moveToZ(z, 10, max_wait_seconds, yaw_mode, 0, 1)
        # self.moveToZ(z, 10, max_wait_seconds, yaw_mode, 1, 1)
        time.sleep(max_wait_seconds)

    def dist_home(self):
        """
        Compute current distance from home point.
        """
        current_pos = self.getPosition()
        dist = np.sqrt((current_pos[0] - self.home_pos[0])**2 + (current_pos[1] - self.home_pos[1])**2)

        return dist

    def report_status(self):
        """
        Report position, velocity, and other current states.
        """
        print("* STATUS *")
        print("Position lat=%g, lon=%g, alt=%g" % tuple(self.getGpsLocation()))
        print("Velocity vx=%g, vy=%g, vz=%g" % tuple( self.getVelocity()))
        print("Attitude pitch=%g, roll=%g, yaw=%g" % tuple(self.getRollPitchYaw()))

    def step(self, action):
        """
        Step agent based on computed action.
        Return reward and if check if episode is done.
        """
        # take action
        # wait_time = self.drone_turn(float(action))
        wait_time = self.drone_bank(float(action))
        time.sleep(wait_time)
        self.correct_altitude()

        # get next state
        img = self.grab_depth()
        res = self.preprocess(img)

        # compute reward
        reward = self.compute_reward(res)

        # check if done
        current_pos = self.getPosition()
        if current_pos[0] < self.map_length:
            done = 0
        else:
            done = 1

        return res, reward, done, {}

    def step_dqn(self, action):
        """
        Step agent based on computed action.
        Return reward and if check if episode is done.
        """
        # map action (example: convert from 0,1,2 to -1,0,1)
        action = action - 1

        # wait_time = self.drone_turn(float(action))
        wait_time = self.drone_bank(float(action))
        time.sleep(wait_time)

        # get next state
        img = self.grab_depth()
        res = self.preprocess(img)

        # compute reward
        reward = self.compute_reward(res)

        # check if done
        current_pos = self.getPosition()
        if current_pos[0] < self.map_length: # 50 for small course / 105 for big
            done = 0
        else:
            done = 1

        return res, reward, done, {}

    def step_multi(self, action):
        """
        Step agent based on computed action.
        Return reward and if check if episode is done.
        """
        # take action
        wait_time = self.drone_turn_multi(action)
        time.sleep(wait_time)

        # get next state
        img = self.grab_depth()
        res = self.preprocess(img)

        # compute reward
        reward = self.compute_reward(res)

        # check if done
        done = 0
        current_pos = self.getPosition()
        if current_pos[0] < 110: # 50 for small course / 105 for big
            done = 0
        else:
            done = 1

        return res, reward, done, {}

    def compute_reward(self, img):
        """
        Compute reward based on image received.
        """
        # normalize pixels
        # all white = 0, all black = 1
        reward = 1 - np.sum(img) / (img.shape[0] * img.shape[1])  # (36*64)
        return reward

    def reset(self):
        """
        Take initial camera data.
        """
        # # make sure offboard mode is on
        # self.setOffboardModeTrue()

        self.connect_AirSim()  # moved to here from __init__() - KL
            # causes issue with ground vehicle not needing/able to takeoff

        # skip going home if "inf mode"
        if self.inf_mode:
            print("Inf Mode: Data saved. Keep going.")

            # takeoff if landed
            if (self.getLandedState() == LandedState.Landed):
                print("Landed.")
                try:
                    self.drone_takeoff()
                except:
                    print("Takeoff failed. Trying again...")
                    # might need to reconnect first
                    CustomAirSim.__init__(self)
                    self.drone_takeoff()

                # arm drone
                if (not self.arm()):
                    print("Failed to arm the drone.")
                    sys.exit(1)

        else:
            # go home
            if (self.getLandedState() != LandedState.Landed):
                self.drone_gohome()
            else:
                try:
                    self.drone_takeoff()
                except:
                    print("Takeoff failed. Trying again. Reconnecting...")
                    # might need to reconnect first
                    CustomAirSim.__init__(self, self.n_steps)
                    self.drone_takeoff()

            # arm drone
            if (not self.arm()):
                print("Failed to arm the drone.")
                sys.exit(1)

        # get next state
        img = self.grab_depth()
        res = self.preprocess(img)

        return res

    def preprocess(self, img):
        """
        Resize image. Converts and down-samples the input image.
        """
        # resize img
        # cv2.imshow('orig', img)
        res = cv2.resize(img, None, fx=self.reduc_factor, fy=self.reduc_factor, interpolation=cv2.INTER_AREA)
        # cv2.imshow('processed', res)
        # cv2.waitKey(1)
        # normalize image
        res = res / 255.0  # change 255 -> 255.0 for float calc - KL (Py2.7)
        # print res
        # cv2.imshow('processed, normalized', res)
        # cv2.waitKey(1)

        return res

    @property
    def action_space(self):
        """
        Maximum roll command. It can be scaled after.
        """
        return spaces.Box(low=np.array(-1),
                          high=np.array(1))

    @property
    def observation_space(self):
        """
        2D image of depth sensor.
        """
        screen_height = 144
        screen_width = 256
        # frames = 1
        return spaces.Box(low=0, high=255, shape=(screen_height, screen_width))


class GroundAirSim(CustomAirSim, gym.Env):
    """
    Custom class for handling Ground AirSim commands.
    """
    def __init__(self, n_steps, inf_mode=False, use_gui=False):
        super(GroundAirSim, self).__init__(n_steps, inf_mode, use_gui)
        self.inf_mode = inf_mode

        self.forward_vel = 1.5
        self.turn_scale = 1  
        self.prev_xy_key = 9
        self.prev_th_key = 9

        self.ran_start = True

        # parameters for turn maneuver
        # 1.19 = collision, 1.18 no collision
        self.set_z = 0.97  # ground vehicle height # 0.22  # ~9 inches

        self.tsh_val = 249  # img vals < 100 ~ 20+m, 180 ~ 5m, 220 ~ 2.5m
        self.crop_depth_h_frac = 0.25
        self.crop_depth_w_frac = 0.25

        # computer vision params
        pos = self.getPosition()
        orq = self.getOrientation()
        # self.pos0, self.orq0 = (pos, orq)
        self.pos0, self.orq0 = ([0.0, 0.0, self.set_z], [1.0, 0.0, 0.0, 0.0])
        self.pos_new, self.orq_new = (pos, orq)

        # action limits
        self.map_length = 30

    # CUSTOM ARL GROUND VEHICLE COMMANDS
    #####################


    def drone_turn(self, key):
        """
        Sets position and orientation for cv mode ground vehicle.
        """
        duration = self.dt

        pos = self.getPosition()
        orq = self.getOrientation()
        ore = self.toEulerianAngle(orq)

        # if key == 0:
        #     self.prev_xy_key = self.prev_th_key = 0
        # else:            
        #     if key == -1:
        #         ore = [ore[0], ore[1], ore[2] - scale_th*duration]
        #         orq = self.toQuaternion(ore)
        #     elif key == 1:
        #         ore = [ore[0], ore[1], ore[2] + scale_th*duration]
        #         orq = self.toQuaternion(ore)

        # if abs(key) > 1:
        #     key = np.clip(key, -1, 1)
        #     print "clipped key: " + str(key)
        ore = [ore[0], ore[1], ore[2] + key*self.turn_scale*duration]
        orq = self.toQuaternion(ore)

        # constant velocity forward
        pos[0] = pos[0] + math.cos(ore[2])*self.forward_vel*duration
        pos[1] = pos[1] + math.sin(ore[2])*self.forward_vel*duration

        self.simSetPose(pos, orq)

        return duration

    def drone_bank(self, key):
        """
        Move laterally
        """
        # define move parameters
        duration = self.dt

        pos = self.getPosition()
        orq = self.getOrientation()

        # print key
        # key = 0.0 if abs(key) < 0.12 else key

        # constant velocity forward (x direction)
        pos[0] = pos[0] + self.forward_vel*duration
        pos[1] = pos[1] + (key+0)*self.turn_scale*duration  
        # 400->+2.6754899 100->-0.33134952 200->+0.38967273 500->+0.11149321
        # print key

        self.simSetPose(pos, orq)

        return duration

    def drone_brake(self):
        """
        Brake the ground vehicle by canceling turn commands.
        Sends arbitrary turn 'key'
        """
        self.drone_turn(0)  
        

    def drone_gohome(self):
        """
        Go home.
        """
        # move high above obstacles, then down to home position
        pos = self.getPosition()
        orq = self.getOrientation()
        self.simSetPose([pos[0], pos[1], -50.0], orq)
        self.simSetPose([pos[0] - 1, pos[1], -50.0], orq)
        self.simSetPose([pos[0] + 1, pos[1], -50.0], orq)
        self.simSetPose([pos[0], pos[1] - 1, -50.0], orq)
        self.simSetPose([pos[0], pos[1] + 1, -50.0], orq)
        self.simSetPose([self.pos0[0], self.pos0[1], -50.0], self.orq0)
        if self.ran_start:
            self.simSetPose([self.pos0[0], self.pos0[1] + 1.5*float(np.random.randint(-1, 2, 1)), self.pos0[2]], self.orq0)
        else:
            self.simSetPose(self.pos0, self.orq0)

        time.sleep(2)

    # def gohome_turtle(self):
    #     """
    #     Climb high, go home, and land.
    #     """
    #     # make sure offboard mode is on
    #     self.setOffboardModeTrue()

    #     # move home
    #     print('Moving to turtle HOME position...')

    #     # compute distance from home
    #     dist = self.dist_home()

    #     while dist > 1:
    #         self.goHome()
    #         time.sleep(5)

    #         # compute distance from home
    #         dist = self.dist_home()

    #     print('Close enough.')
    #     z = self.set_z
    #     max_wait_seconds = 30
    #     drivetrain = DrivetrainType.MaxDegreeOfFreedom
    #     yaw_mode = YawMode(False, 0)

    #     print('Descending to %i meters...' %(z*(-1)))
    #     self.moveToZ(z, 10, max_wait_seconds, yaw_mode, 0, 1)
    #     # self.moveToZ(z, 10, max_wait_seconds, yaw_mode, 1, 1)
    #     time.sleep(max_wait_seconds)


    def step(self, action):
        """
        Step ground agent in cv mode based on computed action.
        Return reward and if check if episode is done.
        """
        # take action
        # wait_time = self.drone_turn(float(action))
        wait_time = self.drone_bank(float(action))
        time.sleep(wait_time)

        # get next state
        img = self.grab_depth()
        res = self.preprocess(img)

        # compute reward
        # crop res to ignore some of ground plane immediately in front
        reward = self.compute_reward(res)
        # print reward

        # check if done
        current_pos = self.getPosition()
        done = 0 if current_pos[0] < self.map_length else 1

        # from AirSim/AirLib/include/api/RpcLibAdapters.hpp
        # struct CollisionInfo {
        #     bool has_collided = false;
        #     Vector3r normal;
        #     Vector3r impact_point;
        #     Vector3r position;
        #     msr::airlib::real_T penetration_depth = 0;
        #     msr::airlib::TTimePoint time_stamp = 0;
        #     ...
        collision = self.getCollisionInfo()
        # if collision[0] is True:
        #     print "Collision detected --> Resetting to Home position."
        #     self.drone_gohome()

        return res, reward, done, collision[0]


    def step_dqn(self, action):
        """
        Step agent based on computed action.
        Return reward and if check if episode is done.
        """
        # map action (example: convert from 0,1,2 to -1,0,1)
        # action = action - 1  # from CustomAirSim class - KL

        wait_time = self.drone_turn(float(action))
        time.sleep(wait_time)

        # get next state
        img = self.grab_depth()
        res = self.preprocess(img)

        # compute reward
        reward = self.compute_reward(res)

        # check if done
        current_pos = self.getPosition()
        if current_pos[0] < self.map_length: # 50 for small course / 105 for big
            done = 0
        else:
            done = 1

        return res, reward, done, {}

    def compute_reward(self, img):
        """
        Compute reward based on image received.
        """
        # normalize pixels
        # all white = 0, all black = 1
        blurred = np.ones(img.shape) - img
        gaus_filter = signal.gaussian(blurred.shape[1], std=6)
        # print gaus_filter
        for i in range(blurred.shape[0]):
            blurred[i, :] = 2*np.multiply(blurred[i, :], gaus_filter)
        cv2.imshow('gaussian blurred', blurred)
        # cv2.imshow('orig', img)
        cv2.waitKey(1)
        
        # reward = 1 - np.sum(img) / (img.shape[0] * img.shape[1])
        reward = np.sum(blurred) / (blurred.shape[0] * blurred.shape[1])
        # print "gaussed"
        # print blurred
        # print np.sum(blurred[1, :])
        # print reward

        return reward

    def reset(self):
        """
        Ground vehicle does not takeoff or need home set (cv mode) like 
        flying drones. Returns observation.
        """
        # get next state
        self.drone_gohome()  # send back to starting position
        img = self.grab_depth()
        res = self.preprocess(img)

        return res

    def preprocess(self, img):
        """
        Resize image. Converts and down-samples the input image.
        """
        # resize img
        res = cv2.resize(img, None, fx=self.reduc_factor, fy=self.reduc_factor, interpolation=cv2.INTER_AREA)
        # cv2.imshow('processed', res)
        # cv2.waitKey(1)
        # normalize image

        res = self.tsh_distance(self.tsh_val, res)
        
        height = res.shape[0]
        width = res.shape[1]
        res = res[int(self.crop_depth_h_frac*height):int(height - 2*self.crop_depth_h_frac*height), 
        int(self.crop_depth_w_frac*width):int(width - self.crop_depth_w_frac*width)] / 255.0
        # :] / 255.0
        # int(self.crop_depth_frac*width):int(width - self.crop_depth_frac*width)] / 255.0  # change 255 -> 255.0 for float calc - KL (Py2.7)
        
        # cv2.imshow('processed, normalized', res)
        # cv2.waitKey(1)
        # print "res shape: " + str(res.shape)  # (9, 32)

        return res

    def tsh_distance(self, tsh_val, img):
        """
        Threshold pixel values on image. Set to zero if less than tsh_val.
        img vals < 100 indicate 20+ meters depth 
            (figures->validate_depth->pixel_distance)
        """
        low_val_idx = img < tsh_val
        img[low_val_idx] = 0

        return img

    @property
    def action_space(self):
        """
        Maximum roll command. It can be scaled after.
        """
        return spaces.Box(low=np.array(-1),
                          high=np.array(1))
        # return spaces.Discrete(3)  # idk - KL

    @property
    def observation_space(self):
        """
        2D image of depth sensor.
        """
        screen_height = 144
        screen_width = 256
        # frames = 1
        return spaces.Box(low=0, high=255, shape=(screen_height, screen_width))

