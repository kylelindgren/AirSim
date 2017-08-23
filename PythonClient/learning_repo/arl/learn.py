#!/usr/bin/env python
""" learn.py:
Learning algorithms and agents for Reinforcement Learning.
"""

__author__ = "Vinicius Guimaraes Goecks"
__version__ = "0.0.0"
__status__ = "Prototype"
__date__ = "June 13, 2017"

# import
import numpy as np
from collections import deque
from multiprocessing import Pool, Process
import os
import itertools
import random
import gym
import sys
import tensorflow as tf
import tflearn
from sklearn.preprocessing import StandardScaler, scale

import pygame, time           # for keyboard multi-threaded inputs
from pygame.locals import *

from neural import DeepNeuralNet, load_neural, save_neural, ImitationNetwork
from keras.utils import plot_model
from keras.optimizers import SGD

# ===========================
#   Intervention Agent (Multi Actions)
# ===========================
class InterventionAgentMulti(object):
    """
    Agent that chooses actions based on a pre-trained network, but still allow
    human to take over its control.
    Taylored only for AirSim for now.
    Updated to work with multiple actions.
    """

    def __init__(self, env, n_episodes):
        self.name = 'interv'
        # get env characteristics
        self.action_space = env.action_space
        self.drone = env

        # flag to use replay buffer
        self.has_replay = False
        self.history = np.zeros((n_episodes, 2))  # step and rew

        # depth data resolution
        self.width = 64
        self.height = 36
        self.n_act = 2 # number of actions

        # create replay buffer
        buffer_size = 32
        self.replay = ReplayBuffer(buffer_size)

        # load network
        self.mode = 'load' # or 'new' network

        if self.mode == 'load':
            net_name = 'exp4_imit_conv_lstm'
            self.net_name = 'conv_lstm'
            print('Loading neural network: ', net_name)
            self.model = load_neural(name=net_name, loss='mse', opt='adam')

        elif self.mode == 'new':
            print('Creating new neural network ...')
            self.net_name = 'blank_net'
            net = ImitationNetwork(n_act=self.n_act)
            self.model = net.model

        # # print summary and plot model
        # print(self.model.summary())
        # plot_model(self.model, to_file='model.png')

        # intervention flag to allow human control
        self.interv = True # True = human start commanding
        self.interv_time = 0 # measure interval between each intervention

        # initialize pygame for inputs
        pygame.display.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((500, 120))
        pygame.display.set_caption('CLICK HERE TO CONTROL DRONE :)')
        time.sleep(3)

        self.show_score('N/A','N/A','N/A')

        # repeating last action
        self.last_act = np.zeros(self.n_act)


    def get_input(self):
        """
        Get input from keyboard.
        """
        # start with past action and no changes in altitude
        act = self.last_act
        lat = self.last_act[0]
        lon = self.last_act[1]

        # check for human intervention (space bar pressed)
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_SPACE:
                    # reverse interv flag
                    self.interv = not self.interv
                    # if intervention, start timer
                    self.interv_time = time.time()

                # LATERAL COMMANDS
                if event.key == pygame.K_d:
                    lat = 1

                if event.key == pygame.K_a:
                    lat = -1

                if event.key == pygame.K_s:
                    lat = 0

                # LONGITUDINAL COMMANDS
                if event.key == pygame.K_o:
                    lon = -1 # speed forward

                if event.key == pygame.K_k:
                    lon = 0 # break drone

                if event.key == pygame.K_m:
                    lon = 1 # break drone

        act = np.array([lat,lon])

        return act

    def act(self, img_input):
        """
        Predict action based on image input. Returns action.
        """
        # read act from keyboard
        act = self.get_input()

        # if under human control
        if self.interv:
            # record selected action
            self.last_act = act

        else:
            # get number of images
            samples = 1 #img_input.shape[0]

            # reshape input to fit on keras
            img_input = img_input.reshape((samples, self.height, self.width, 1))
            act = self.model.predict(img_input, batch_size=1, verbose=2)

            # fix shape of outuputs
            act = act[0]

        # display status
        if self.interv:
            current_time = time.time() - self.interv_time
        else:
            current_time = 'N/A'
        self.show_score(act, self.interv, current_time)

        return act

    def show_score(self, act, interv, interv_time):
        """
        PyGame function that display score on screen.
        """
        text_color = (0, 255, 0)
        # pygame parameters
        myfont = pygame.font.SysFont('C:/Windows/Fonts/calibri.ttf', 22)
        self.screen.fill((0,0,0))

        # define message
        msg_text0 = 'A: bank left | S: center | D: bank right | W: fast forward | F: brake'
        textsurface0 = myfont.render(msg_text0, False, text_color)
        self.screen.blit(textsurface0,(10,10))

        msg_text1 = 'Actions taken [lat lon]: ' + str(act)
        textsurface1 = myfont.render(msg_text1, False, text_color)
        self.screen.blit(textsurface1,(10,40))

        msg_text2 = 'Human controlling? ' +  str(interv) + ' | Intervention time (sec): ' + str(interv_time)[:5]
        textsurface2 = myfont.render(msg_text2, False, text_color)
        self.screen.blit(textsurface2,(10,70))

        pygame.display.update()

    def update_net(self, x, y):
        """
        Update network with recorded states (x) and actions (y) during human
        intervention.
        """
        # print('Update network with data from human intervention...')
        samples = x.shape[0]
        # print('Number of samples: ', samples)

        x = x.reshape((samples, self.height, self.width, 1))
        y = y.reshape((samples, self.n_act))

        self.model.fit(x, y, batch_size=32,
                         epochs=1,
                         verbose=0,
                         shuffle=True)

        print('Network updated.')


# ===========================
#   Intervention Agent
# ===========================
class InterventionAgent(object):
    """
    Agent that chooses actions based on a pre-trained network, but still allow
    human to take over its control.
    Taylored only for AirSim for now.
    """

    def __init__(self, env, n_episodes):
        self.name = 'interv'
        # get env characteristics
        self.action_space = env.action_space

        # flag to use replay buffer
        self.has_replay = False
        self.history = np.zeros((n_episodes, 2))  # step and rew

        # depth data resolution
        self.width = 64
        self.height = 36

        # load network
        self.mode = 'load' # or 'new' network

        if self.mode == 'load':
            net_name = 'trained_imit_exp2_big'
            print('Loading neural network: ', net_name)
            self.model = load_neural(name=net_name, loss='mse', opt='adam')

        elif self.mode == 'new':
            print('Creating new neural network ...')
            net = ImitationNetwork()
            self.model = net.model

        # # print summary and plot model
        # print(self.model.summary())
        # plot_model(self.model, to_file='model.png')

        # intervention flag to allow human control
        self.interv = False
        self.interv_time = 0 # measure interval between each intervention

        # initialize pygame for inputs
        pygame.display.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((500, 120))
        pygame.display.set_caption('CLICK HERE TO CONTROL DRONE :)')
        time.sleep(3)

        self.show_score('N/A','N/A','N/A')

        # repeating last action
        self.last_act = 0

    def get_input(self):
        """
        Get input from keyboard.
        """
        # start with past action and no changes in altitude
        act = self.last_act

        # check for human intervention (space bar pressed)
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_SPACE:
                    # reverse interv flag
                    self.interv = not self.interv
                    # if intervention, start timer
                    self.interv_time = time.time()

                if event.key == pygame.K_d:
                    act = self.action_space.high

                if event.key == pygame.K_a:
                    act = self.action_space.low

                if event.key == pygame.K_s:
                    act = 0 # no action

                if event.key == pygame.K_w:
                    act = 2 # speed forward

                if event.key == pygame.K_f:
                    act = 3 # break drone

        return act

    def act(self, img_input):
        """
        Predict action based on image input. Returns action.
        """
        # read act from keyboard
        act = self.get_input()

        # if under human control
        if self.interv:
            # record selected action
            self.last_act = act

        else:
            # get number of images
            samples = 1 #img_input.shape[0]

            # reshape input to fit on keras
            img_input = img_input.reshape((samples, self.height, self.width, 1))
            act = self.model.predict(img_input, batch_size=1, verbose=2)

            # fix shape, return on value
            act = act[0]

        # display status
        if self.interv:
            current_time = time.time() - self.interv_time
        else:
            current_time = 'N/A'
        self.show_score(act, self.interv, current_time)

        return act

    def show_score(self, act, interv, interv_time):
        """
        PyGame function that display score on screen.
        """
        text_color = (0, 255, 0)
        # pygame parameters
        myfont = pygame.font.SysFont('C:/Windows/Fonts/calibri.ttf', 22)
        self.screen.fill((0,0,0))

        # define message
        msg_text0 = 'A: bank left | S: center | D: bank right | W: fast forward | F: brake'
        textsurface0 = myfont.render(msg_text0, False, text_color)
        self.screen.blit(textsurface0,(10,10))

        msg_text1 = 'Action taken: ' + str(act)[:7] + ' | Human controlling? ' +  str(interv)
        textsurface1 = myfont.render(msg_text1, False, text_color)
        self.screen.blit(textsurface1,(10,40))

        msg_text2 = 'Intervention time (sec): ' + str(interv_time)[:5]
        textsurface2 = myfont.render(msg_text2, False, text_color)
        self.screen.blit(textsurface2,(10,70))

        pygame.display.update()

    def update_net(self, x, y):
        """
        Update network with recorded states (x) and actions (y) during human
        intervention.
        """
        # print('Update network with data from human intervention...')
        samples = x.shape[0]
        # print('Number of samples: ', samples)

        x = x.reshape((samples, self.height, self.width, 1))
        y = y.reshape((samples, 1))

        self.model.fit(x, y, batch_size=32,
                         epochs=1,
                         verbose=0,
                         shuffle=True)

        # print('Network updated.')


# ===========================
#   Imitate Agent
# ===========================
class ImitationAgent(object):
    """
    Agent that chooses actions based on a pre-trained network.
    Taylored only for AirSim for now.
    """

    def __init__(self, env, n_episodes):
        self.name = 'imitation'
        # get env characteristics
        self.action_space = env.action_space

        # flag to use replay buffer
        self.has_replay = False
        self.history = np.zeros((n_episodes, 2))  # step and rew
        self.n_act = 1

        # depth data resolution
        self.width = 64
        self.height = 36

        # load network
        print('Loading neural network...')
        self.normalized = False
        self.model = load_neural(name='eval_imit_19', loss='mse', opt='adam')
        # lrate = .01
        # epochs = 300
        # decay = lrate/epochs
        # momentum = .9
        # sgd = SGD(lr=lrate, momentum=momentum, decay=decay, nesterov = False)
        # self.model = load_neural(name='model', loss='mse', opt=sgd)

    def act(self, img_input):
        """
        Predict action based on image input. Returns action.
        """
        # get number of images
        samples = 1 #img_input.shape[0]

        # reshape input to fit on keras
        if self.normalized:
            img_input = img_input.reshape((samples, self.height, self.width, 1))
        else:
            img_input = 255*img_input.reshape((samples, self.height, self.width, 1))
        act = self.model.predict(img_input, batch_size=1, verbose=2)
        # print(act)

        # # random action
        # act = [np.random.randint(-1, 1, 1)]

        return act[0]


# ===========================
#   Human Agent
# ===========================
class HumanAgent(object):
    """
    Human agent sending actions to RL environments.
    Taylored only for AirSim for now.
    """

    def __init__(self, env, n_episodes):
        self.name = 'human'
        self.ground = False  # KL
        # get env characteristics
        self.action_space = env.action_space

        # flag to use replay buffer
        self.has_replay = False
        self.history = np.zeros((n_episodes, 2))  # step and rew

        # initialize pygame for inputs
        pygame.display.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((500, 100))
        pygame.display.set_caption('CLICK HERE TO CONTROL DRONE :)')
        time.sleep(3)

        self.show_score('N/A')

        # repeating last action
        self.last_act = 0

    # def act_event(self, observation):
    #     # start with past action and returns flag if action is new
    #     act = self.last_act
    #     new_act = False

    #     # check if there is any key pressed
    #     if self.ground:  # ground support - KL
    #         for event in pygame.event.get():
    #             if event.type == pygame.KEYDOWN:
    #                 new_act = True
    #                 if event.key == pygame.K_UP:
    #                     act = 8

    #                 if event.key == pygame.K_DOWN:
    #                     act = 2

    #                 if event.key == pygame.K_LEFT:
    #                     act = 4

    #                 if event.key == pygame.K_RIGHT:
    #                     act = 6

    #                 if event.key == pygame.K_w:
    #                     act = 5

    #                 if event.key == pygame.K_a:
    #                     act = 1

    #                 if event.key == pygame.K_d:
    #                     act = 3

    #                 if event.key == pygame.K_w:
    #                     act = 0

    #                 if event.key == pygame.K_SPACE:
    #                     act = 9  # stop
    #     else:
    #         for event in pygame.event.get():
    #             if event.type == pygame.KEYDOWN:
    #                 new_act = True
    #                 if event.key == pygame.K_d:
    #                     act = self.action_space.high

    #                 if event.key == pygame.K_a:
    #                     act = self.action_space.low

    #                 if event.key == pygame.K_s:
    #                     act = 0 # no action

    #                 if event.key == pygame.K_w:
    #                     act = 2 # speed forward

    #                 if event.key == pygame.K_f:
    #                     act = 3 # break drone

    #     # record selected action
    #     self.last_act = act
    #     self.show_score(act)

    #     return act, new_act

    def act(self, observation):
        # start with past action
        act = self.last_act

        # check if there is any key pressed
        if self.ground:  # ground support - KL
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    
                    if event.key == pygame.K_UP:
                        act = 8

                    if event.key == pygame.K_DOWN:
                        act = 2

                    if event.key == pygame.K_LEFT:
                        act = 4

                    if event.key == pygame.K_RIGHT:
                        act = 6

                    if event.key == pygame.K_w:
                        act = 5

                    if event.key == pygame.K_a:
                        act = 4

                    if event.key == pygame.K_d:
                        act = 6

                    if event.key == pygame.K_w:
                        act = 8

                    if event.key == pygame.K_SPACE:
                        act = 9  # stop
        else:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    
                    if event.key == pygame.K_d:
                        act = self.action_space.high

                    if event.key == pygame.K_a:
                        act = self.action_space.low

                    if event.key == pygame.K_s:
                        act = 0 # no action

                    if event.key == pygame.K_w:
                        act = 2 # speed forward

                    if event.key == pygame.K_f:
                        act = 3 # break drone

        self.last_act = act
        self.show_score(act)
        return act

    def show_score(self, act):
        """
        PyGame function that display score on screen.
        """
        # pygame parameters
        myfont = pygame.font.SysFont('monospace', 24)
        self.screen.fill((0,0,0))

        # define message
        msg_text0 = 'A: bank left | S: center | D: bank right'
        textsurface0 = myfont.render(msg_text0, False, (255, 255, 255))
        self.screen.blit(textsurface0,(10,10))

        msg_text1 = 'Action taken: ' + str(act)
        textsurface1 = myfont.render(msg_text1, False, (255, 255, 255))
        self.screen.blit(textsurface1,(10,40))

        pygame.display.update()

# ===========================
#   Human Agent - XBOX CONTROLLER
# ===========================
class HumanAgentXBox(object):
    """
    Human agent sending actions to RL environments.
    Taylored only for AirSim for now.
    Designed to use the XBox Controller.
    """

    def __init__(self, env, n_episodes):
        self.name = 'human'
        # get env characteristics
        self.action_space = env.action_space

        # flag to use replay buffer
        self.has_replay = False
        self.history = np.zeros((n_episodes, 2))  # step and rew

        # initialize pygame for inputs using the Xbox controller
        pygame.init()
        self.my_joystick = pygame.joystick.Joystick(0)
        self.my_joystick.init()


    def act(self, observation):
        # read controller
        pygame.event.get()
        act = self.my_joystick.get_axis(0) # 0 = left stick

        # flush rest and display
        pygame.event.clear()

        return act

# ===========================
#   Human Agent Multi - XBOX CONTROLLER
# ===========================
class HumanAgentXBoxMulti(object):
    """
    Human agent sending actions to RL environments.
    Taylored only for AirSim for now.
    Designed to use the XBox Controller.
    """

    def __init__(self, env, n_episodes):
        self.name = 'human'
        self.ground = False  # KL
        # get env characteristics
        self.action_space = env.action_space

        # flag to use replay buffer
        self.has_replay = False
        self.history = np.zeros((n_episodes, 2))  # step and rew
        self.n_act = 2 # number of actions

        # initialize pygame for inputs using the Xbox controller
        pygame.init()
        self.my_joystick = pygame.joystick.Joystick(0)
        self.my_joystick.init()


    def act(self, observation):
        # read controller
        pygame.event.get()
        lat = self.my_joystick.get_axis(0) # 0 = left stick lat
        lon = self.my_joystick.get_axis(1) # 0 = left stick long

        act = np.array([lat,lon])

        # flush rest and display
        pygame.event.clear()

        return act



# ===========================
#   Deep-Q Networks
# ===========================
class DQN_AirSim(object):
    """
    Deep Q-Network implementation based on the famous Atari work.
    Modified to work with AirSim.
    """

    def __init__(self,
                 env,
                 n_steps=50,
                 n_episodes=1,
                 gamma=.9,
                 eps_min=.1,
                 eps_max=1,
                 lbd=.001,
                 batch_size=64,
                 buffer_size=100000,
                 target_update_freq=100,
                 pre_fill_buffer=False,
                 target=False,
                 eval_factor=100,
                 dqn_agent=None):

        # agent info
        self.name = 'DQN_AirSim'
        self.history = np.zeros((n_episodes, 2))  # step and rew
        self.rand_counter = 0
        self.act_counter = 0
        self.env = env
        self.dqn_agent = dqn_agent

        # get env characteristics
        self.action_space = env.action_space

        # depth data resolution
        self.width = 64
        self.height = 36

        self.n_ob = self.height*self.width # 1 image with height and width, 1 channel
        self.n_ac = 1 # only lat or long and lat actions
        self.total_n_ac = 3 # -1, 0, 1

        # intervention flag to allow human control
        self.interv = False
        self.interv_time = 0 # measure interval between each intervention

        # parameters
        self.t = 0
        self.n_steps = n_steps
        self.n_episodes = n_episodes
        self.gamma = gamma
        self.epsilon = eps_max
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.lbd = lbd
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.target_update_freq = target_update_freq
        self.target = target

        # can manually update epsilon
        self.last_epsilon = eps_max
        self.delta_eps = 0

        # can load imitation netowrk to bypass random actions
        print('Loading imitation network...')
        self.use_imitation = True
        self.imit_model = load_neural(name='eval_imit_19', loss='mse', opt='adam')

        # credit assignment and inverse rl
        self.r_max = 1
        self.similarity = 0 # between action from imit and deep rl
        self.credit = 0

        # neural models (original and target)
        if self.dqn_agent != 'none':
            net_name = self.dqn_agent
            print('Loading DQN agent ', net_name)
            self.model = load_neural(name=net_name, loss='mse', opt='adam')
        else:
            dqn_model = ImitationNetwork(dqn=True)
            self.model = dqn_model.model
        if target:
            self.target_model = self.model

        # start replay buffer
        self.replay = ReplayBuffer(buffer_size)
        self.has_replay = True
        if pre_fill_buffer:
            self.fill_buffer()

        # initialize pygame for inputs
        pygame.display.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((500, 120))
        pygame.display.set_caption('CLICK HERE TO CONTROL DRONE :)')
        time.sleep(3)

        self.show_score('N/A','N/A','N/A')

        # repeating last action
        self.last_act = 0

    def manual_epsilon(self):
        """
        Update epsilon from keyboard inputs.
        """
        # start with past action and no changes in altitude
        eps = self.last_epsilon

        # update upsilon based on its current value
        eps = self.last_epsilon + self.delta_eps
        eps = np.clip(eps,0,1) # should be between 0 and 100 %

        self.last_epsilon = eps
        self.delta_eps = 0

        return eps

    def get_input(self):
        """
        Get input from keyboard.
        """
        # start with past action and no changes in altitude
        act = self.last_act

        # check for human intervention (space bar pressed)
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_SPACE:
                    # reverse interv flag
                    self.interv = not self.interv
                    # if intervention, start timer
                    self.interv_time = time.time()

                if event.key == pygame.K_d:
                    act = 2 #self.action_space.high

                if event.key == pygame.K_a:
                    act = 0 #self.action_space.low

                if event.key == pygame.K_s:
                    act = 1 # no action

                if event.key == pygame.K_o:
                    self.delta_eps = .1

                if event.key == pygame.K_k:
                    self.delta_eps = -.1

                if event.key == pygame.K_u:
                    self.credit = 10

                if event.key == pygame.K_h:
                    self.credit = -10

        return act

    def act(self, state):
        """
        Check if human is controlling or use DQN.
        """
        def start_imit_deep_rl(command):
            '''
            Function that handles computation of imitation action and depe rl
            using different cores of the cpu.
            '''
            if command == 'imit':
                state = 255*self.state.reshape((1, self.height, self.width, 1))
                sugg_act = self.imit_model.predict(state, batch_size=1, verbose=2)
                sugg_act = np.clip(sugg_act,-1,1)
                sugg_act = int(np.rint(sugg_act[0])) + 1
                print('Imitation would suggest {} action.'.format(sugg_act))
                return sugg_act

            if command == 'deep_rl':
                qval = self.model.predict(
                    self.state.reshape((1, self.height, self.width, 1)),
                    batch_size=1)
                # print('Q-values at time {}: {}'.format(self.t,qval))

                # choose the best action from Q(s,a) values
                act = (np.argmax(qval))
                print('DQN would suggest {} action.'.format(act))

                return act

        # read act from keyboard and store state
        act = self.get_input()
        self.state = state

        # check for credit assignment
        if self.credit != 0:
            self.replay.credit_assignment(self.credit)
            self.credit = 0

        # manually update epsilon
        self.epsilon = self.manual_epsilon()

        # if under human control
        if self.interv:
            # record selected action
            self.last_act = act

        else:

            # apply e-greedy policy (chance of picking a random action)
            if (random.random() < self.epsilon):
                if self.use_imitation:
                    print('Imitation action.')
                    state = 255*state.reshape((1, self.height, self.width, 1))
                    act = self.imit_model.predict(state, batch_size=1, verbose=2)
                    act = np.clip(act,-1,1)
                    act = int(np.rint(act[0]))
                    # convert to 0,1,2 scheme
                    act = act + 1
                else:
                    print('Random action.')
                    #choose random action between 0,1,2
                    act = np.random.randint(0, 3, self.n_ac)
                    act = act[0]
                    self.rand_counter += 1
            else:
                # MAIN GOAL HERE:
                # the idea is to create two separated process (multiprocessing,
                # sending each process to a different cpu core) to speed up computation
                # of the imitation learning suggestion and the deep rl one.
                # when receive their actions, compare them and compute a intrinsic
                # reward out of that.

                ### OLD CODE BELOW, IN CASE IT DOESNT WORK
                # print('DQN Action.')
                # # run Q function on states (obs) to get Q values
                # # for all possible actions
                # qval = self.model.predict(
                #     state.reshape((1, self.height, self.width, 1)),
                #     batch_size=1)
                # # print('Q-values at time {}: {}'.format(self.t,qval))
                #
                # # compare DQN with imitation
                # state = 255*state.reshape((1, self.height, self.width, 1))
                # sugg_act = self.imit_model.predict(state, batch_size=1, verbose=2)
                # sugg_act = np.clip(sugg_act,-1,1)
                # sugg_act = int(np.rint(sugg_act[0])) + 1
                # print('Imitation would suggest {} action.'.format(sugg_act))
                #
                # # choose the best action from Q(s,a) values
                # act = (np.argmax(qval))
                # print(act)
                # self.act_counter += 1
                ###############################

                # start multithread to compute actions from imitation and deep rl
                with Pool(2) as p:
                    mp_actions = p.map(start_imit_deep_rl, ['imit','deep_rl'])

                # split computed actions from multiprocessing
                sugg_act = mp_actions[0]
                act = mp_actions[1]
                print(act.__class__)

        # display status
        if self.interv:
            current_time = time.time() - self.interv_time
        else:
            current_time = 'N/A'
        self.show_score(act, self.interv, current_time)

        return act

    def show_score(self, act, interv, interv_time):
        """
        PyGame function that display score on screen.
        """
        text_color = (0, 255, 0)
        # pygame parameters
        myfont = pygame.font.SysFont('C:/Windows/Fonts/calibri.ttf', 22)
        self.screen.fill((0,0,0))

        # define message
        msg_text0 = 'A: bank left | S: center | D: bank right'
        textsurface0 = myfont.render(msg_text0, False, text_color)
        self.screen.blit(textsurface0,(10,10))

        msg_text1 = 'Action taken: ' + str(act)[:7] + ' | Human controlling? ' +  str(interv)
        textsurface1 = myfont.render(msg_text1, False, text_color)
        self.screen.blit(textsurface1,(10,40))

        msg_text2 = 'Intervention time (sec): ' + str(interv_time)[:5]
        textsurface2 = myfont.render(msg_text2, False, text_color)
        self.screen.blit(textsurface2,(10,70))

        msg_text2 = 'Chance of Random Action: ' + str(self.epsilon*100)[:3] + ' %'
        textsurface2 = myfont.render(msg_text2, False, text_color)
        self.screen.blit(textsurface2,(10,100))

        pygame.display.update()

    def update_net(self, x, y):
        """
        Update network with recorded states (x) and actions (y) during human
        intervention.
        """
        # print('Update network with data from human intervention...')
        samples = x.shape[0]
        # print('Number of samples: ', samples)

        x = x.reshape((samples, self.height, self.width, 1))
        y = y.reshape((samples, 1))

        self.model.fit(x, y, batch_size=32,
                         epochs=1,
                         verbose=0,
                         shuffle=True)

        # print('Network updated.')

    def update_epsilon(self):
        '''
        Update epsilon according to episode count
        '''
        self.epsilon = self.eps_min + (self.eps_max - self.eps_min) * np.exp(
            -self.lbd * self.t)

    def predict(self, s, target=False):
        '''
        Predict action based on state.
        '''
        # reshape to image format
        s = s.reshape((1, self.height, self.width, 1))

        if target:
            return self.target_model.predict(s, batch_size=1)
        else:
            return self.model.predict(s, batch_size=1)

    def use_replay(self, state, action, reward, done, next_state):
        '''
        Add experience to replay and train deep q-network.
        '''
        # add last experience to buffer
        self.replay.add(state, action, reward, done, next_state)

        # check if need to update target network when using it
        if self.target and (self.t % self.target_update_freq == 0):
            self.target_model = self.neural.copy_weights(self.model,
                                                         self.target_model)

        self.t += 1
        # self.update_epsilon()

        # sample buffer
        mem_s0_all, mem_a_all, mem_rew_all, mem_done_all, mem_s1_all = self.replay.sample_batch(self.batch_size)

        # get train values for each experience sampled
        n_samples = mem_a_all.shape[0]
        X_train = np.zeros((n_samples, self.n_ob))
        y_train = np.zeros((n_samples, self.total_n_ac))
        for i in range(n_samples):
            # get data for each minimatch
            mem_s0 = mem_s0_all[i,:]
            mem_a = mem_a_all[i]
            mem_rew = mem_rew_all[i]
            mem_done = mem_done_all[i]
            mem_s1 = mem_s1_all[i,:]

            # recover q-values for experience sampled
            qval_s0 = self.predict(mem_s0)
            qval_s1 = self.predict(mem_s1, target=self.target)
            # print(qval_s1)
            maxQ = np.max(qval_s1)
            # print(maxQ)

            # get target q-values (y)
            y = qval_s0

            # if final state
            if mem_done:
                update = reward
            else:
                update = reward + (self.gamma * maxQ)

            # update target q-values y with new update
            # print(update)
            # print(mem_rew)
            y[0, mem_a] = update
            # print(y)

            # save training data
            X_train[i,:] = mem_s0.flatten()
            # print(i)
            # print(mem_s0.shape)
            # print(X_train.shape)
            y_train[i,:] = y

        # after worked on sampled experiences, update network
        self.model.fit(X_train.reshape((i+1, self.height, self.width, 1)),
                       y_train,
                       batch_size=self.batch_size,
                       epochs=1,
                       verbose=0)

        # # update last state to last experienced
        # self.state = mem_s1

    def fill_buffer(self):
        """
        Pre-fill buffer using random experiences.
        """
        pass


# ===========================
#   Random Agent
# ===========================
class RandomAgent(object):
    """
    Take random actions!
    """

    def __init__(self, env, n_episodes):
        self.name = 'random'
        # get env characteristics
        self.action_space = env.action_space

        # flag to use replay buffer
        self.has_replay = False
        self.history = np.zeros((n_episodes, 2))  # step and rew

    def act(self, observation):
        return self.action_space.sample()


# ===========================
#   Replay Buffer
# ===========================
class ReplayBuffer(object):
    """
    Replay Buffer class modified from Patrick Emani. Original at:
    http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html
    """

    def __init__(self, buffer_size, random_seed=123, credit_size=5):
        self.buffer_size = buffer_size
        self.count = 0
        self.credit_size = credit_size
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, s, a, r, done, s2):
        experience = (s, a, r, done, s2)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        """
        batch_size specifies the number of experiences to add
        to the batch. If the replay buffer has less than batch_size
        elements, simply return all of the elements within the buffer.
        Generally, you'll want to wait until the buffer has at least
        batch_size elements before beginning to sample from it.
        """
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        done_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, done_batch, s2_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0

    def tail_replay(self,n):
        '''
        Returns tail of replay with n samples.
        '''
        size_replay = len(self.buffer)
        tail = list[itertools.islice(self.buffer,size_replay-n,size_replay)]
        print(tail.__class__)
        return tail

    def credit_assignment(self,max_rew):
        '''
        Distribute max_rew in the last experiences added to the queue.
        '''
        # get last experiences
        last_exp = self.tail_replay(self.credit_size)

        # add fraction of max_rew in each experience
        print('***')
        for i in len(last_exp):
            # add to current rew
            print(i)
            print('current rew = ',last_exp[-i-1][2])
            last_exp[-i-1][2] += max_rew/(i+1)
            print('new rew = ',last_exp[-i-1][2])

            # push exp back to replay
            self.buffer.append(last_exp[-i-1])


# ===========================
#   Deep-Q Networks
# ===========================
class DQN(object):
    """
    Deep Q-Network implementation based on the famous Atari work.
    """

    def __init__(self,
                 env,
                 n_steps=50,
                 n_episodes=1,
                 gamma=.9,
                 eps_min=.1,
                 eps_max=1,
                 lbd=.001,
                 batch_size=64,
                 buffer_size=100000,
                 target_update_freq=100,
                 hidden_layers=2,
                 neurons=[6,32,32,9],
                 dropout=0.,
                 pre_fill_buffer=False,
                 target=False,
                 eval_factor=100):

        # agent info
        self.name = 'DQN'
        self.history = np.zeros((n_episodes, 3))  # step and rew and epsilon
        self.rand_counter = 0
        self.act_counter = 0

        # check if discrete or multi discrete
        if env.action_space.__class__ == gym.spaces.discrete.Discrete:
            a_type = 'discrete'
        elif env.action_space.__class__ == gym.spaces.multi_discrete.MultiDiscrete:
            a_type = 'multi_discrete'

        # environment parameters
        self.env = env
        self.n_ob = env.observation_space.shape[0]
        if a_type == 'discrete':
            self.n_ac = env.action_space.n
        elif a_type == 'multi_discrete':
            self.n_ac = env.action_space.num_discrete_space

        # to plot evaluation on standard env
        self.eval_factor = eval_factor
        self.eval = np.zeros((int(n_episodes/self.eval_factor), 2))  # step and rew and epsilon

        # parameters
        self.t = 0
        self.n_steps = n_steps
        self.n_episodes = n_episodes
        self.gamma = gamma
        self.epsilon = eps_max
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.lbd = lbd
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.target_update_freq = target_update_freq
        self.target = target

        # neural models (original and target)
        self.hidden_layers = hidden_layers
        self.neurons = neurons
        self.dropout = dropout
        self.neural = DeepNeuralNet(hidden_layers=hidden_layers,
                                    neurons=neurons,
                                    dropout=dropout)
        self.model = self.neural.model
        if target:
            self.target_model = self.neural.model

        # start replay buffer
        self.replay = ReplayBuffer(buffer_size)
        self.has_replay = True
        if pre_fill_buffer:
            self.fill_buffer()

    def act(self, state):
        # apply e-greedy policy (chance of picking a random action)
        if (random.random() < self.epsilon):  #choose random action
            action = self.env.action_space.sample()
            self.rand_counter += 1
        else:
            # run Q function on states (obs) to get Q values
            # for all possible actions
            qval = self.model.predict(
                state.reshape(1, self.n_ob),
                batch_size=1)


            # choose the best action from Q(s,a) values
            action = (np.argmax(qval))
            self.act_counter += 1

        return action

    def update_epsilon(self):
        # update epsilon according to episode count
        self.epsilon = self.eps_min + (self.eps_max - self.eps_min) * np.exp(
            -self.lbd * self.t)

    def predict(self, s, target=False):
        if target:
            return self.target_model.predict(
                s.reshape(1, self.n_ob), batch_size=1)
        else:
            return self.model.predict(s.reshape(1, self.n_ob), batch_size=1)

    def use_replay(self, state, action, reward, done, next_state):
        # add last experience to buffer
        self.replay.add(state, action, reward, done, next_state)

        # check if need to update target network when using it
        if self.target and (self.t % self.target_update_freq == 0):
            self.target_model = self.neural.copy_weights(self.model,
                                                         self.target_model)

        self.t += 1
        self.update_epsilon()

        # sample buffer
        mem_s0_all, mem_a_all, mem_rew_all, mem_done_all, mem_s1_all = self.replay.sample_batch(self.batch_size)

        # get train values for each experience sampled
        n_samples = mem_a_all.shape[0]
        X_train = np.zeros((n_samples, self.n_ob))
        y_train = np.zeros((n_samples, self.n_ac))
        for i in range(n_samples):
            # get data for each minimatch
            mem_s0 = mem_s0_all[i,:]
            mem_a = mem_a_all[i]
            mem_rew = mem_rew_all[i]
            mem_done = mem_done_all[i]
            mem_s1 = mem_s1_all[i,:]

            # recover q-values for experience sampled
            qval_s0 = self.predict(mem_s0)
            qval_s1 = self.predict(mem_s1, target=self.target)
            # print(qval_s1)
            maxQ = np.max(qval_s1)
            # print(maxQ)

            # get target q-values (y)
            y = qval_s0

            # if final state
            if mem_done:
                update = reward
            else:
                update = reward + (self.gamma * maxQ)

            # update target q-values y with new update
            #print(update)
            y[0, mem_a] = update

            # save training data
            X_train[i,:] = mem_s0
            y_train[i,:] = y

        # after worked on sampled experiences, update network
        self.model.fit(X_train,
                       y_train,
                       batch_size=self.batch_size,
                       nb_epoch=1,
                       verbose=0)

        # # update last state to last experienced
        # self.state = mem_s1

    def fill_buffer(self):
        """
            Pre-fill buffer using random experiences:
            """
        # pre fill buffer with random experiences
        print('Filling experience buffer with random experiences...')

        while self.replay.size() != self.buffer_size:
            state = self.env.reset()

            for t in range(self.n_steps):
                # sample a random action
                action = self.env.action_space.sample()

                # execute this sampled action
                next_state, reward, done, info = self.env.step(action)

                # add to replay
                self.replay.add(state, action, reward, done, next_state)
                state = next_state

                # check if done
                if done:
                    break

# ===========================
#   Actor Network
# ===========================
class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.

    The output layer activation is a tanh to keep the action
    between -2 and 2
    """
    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau

        # Actor Network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[len(self.network_params):]

        # Op for periodically updating target network with online network weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) + \
                tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        self.actor_gradients = tf.gradients(self.scaled_out, self.network_params, -self.action_gradient)

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        net = tflearn.fully_connected(inputs, 400, activation='relu')
        net = tflearn.fully_connected(net, 300, activation='relu')
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, self.a_dim, activation='tanh', weights_init=w_init)
        scaled_out = tf.multiply(out, self.action_bound) # Scale output to -action_bound to action_bound
        # print('a_dim: ', self.a_dim)
        # print('out: ', out)
        # print('scaled_out: ', scaled_out)
        return inputs, out, scaled_out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


# ===========================
#   Critic Network
# ===========================
class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.

    """
    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, num_actor_vars, batch_size):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size

        # Create the critic network
        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action (i.e., sum of dy/dx over all ys). We then divide
        # through by the minibatch size to scale the gradients down correctly.
        self.action_grads = tf.div(tf.gradients(self.out, self.action), tf.constant(self.batch_size, dtype=tf.float32))

    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])
        net = tflearn.fully_connected(inputs, 400, activation='relu')

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net, 300)
        t2 = tflearn.fully_connected(action, 300)

        net = tflearn.activation(tf.matmul(net,t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, 1, weights_init=w_init)
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

# ===========================
#   Deep Deterministic Policy Gradient (DDPG)
# ===========================
class DDPG(object):
    """
    Deep Deterministic Policy Gradient (DDPG) implementation.
    Allow continuous actions using DNN.

    Reference:
    https://arxiv.org/pdf/1509.02971v2.pdf
    https://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html
    """

    def __init__(self,
                 sess,
                 env,
                 n_steps=50,
                 n_episodes=1,
                 actor_lrate=.0001, # learning rate for the actor network
                 critic_lrate=.001, # learning rate for the critic network
                 gamma=.99,         # discount factor
                 tau=.001,          # soft target update param
                 batch_size=64,
                 buffer_size=100000,
                 target_update_freq=100,
                 hidden_layers=2,
                 neurons=[6,32,32,9],
                 dropout=0.,
                 pre_fill_buffer=False,
                 target=False,
                 eval_factor=100):

        # init parameters
        self.name = 'DDPG'
        self.history = np.zeros((n_episodes, 2))  # step and rew
        self.sess = sess                          # tensorflow session
        self.n_steps=n_steps
        self.n_episodes=n_episodes
        self.actor_lrate=actor_lrate              # learning rate for the actor network
        self.critic_lrate=critic_lrate            # learning rate for the critic network
        self.gamma=gamma                          # discount factor
        self.tau=tau                              # soft target update param
        self.batch_size=batch_size
        self.buffer_size=buffer_size
        self.target_update_freq=target_update_freq
        self.eval_factor=eval_factor
        self.curr_t = 0
        self.curr_ep = 0

        # environment parameters
        self.env = env
        self.n_ob = env.observation_space.shape[0]
        self.n_ac = env.action_space.shape[0]
        self.action_bound = np.ones(self.n_ac)*np.abs(env.action_space.low)
        # self.action_bound = np.zeros((self.n_ac,2)) # low and high bounds
        # for i in range(self.n_ac):
        #     self.action_bound[i,:] = [env.action_space.low, env.action_space.high]

        # create actor and critic network and initialize weights
        sess.run(tf.global_variables_initializer())
        self.actor = ActorNetwork(sess, self.n_ob, self.n_ac, self.action_bound, actor_lrate, tau)
        self.critic = CriticNetwork(sess, self.n_ob, self.n_ac, critic_lrate, tau, self.actor.get_num_trainable_vars(), batch_size)


        # start replay buffer
        self.replay = ReplayBuffer(buffer_size)
        self.has_replay = True
        if pre_fill_buffer:
            self.fill_buffer()

    def act(self, observation):
        """
        Select an action based on current observation.
        """
        # predict action
        observation = np.reshape(observation, (1,self.n_ob)) + self.noise()
        action = self.actor.predict(observation)

        return action

    def use_replay(self, past_observation, action, reward, done, observation):
        """
        Use experience replay to break dependency of state/actions and drive
        value function to optimal.
        """
        # add to replay buffer
        past_observation = np.reshape(past_observation, (self.n_ob,))
        action = np.reshape(action, (self.n_ac,))
        observation = np.reshape(observation, (self.n_ob,))
        self.replay.add(past_observation, action, reward, done, observation)

        # keep adding experience to the memory until
        # there are at least minibatch size samples
        if self.replay.size() > self.batch_size:
            s_batch, a_batch, r_batch, t_batch, s2_batch = \
                self.replay.sample_batch(self.batch_size)

            # calculate targets
            target_q = self.critic.predict_target(s2_batch, self.actor.predict_target(s2_batch))

            y_i = []
            for k in range(self.batch_size):
                if t_batch[k]:
                    y_i.append(r_batch[k])
                else:
                    y_i.append(r_batch[k] + self.gamma * target_q[k])

            # update the critic given the targets
            predicted_q_value, _ = self.critic.train(s_batch, a_batch, np.reshape(y_i, (self.batch_size, 1)))

            # update the actor policy using the sampled gradient
            a_outs = self.actor.predict(s_batch)
            grads = self.critic.action_gradients(s_batch, a_outs)
            self.actor.train(s_batch, grads[0])

            # update target networks
            self.actor.update_target_network()
            self.critic.update_target_network()

    def update_nets(self):
        """
        Initialize network weights before starting.
        """
        self.actor.update_target_network()
        self.critic.update_target_network()

    def noise(self):
        """
        Compute noise based on current episode and time step.
        """
        noise = 1. / (1. + self.curr_t + self.curr_ep)
        return noise

# ================================
#    EXPLORATION NOISE
# ================================
class ExplorationNoise(object):

    # ================================
    #    WHITE NOISE PROCESS
    # ================================
    @staticmethod
    def white_noise(mu, sigma, num_steps):
        # Generate random noise with mean 0 and variance 1
        return np.random.normal(mu, sigma, num_steps)

    # ================================
    #    ORNSTEIN-UHLENBECK PROCESS
    # ================================
    @staticmethod
    def ou_noise(theta, mu, sigma, num_steps, dt=1.):
        noise = np.zeros(num_steps)

        # Generate random noise with mean 0 and variance 1
        white_noise = np.random.normal(0, 1, num_steps)

        # Solve using Euler-Maruyama method
        for i in xrange(1, num_steps):
            noise[i] = noise[i - 1] + theta * (mu - noise[i - 1]) * \
                                                dt + sigma * np.sqrt(dt) * white_noise[i]

        return noise

    # ================================
    #    EXPONENTIAL NOISE DECAY
    # ================================
    @staticmethod
    def exp_decay(noise, decay_end):
        num_steps = noise.shape[0]
        # Check if decay ends before end of noise sequence
        assert(decay_end <= num_steps)

        scaling = np.zeros(num_steps)

        scaling[:decay_end] = 2. - np.exp(np.divide(np.linspace(1., decay_end, num=decay_end) * np.log(2.), decay_end))

        return np.multiply(noise, scaling)

    # ================================
    #    TANH NOISE DECAY
    # ================================
    @staticmethod
    def tanh_decay(noise, decay_start, decay_length):
        num_steps = noise.shape[0]
        # Check if decay ends before end of noise sequence
        assert(decay_start + decay_length <= num_steps)

        scaling = 0.5*(1. - np.tanh(4. / decay_length * np.subtract(np.linspace(1., num_steps, num_steps),
                                                              decay_start + decay_length/2.)))

        return np.multiply(noise, scaling)
