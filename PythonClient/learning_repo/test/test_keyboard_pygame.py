#!/usr/bin/env python
""" test_keyboard_pygame.py:
Testing different ways of controlling the drone using PyGame.
"""

__author__ = "Vinicius Guimaraes Goecks"
__version__ = "0.0.0"
__status__ = "Prototype"
__date__ = "June 13, 2017"

# import
import pygame, time           # for keyboard multi-threaded inputs
from pygame.locals import *

# initialize pygame for inputs
pygame.display.init()
pygame.font.init()
screen = pygame.display.set_mode((500, 120))
pygame.display.set_caption('CLICK HERE TO CONTROL DRONE :)')
time.sleep(3)

last_lat = 0
last_lon = 0

while True:
    lat = last_lat
    lon = last_lon

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:

            # LATERAL COMMANDS
            if event.key == pygame.K_d:
                lat = 1

            if event.key == pygame.K_a:
                lat = -1

            if event.key == pygame.K_s:
                lat = 0

            # LONGITUDINAL COMMANDS
            if event.key == pygame.K_o:
                lon = 1 # speed forward

            if event.key == pygame.K_k:
                lon = 0 # break drone

            if event.key == pygame.K_m:
                lon = -1 # break drone

    # record keys
    last_lat = lat
    last_lon = lon

    print("LAT: %i | LON: %i" %(lat,lon))
