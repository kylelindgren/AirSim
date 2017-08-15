#!/usr/bin/env python
""" validate_joystick.py
Read and print all buttons available for the connected joystick.
"""

__author__ = "Vinicius Guimaraes Goecks"
__version__ = "0.0.0"
__status__ = "Prototype"
__date__ = "August 03, 2017"

# import

import pygame
import time

def get_input():
    '''
    Reads and prints inputs.
    '''
    # read controller
    pygame.event.get()

    # print readings
    print('\nBUTTONS')
    for i in range(n_buttons):
        print (i, my_joystick.get_button(i))

    print('\nAXES')
    for i in range(n_axes):
        print (i, my_joystick.get_axis(i))

    # flush rest
    pygame.event.clear()


pygame.init()
print ("Joysticks: ", pygame.joystick.get_count())
my_joystick = pygame.joystick.Joystick(0)
my_joystick.init()
n_axes = my_joystick.get_numaxes()
n_buttons = my_joystick.get_numbuttons()
print('number of axes: ', n_axes)
print('number of buttons: ', n_buttons)

while 1:
    print('***** NEW READ *****')
    get_input()
    time.sleep(.1)
