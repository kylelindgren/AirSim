import pygame
import time

def get_input():
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
