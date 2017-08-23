from inputs import get_gamepad
import time


def get_input():
    # new read
    events = get_gamepad()
    print(len(events))
    event = events[0] # just using one input

    if (event.code == 'ABS_Z'):
        print('Left Button: ',event.state)

    if (event.code == 'ABS_RZ'):
        print('Right Stick: ',event.state)


while 1:
    print('got input')
    get_input()
    time.sleep(.5)
