from PythonClient import *
import sys
import time
import msgpackrpc
import math
import select, termios, tty

client = AirSimClient('127.0.0.1')

msg = """
Control Your Turtlebot!
---------------------------
Moving around:
   u    i    o
   j    k    l
   m    ,    .

q/z : increase/decrease max speeds by 10%
w/x : increase/decrease only linear speed by 10%
e/c : increase/decrease only angular speed by 10%
space key, k : force stop
anything else : stop smoothly

CTRL-C to quit
"""

moveBindings = {
        'i':(1,0),
        'o':(1,-1),
        'j':(0,1),
        'l':(0,-1),
        'u':(1,1),
        ',':(-1,0),
        '.':(-1,1),
        'm':(-1,-1),
}

speedBindings={
        'q':(1.1,1.1),
        'z':(.9,.9),
        'w':(1.1,1),
        'x':(.9,1),
        'e':(1,1.1),
        'c':(1,.9),
}

def getKey():
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)  # (3) for arrows
        # if key == '\033[A' or key == '\033[B' or key == '\033[C' or key == '\033[D':
        #     print 'getKey arrow pressed'
    else:
        key = ''

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

speed = .2
turn = 1

def vels(speed,turn):
    return "currently:\tspeed %s\tturn %s " % (speed,turn)

if __name__=="__main__":
    settings = termios.tcgetattr(sys.stdin)
    height = 0.5
    key_dur = 0.1
    
    # rospy.init_node('turtlebot_teleop')
    # pub = rospy.Publisher('~cmd_vel', Twist, queue_size=5)

    x = 0
    th = 0
    status = 0
    count = 0
    acc = 0.1
    target_speed = 0
    target_turn = 0
    control_speed = 0
    control_turn = 0
    try:
        print msg
        print vels(speed,turn)
        while(1):
            key = getKey()
            if key in moveBindings.keys():
                x = moveBindings[key][0]
                th = moveBindings[key][1]
                count = 0
            elif key in speedBindings.keys():
                speed = speed * speedBindings[key][0]
                turn = turn * speedBindings[key][1]
                count = 0

                print vels(speed,turn)
                if (status == 14):
                    print msg
                status = (status + 1) % 15
            elif key == ' ' or key == 'k' :
                x = 0
                th = 0
                control_speed = 0
                control_turn = 0
            else:
                count = count + 1
                if count > 4:
                    x = 0
                    th = 0
                if (key == '\x03'):
                    break

            target_speed = speed * x
            target_turn = turn * th

            if target_speed > control_speed:
                control_speed = min( target_speed, control_speed + 0.02 )
            elif target_speed < control_speed:
                control_speed = max( target_speed, control_speed - 0.02 )
            else:
                control_speed = target_speed

            if target_turn > control_turn:
                control_turn = min( target_turn, control_turn + 0.1 )
            elif target_turn < control_turn:
                control_turn = max( target_turn, control_turn - 0.1 )
            else:
                control_turn = target_turn


            client.moveByVelocityZ(control_speed, 0, -height, key_dur, 
                DrivetrainType.MaxDegreeOfFreedom, YawMode(True, -control_turn))
            client.rotateByYawRate(-control_turn, key_dur)

    except:
        print e

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
            

    # settings = termios.tcgetattr(sys.stdin) 
    # key_dur = 0.1
    # vel = 1
    # vel_min = 0.1
    # vel_max = 2
    # count = 0
    # while(1):
    #     ori = client.getOrientation()
    #     pos = client.getPosition()
    #     pos[2] = -1
    #     # if count % 10 is 0:
    #     #     print str(pos[0]) + "\t" + str(pos[1]) + "\t" + str(pos[2])
    #     # count = count + 1

    #     key = getKey()
    #     if key == '8':
    #         # print "up key pressed"
    #         client.moveByVelocityZ(0, vel, pos[2], key_dur, 
    #             DrivetrainType.MaxDegreeOfFreedom, YawMode(False, ori[0]))
    #     elif key == '2':
    #         # print "down key pressed"
    #         client.moveByVelocityZ(0, -vel, pos[2], key_dur, 
    #             DrivetrainType.MaxDegreeOfFreedom, YawMode(False, ori[0]))
    #     elif key == '4':
    #         # print "left key pressed"
    #         client.moveByVelocityZ(vel, 0, pos[2], key_dur, 
    #             DrivetrainType.MaxDegreeOfFreedom, YawMode(False, ori[0]))
    #     elif key == '6':
    #         # print "right key pressed"
    #         client.moveByVelocityZ(-vel, 0, pos[2], key_dur, 
    #             DrivetrainType.MaxDegreeOfFreedom, YawMode(False, ori[0]))
    #     elif key == '7':
    #         client.moveByVelocityZ(vel, vel, pos[2], key_dur, 
    #             DrivetrainType.MaxDegreeOfFreedom, YawMode(False, ori[0]))
    #     elif key == '9':
    #         client.moveByVelocityZ(-vel, vel, pos[2], key_dur, 
    #             DrivetrainType.MaxDegreeOfFreedom, YawMode(False, ori[0]))
    #     elif key == '3':
    #         client.moveByVelocityZ(-vel, -vel, pos[2], key_dur, 
    #             DrivetrainType.MaxDegreeOfFreedom, YawMode(False, ori[0]))
    #     elif key == '1':
    #         client.moveByVelocityZ(vel, -vel, pos[2], key_dur, 
    #             DrivetrainType.MaxDegreeOfFreedom, YawMode(False, ori[0]))
    #     elif key == '+':
    #         if vel < vel_max:
    #             vel = vel + 0.1;
    #             print "velocity = " + str(vel)
    #         else:
    #             print "velocity maxed!"
    #     elif key == '-':
    #         if vel > (vel_min + 0.05):
    #             vel = vel - 0.1;
    #             print "velocity = " + str(vel)
    #         else:
    #             print "live a litte, grandma!"
    #     elif key == '\x03' or key == 'q':  # ^C or q
    #         break

