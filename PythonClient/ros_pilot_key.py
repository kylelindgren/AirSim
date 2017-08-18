from PythonClient import *
import sys
import time
import msgpackrpc
import math

import rospy
from geometry_msgs.msg import Twist

client = AirSimClient('127.0.0.1')

height = 1
key_dur = 0.1

# stores static variable yaw
class Yaw:
    val = 0  # radians

def callback(data):
    Yaw.val = Yaw.val + (math.radians(-data.angular.z) * key_dur)
    client.moveByVelocityZ(data.linear.x * math.cos(Yaw.val), 
        data.linear.x * math.sin(Yaw.val), -height, key_dur, 
        DrivetrainType.MaxDegreeOfFreedom, YawMode(False, math.degrees(Yaw.val)))
    # time.sleep(key_dur)
    # client.rotateByYawRate(-data.angular.z, key_dur)

def listener():
    rospy.init_node('airsim_turtlebot_teleop')
    sub = rospy.Subscriber('/turtlebot_teleop_keyboard/cmd_vel', Twist, callback)

    rospy.spin()

if __name__=="__main__":
    print "turtlebot keyboard listener running"
    listener()
