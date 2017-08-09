from PythonClient import *
import cv2
import time
import sys
import rospy
from sensor_msgs.msg import Image
import numpy as np
from cv_bridge import CvBridge, CvBridgeError

def printUsage():
   print("Usage: python camera.py [depth|segmentation|scene]")

# ros setup
pub = rospy.Publisher('/airsim_depth_image', Image, queue_size=10)
rospy.init_node('depth_pub', anonymous=True)
# rate = rospy.Rate(60)
bridge = CvBridge()

cameraType = "depth"

for arg in sys.argv[1:]:
  cameraType = arg.lower()

cameraTypeMap = { 
 "depth": AirSimImageType.Depth,
 "segmentation": AirSimImageType.Segmentation,
 "seg": AirSimImageType.Segmentation,
 "scene": AirSimImageType.Scene,
}

if (not cameraType in cameraTypeMap):
  printUsage()
  sys.exit(0)

# print cameraTypeMap[cameraType]

client = AirSimClient('127.0.0.1')

help = False

frameCount = 0
startTime = time.clock()
fps = 0

# save video
# fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# out = cv2.VideoWriter('depth_output_10fps.avi', fourcc, 10.0, (1280, 720))

while True and not rospy.is_shutdown():
    # because this method returns std::vector<uint8>, msgpack encodes it as a string
    rawImage = client.simGetImage(0, cameraTypeMap[cameraType])
    if (rawImage is None):
        print("Camera is not returning image, please check airsim for error messages")
        sys.exit(0)
    else:
        png = cv2.imdecode(rawImage, cv2.IMREAD_UNCHANGED)
        msg = bridge.cv2_to_imgmsg(png, encoding="bgr8")
        cv2.imshow("Depth", png)
        # out.write(png)

    pub.publish(msg)
    frameCount = frameCount + 1
    endTime = time.clock()
    diff = endTime - startTime
    if (diff > 1):
        fps = frameCount
        frameCount = 0
        startTime = endTime
    
    key = cv2.waitKey(1) & 0xFF;
    if (key is 27 or key is ord('q') or key is ord('x')):
        break;
