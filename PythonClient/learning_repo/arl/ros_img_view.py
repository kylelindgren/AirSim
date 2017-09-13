import sys
import time
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


bridge = CvBridge()

def callback(img):
    # img_cv = bridge.imgmsg_to_cv2(img, "mono16")
    # img_cv = bridge.compressed_imgmsg_to_cv2(img, "16UC1")
    # img_data = np.uint8(img.data)
    # print np.float(img.data)
    # print img.encoding
    img_cv = bridge.imgmsg_to_cv2(img)
    img_cv_cvt = np.uint8(img_cv)
    # print img_cv_cvt

    # img_cv = bridge.imgmsg_to_cv2(img, "passthrough")
    # img_cv_cvt = np.uint8(img_cv)
    
    # img_cv_cvt = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)
    # img_cv_cvt = np.uint16(img_cv)
    # print img_cv_cvt
    cv2.imshow('depth', img_cv_cvt)
    cv2.waitKey(1)

def listener():
    rospy.init_node('ros_img_view')
    sub = rospy.Subscriber('/zed/depth/depth_registered', Image, callback)

    rospy.spin()

if __name__=="__main__":
    print "turtlebot image listener running"
    listener()
