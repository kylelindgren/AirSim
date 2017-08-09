import sys
import rospy
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError

def webcam_pub():
    pub = rospy.Publisher('/camera_reading', Image, queue_size=1)
    rospy.init_node('webcam_pub', anonymous=True)
    rate = rospy.Rate(60) # 60hz

    cam = cv2.VideoCapture("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080,format=(string)I420, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")
    #print cam.isOpened()
    bridge = CvBridge()

    if not cam.isOpened():
         sys.stdout.write("Webcam is not available")
         return -1

    while not rospy.is_shutdown():
        ret, frame = cam.read()
        msg = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
#	cv2.imshow('test', frame)
#	cv2.waitKey(1)
        if not ret:
            rospy.loginfo("Capturing image failed.")

        pub.publish(msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        webcam_pub()
    except rospy.ROSInterruptException:
        pass

