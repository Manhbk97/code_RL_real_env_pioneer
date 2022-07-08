#! /usr/bin/env python
import rospy
import numpy	
import time
from sensor_msgs.msg import Image

def listener():
	rospy.init_node('image_proces')
	rospy.Subscriber("/camera/rgb/image_raw", Image, self._camera_rgb_image_raw_callback)
	rospy.spin()

def _camera_rgb_image_raw_callback(self, data):
	self.camera_rgb_image_raw = data

if __name__ == '__main__':
	listener()