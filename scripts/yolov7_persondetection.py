#!/usr/bin/env python3

import pdb
import sys
import os
import torch
import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import cv2
import matplotlib.pyplot as plt
from metrics_refbox_msgs.msg import PersonDetectionResult
from metrics_refbox_msgs.msg import Command
import pandas as pd
import copy
import rospkg
import numpy as np
from persondetector import detect
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# importing Yolov5 model
import pdb

#IMAGE_TOPIC = "/hsrb/head_rgbd_sensor/rgb/image_raw"
#IMAGE_TOPIC = "/camera/rgb/image_raw"
IMAGE_TOPIC="/camera/color/image_raw"
class PersonDetector:
    def __init__(self):
        self.bounding_box = 0
        self.img = 0
        self.clip_size = 1
        self.image_queue = None
        self.bb_pub = rospy.Publisher(
            "/metrics_refbox_client/person_detection_result", PersonDetectionResult, queue_size=10)
        self.referee_command_sub = rospy.Subscriber(
            "/metrics_refbox_client/command", Command, self._referee_command_cb)

        rospy.loginfo("[Person Detection] Waiting for referee box to be ready...")
        self.stop_sub_flag = False
        self.cv_bridge = CvBridge()
        self.person_flag = False
        self.image_sub = None
        
        
        # yolo model config
        #self.model_name = 'best_overfit.pt'
        self.model_name_og = 'yolov7.pt'
        self.confidence_threshold = 0.5


    def detector(self):
        self.img = self.image_queue[0]
        rospy.loginfo("Running detector..")
        rospack = rospkg.RosPack()
        
        # get the file path for object_detection package
        pkg_path = rospack.get_path('person_detect')
        model_path = pkg_path + "/models/"
        
        
        result = PersonDetectionResult()
        result.message_type = result.RESULT
        predictions = {}

        weight = model_path + self.model_name_og 
        person = detect(weight, self.img) 


        rospy.loginfo("publishing detection..")
        if bool(person):
            # img = cv2.imread(os.path.join(self.sample_images_path, 'image.jpg'), cv2.IMREAD_COLOR)

            if person['conf'] > self.confidence_threshold:
                result.person_found = True

                predictions['boxes'] = (person['xmin'], person['ymin'],
                                        person['xmax'], person['ymax'])

                predictions['scores'] = person['conf']

                predictions['labels'] = person['name']
                result.image = self.cv_bridge.cv2_to_imgmsg(
                    self.img, encoding='passthrough')
                result.box2d.min_x = int(person['xmin'])
                result.box2d.max_x = int(person['xmax'])
                result.box2d.max_y = int(person['ymin'])
                result.box2d.min_y = int(person['ymax'])
                self.bb_pub.publish(result)
            else:
                result.person_found = False
                result.image = self.cv_bridge.cv2_to_imgmsg(
                    self.img, encoding='passthrough')
                self.bb_pub.publish(result)

        else:
            result.person_found = False
            result.image = self.cv_bridge.cv2_to_imgmsg(
                self.img, encoding='passthrough')
            self.bb_pub.publish(result)


        self.stop_sub_flag = False
        self.image_queue = []

        return predictions

        
    def _input_image_cb(self, msg):
        """
        :msg: sensor_msgs.Image
        :returns: None
        """
        rospy.loginfo("type of msg: {}".format(type(msg)))

        try:
            # if not self.stop_sub_flag:
            rospy.loginfo("Image received..")
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            if self.image_queue is None:
                self.image_queue = []
            self.image_queue.append(cv_image)
            print("Counter: ", len(self.image_queue))
            print("length of queue: ", len(self.image_queue))
            if len(self.image_queue) == self.clip_size:
    
                self.stop_sub_flag = True
                rospy.loginfo("Input images saved on local drive")
                self.image_sub.unregister()

                result = self.detector()
                self.stop_sub_flag = False
                print(result)
        # else:
            #     print("Clip size reached")

        except CvBridgeError as e:
            rospy.logerr(
                "Could not convert ros sensor msgs Image to opencv Image.")
            rospy.logerr(str(e))
            self._check_failure()
            return

    def _referee_command_cb(self, msg):

        # Referee comaand message (example)
        '''
        task: 1
        command: 1
        task_config: "{\"Target object\": \"Cup\"}"
        uid: "0888bd42-a3dc-4495-9247-69a804a64bee"
        '''

        # START command from referee
        if msg.task == 2 and msg.command == 1:

            print("\nStart command received")

            self.image_sub = rospy.Subscriber(IMAGE_TOPIC,
                                               Image,
                                               self._input_image_cb)

            print("\n")
            print("Initiating person detection - Heartmet")
            print("\n")

        # STOP command from referee
        if msg.command == 2:
            self.stop_sub_flag = True
            self.image_sub.unregister()

            rospy.loginfo("Referee Subscriber stopped")


if __name__ == '__main__':
    rospy.init_node('person_detect')
    PersonDetector()
    rospy.spin()
