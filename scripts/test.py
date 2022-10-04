#!/usr/bin/env python3

from persondetector import detect
import cv2

weight = '/home/barath/ros1/workspace/src/people_detection/scripts/weights/yolov7.pt'
img = cv2.imread(r'/home/barath/ros1/workspace/src/people_detection/scripts/images/frame0000.jpg')
person = detect(weight, img)
print(person)
