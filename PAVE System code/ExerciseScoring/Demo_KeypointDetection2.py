# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 16:59:52 2022

@author: bxian
"""

#!pip install mediapipe
import cv2
import mediapipe as mp
import time
import os
import csv
import pandas as pd
import imutils

class PoseDetector:

    def __init__(self, mode = False, modelComplexity = 1, smooth=True,  enable_segmentation=False, smooth_segmentation=True, detectionCon = 0.65, trackCon = 0.65):

        self.mode = mode
        self.modelComplexity = modelComplexity
        self.smooth = smooth
        self. enable_segmentation =  enable_segmentation
        self. smooth_segmentation =  smooth_segmentation
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.modelComplexity, self.smooth, self.enable_segmentation, self.smooth_segmentation, self.detectionCon, self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        #print(results.pose_landmarks)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img

    def getPosition(self, img, draw=True):
        lmList= []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                #print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append(id)
                lmList.append(lm.x)
                lmList.append(lm.y)
                lmList.append(lm.z)
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmList

def main():
    cap = cv2.VideoCapture(vidname)
    pTime = 0
    detector = PoseDetector()
    while cap.isOpened():
        success, img = cap.read()
        img = detector.findPose(img)
        if ImList:
            lmList = detector.getPosition(img)
            #print(lmList)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        cv2.imshow("Image", img)
        # Wait until a key is pressed.
        # Retreive the ASCII code of the key pressed
        k = cv2.waitKey(1) 
        
        # Check if 'ESC' is pressed.
        if k == ord('b'):
            
            # Break the loop.
            break
     
    # Release the VideoCapture object.
    cap.release()
     
    # Close the windows.
    cv2.destroyAllWindows()

##############################################################################

vidname = '/Users/bxian/OneDrive - National University of Singapore/NUS ISS MTech/Term3_intelligent_sensing_systems/Practice Module/KiMoRe_rgb/KiMoRe_rgb/SelfMade/Notproper/NPr_ID2/Es2/person2_ex1_NPr_NP.mp4'
cap = cv2.VideoCapture(vidname)
pTime = 0
detector = PoseDetector()
joints = []


while cap.isOpened():
    success, img = cap.read()
    if success:
        img = detector.findPose(img)
        lmList = detector.getPosition(img)
        if lmList:
            joints.append(lmList)
            #print(lmList)
    
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
    
        cv2.putText(img, str(int(fps)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        resized_img = imutils.resize(img, width=500)
        cv2.imshow("Image", resized_img)
        # Wait until a key is pressed.
        # Retreive the ASCII code of the key pressed
        k = cv2.waitKey(1) 
        
        # Check if 'ESC' is pressed.
        if k == ord('b'):
            
            # Break the loop.
            break
    
    else:
        break
 
# Release the VideoCapture object.
cap.release()
 
# Close the windows.
cv2.destroyAllWindows()

print('done')
fullstep = len(joints)
print('Total number of frames: ' + str(fullstep))
ystep = fullstep//100
numstep = (ystep) * 100

joints_100 = [] #joints_100 is rows that is multiple of 100
for i in range(numstep):
    timestep = i
    #print(timestep)
    joints_100.append(joints[timestep])

df = pd.DataFrame(joints_100)
print(df)