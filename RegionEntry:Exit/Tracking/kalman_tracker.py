#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 09:37:40 2019

@author: venkatesh
"""

'''
Implement and test tracker
'''
import numpy as np
from numpy import dot
from scipy.linalg import inv, block_diag
from collections import deque
from config import config as cfg

class KF_Tracker(): # class for Kalman Filter-based tracker
    def __init__(self,obj_class):
        # Initialize parametes for tracker (history)
        self.id = 0  # tracker's id
        self.box = [] # list to store the coordinates for a bounding box
        self.hits = 0 # number of detection matches
        self.no_losses = 0 # number of unmatched tracks (track loss)
        self.buffer_loss = 0 # number of frames after unmatched tracks are detected
        self.sal_region = -1
        self.cornercount = 0
        

        # Initialize parameters for Kalman Filtering
        # The state is the (x, y) coordinates of the detection box
        # state: [up, up_dot, left, left_dot, down, down_dot, right, right_dot]
        # or[up, up_dot, left, left_dot, height, height_dot, width, width_dot]
        self.x_state=[]
        self.obj_class = obj_class
        self.Initial_State = []
        
        if cfg.class_labels[obj_class].lower() == 'person' or cfg.class_labels[obj_class].lower() == 'pedestrian':            
            self.dt = 1.        # time interval
            self.L = 2.0        # State covariance Initialisation parameter
            self.R_scaler = 1.0 # measurement covariance parameten
            self.R_ratio = 1.0/32
        else:
            self.dt = 1.        # time interval
            self.L = 0.5       # State covariance Initialisation parameter
            self.R_scaler = 1.5 # measurement covariance parameten
            self.R_ratio = 10
        
        self.choose_kalman = -1
        self.prob = 0
        self.Initial_Cord = []
        self.ref_featuer = []
        self.out_of_bound = 0
        self.confidence = 1
        
        self.hist_feat = []
        self.hist_collection = deque(maxlen = cfg.No_avg_histogram)
        
        # Process matrix, assuming constant velocity model
        self.F = np.array([[1, self.dt, 0,  0,  0,  0,  0, 0],
                           [0, 1,  0,  0,  0,  0,  0, 0],
                           [0, 0,  1,  self.dt, 0,  0,  0, 0],
                           [0, 0,  0,  1,  0,  0,  0, 0],
                           [0, 0,  0,  0,  1,  self.dt, 0, 0],
                           [0, 0,  0,  0,  0,  1,  0, 0],
                           [0, 0,  0,  0,  0,  0,  1, self.dt],
                           [0, 0,  0,  0,  0,  0,  0,  1]])

        # Measurement matrix, assuming we can only measure the coordinates

        self.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, 0]])


        # Initialize the state covariance
        self.P = np.diag(self.L*np.ones(8))


        # Initialize the process covariance
        self.Q_comp_mat = np.array([[self.dt**4/4., self.dt**3/2.],
                                    [self.dt**3/2., self.dt**2]])
        self.Q = block_diag(self.Q_comp_mat, self.Q_comp_mat,
                            self.Q_comp_mat, self.Q_comp_mat)

        # Initialize the measurement covariance
        self.R_diag_array = self.R_ratio * np.array([self.L, self.L, self.L, self.L])
        self.R = np.diag(self.R_diag_array)


    def update_R(self):
        R_diag_array = self.R_ratio * np.array([self.L, self.L, self.L, self.L])
        self.R = np.diag(R_diag_array)


    def kalman_filter(self, z):
        '''
        Implement the Kalman Filter, including the predict and the update stages,
        with the measurement z
        '''
        x = self.x_state
        # Predict
        x = dot(self.F, x)
        self.P = dot(self.F, self.P).dot(self.F.T) + self.Q

        #Update
        S = dot(self.H, self.P).dot(self.H.T) + self.R
        K = dot(self.P, self.H.T).dot(inv(S)) # Kalman gain
        y = z - dot(self.H, x) # residual
        x += dot(K, y)
        self.P = self.P - dot(K, self.H).dot(self.P)

        self.x_state = x.astype(int) # convert to integer coordinates
                                     #(pixel values)
        
    def predict_only(self, sal_img):
        '''
        Implment only the predict stage. This is used for unmatched detections and
        unmatched tracks
        '''
        x = self.x_state
        x_inti = self.x_state
        # Predict
        x = dot(self.F, x)
        self.P = dot(self.F, self.P).dot(self.F.T) + self.Q

        self.x_state = x.astype(int)
        
        if sal_img != '':
            _x = x.astype(int)
            xx = _x.T[0].tolist()
    #        xx =[xx[0], xx[2], xx[4], xx[6]]
            roi_img = sal_img[xx[2]:xx[6], xx[0]:xx[4]]
            data = roi_img[ np.where( roi_img > 0.1)]
            sal_roi_area = len(data)
            roi_img_area = (xx[6] - xx[2]) * (xx[4] - xx[0])
            if roi_img_area >= 1:
                SR_Ratio = sal_roi_area/roi_img_area
            else:
                SR_Ratio = 0
                
            if SR_Ratio > 0.7:
                self.sal_region = 1
                width  = x_inti[4] - x_inti[0]
                height = x_inti[6] - x_inti[2]
                x_inti[0] = xx[0] + (xx[1] + xx[5])/2
                x_inti[2] = xx[2] + (xx[3] + xx[7])/2
                x_inti[4] = x_inti[0] + width
                x_inti[6] = x_inti[2] + height
                self.x_state = x.astype(int)
            else:
                self.sal_region = 0
        
