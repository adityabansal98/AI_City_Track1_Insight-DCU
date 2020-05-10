#!/usr/bin/env python3 -W ignore::DeprecationWarning
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 22:15:17 2019

@author: venkatesh_gm
"""

import pickle
import cv2
import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import collections

#from utils import io
from time import sleep
from tqdm import tqdm
from collections import deque
from sklearn.utils.linear_assignment_ import linear_assignment
#from scipy.optimize import linear_sum_assignment as linear_assignment
from config import config as cfg
from kalman_tracker import KF_Tracker
debug = False

# Ignore the warning
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

'''
*********************************************************
Create a video file from the output frames
*********************************************************

avconv -r 30 -i %04d.png -vf "scale=1240:376,format=yuv420p" -codec:v libx264 Object_Tracking.mp4


'''

# Global variables to be used by funcitons of VideoFileClop
frame_count = 0 # frame counter
max_age = cfg.max_age  # no.of consecutive unmatched detection before
             # a track is deleted
min_hits = cfg.min_hits  # no. of consecutive matches needed to establish a track

tracker_list =[] # list for trackers
# list for track ID
track_id_list = deque([x for x in range(1,1001)])


Track_Info = []
Track_Label = []

'''
*******************
Read File content
*******************
'''

class Read_Files:

    def Read_pkl_file(self, filename):
        pkl_file = open(filename, 'rb')
        filedata = pickle.load(pkl_file)
        pkl_file.close()

        return filedata

    def Read_numpy_file(self, filename):
        numpy_data = np.load(filename, mmap_mode='r')

        return numpy_data
    
    def decode_detection(self,Data):
        
        Object_bbox  = Data[0]
        Object_Class = Data[1]
        Object_Score = Data[2]
        
        return Object_bbox, Object_Class, Object_Score

'''
*******************
Draw Utilities
*******************
'''
class Draw:

    def draw_roi_boxes(self, image,pboxes,color=(255,255,0),linewidth = 1):
        image_h, image_w, _ = image.shape

        if len(pboxes):
            for box in pboxes:
                cv2.rectangle(image, (box[0],box[1]), (box[2],box[3]), color, linewidth)

        return image

    def draw_roi_boxes_with_label(self, image,pboxes, label, color=(255,255,0),linewidth = 1):
        image_h, image_w, _ = image.shape

        if len(pboxes):
            index = 0
            for box in pboxes:
                left    = box[0]
                top     = box[1]
                right   = box[2]
                bottom  = box[3]
                cv2.rectangle(image, (left,top), (right,bottom), color, linewidth)
                labelSize, baseLine = cv2.getTextSize(str(label[index]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(image, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (255, 255, 255), cv2.FILLED)
                cv2.putText(image, label[index], (left, top ), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                index = index + 1

        return image
    
    def draw_box_label(self, img, bbox_cv2, box_color=(255, 255, 0.)):
        '''
        Helper funciton for drawing the bounding boxes and the labels
        bbox_cv2 = [left, top, right, bottom]
        '''
        left, top, right, bottom = bbox_cv2[0], bbox_cv2[1], bbox_cv2[2], bbox_cv2[3]
        # Draw the bounding box
        cv2.rectangle(img, (left, top), (right, bottom), box_color, 2)

        return img

    def draw_sal_info(self, org_img, saliency_img):
        heatmap = saliency_img
#        cv2.normalize(heatmap,  heatmap, 0, 255, cv2.NORM_MINMAX)
#        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#        heatmap_color = heatmap
        superimposed_img = cv2.addWeighted(org_img, 1, heatmap.astype(np.uint8), 0.75, 0)

        return superimposed_img

'''
*******************
Draw Utilities
*******************
'''
class Object_Tracking:
    
    def __init__(self):  
        
#        global frame_count
        global tracker_list
        global max_age
        global min_hits
        global track_id_list
        
        self.frame_count = 0 # frame counter
        max_age = cfg.max_age  # no.of consecutive unmatched detection before
                     # a track is deleted
        min_hits = cfg.min_hits  # no. of consecutive matches needed to establish a track
        
        self.tracker_list =[] # list for trackers
        # list for track ID       
        track_id_list = deque([x for x in range(1,1001)])
        
        return
        

    def returnHistogramComparison(self, hist_1, hist_2, method='intersection'):
        if(method=="intersection"):
            comparison = cv2.compareHist(hist_1, hist_2, cv2.HISTCMP_INTERSECT)
        elif(method=="correlation"):
            comparison = cv2.compareHist(hist_1, hist_2, cv2.HISTCMP_CORREL)
        elif(method=="chisqr"):
            comparison = cv2.compareHist(hist_1, hist_2, cv2.HISTCMP_CHISQR)
        elif(method=="bhattacharyya"):
            comparison = cv2.compareHist(hist_1, hist_2, cv2.HISTCMP_BHATTACHARYYA)
        else:
            raise ValueError('ERROR: NOT SUPPORTED')

        return comparison

    def Compute_HSI_Histogram(self,Image, Rect_Box):
        # set up the ROI for tracking
        h_bins = 50
        s_bins = 60
        histSize = [h_bins, s_bins]
        # hue varies from 0 to 179, saturation from 0 to 255
        h_ranges = [0, 180]
        s_ranges = [0, 256]
        ranges = h_ranges + s_ranges # concat lists
        # Use the 0-th and 1-st channels
        channels = [0, 1]

        (X0,Y0,X1,Y1) = Rect_Box
        roi = Image[int(Y0):int(Y1), int(X0):int(X1)]
        if roi.size > 0:
            hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    #        plt.imshow(roi)
    #        plt.show()
    #        mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
            roi_hist = cv2.calcHist([hsv_roi], channels, None, histSize, ranges, accumulate=False)
    #        cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
        else:
            roi_hist = np.zeros(histSize, dtype = np.float32)

        roi_hist = cv2.normalize(roi_hist, roi_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX).flatten()

        return roi_hist
    
    
    def Compute_SAL_Histogram(self,Image, Rect_Box):
        # set up the ROI for tracking
        (X0,Y0,X1,Y1) = Rect_Box
        roi = Image[int(Y0):int(Y1), int(X0):int(X1)]
        if roi.size > 0:
#            hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    #        plt.imshow(roi)
    #        plt.show()
    #        mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
            roi_hist = cv2.calcHist([roi],[0],None,[256],[0,256])
    #        cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
        else:
            roi_hist = np.zeros(256, dtype = np.float32)
        
        roi_hist = cv2.normalize(roi_hist, roi_hist).flatten()

        return roi_hist

    def KF_pipeline(self,z_box,obj_class,img, sal_img):
        '''
        Pipeline function for detection and tracking
        '''
        global debug

#        global Track_Info
#        global Track_Label

        Track_Info  = []
        Track_ID    = []
        Track_Class = []

        if self.frame_count==0:
            self.prev_img = img.copy()
            self.Previous_Cord = collections.defaultdict(list)

        self.org_img = img.copy()
        img_dims = np.shape(self.org_img)
        
        x_box =[]
        x_hist_feat = []
        if len(self.tracker_list) > 0:
            for trk in self.tracker_list:
                x_box.append(trk.box)
                x_hist_feat.append(trk.hist_feat)

        matched, unmatched_dets, unmatched_trks \
        = self.assign_detections_to_trackers_hist(self.prev_img, x_box, x_hist_feat, self.org_img, z_box, iou_thrd = 0.4)
#        matched, unmatched_dets, unmatched_trks \
#        = self.assign_detections_to_trackers(x_box, z_box, iou_thrd = 0.3)
        
        if cfg.USE_D2_CITY_DATASET:
            self.org_img = cv2.cvtColor(self.org_img, cv2.COLOR_BGR2RGB)
            
        if debug:
            print('Frame : ',self.frame_count)
            print('Detection: ', z_box)
            print('x_box: ', x_box)
            print('matched:', matched)
            print('unmatched_det:', unmatched_dets)
            print('unmatched_trks:', unmatched_trks)

            print('\n')

        # Deal with matched detections
        if matched.size > 0:
            for trk_idx, det_idx in matched:
                z = z_box[det_idx]
                z = np.expand_dims(z, axis=0).T
                
                tmp_trk= self.tracker_list[trk_idx]                
                
                tmp_trk.obj_class = obj_class[det_idx]                
                tmp_trk.kalman_filter(z)
#                tmp_trk.hist_feat = self.Compute_HSI_Histogram(img,z_box[det_idx])
                hist_feat = self.Compute_HSI_Histogram(img,z_box[det_idx])
                tmp_trk.hist_collection.appendleft(hist_feat)    
                hist_data = np.array(tmp_trk.hist_collection)                
                data = [a*b for a,b in zip(hist_data,cfg.avg_hist_weights)]
                tmp_trk.hist_feat = np.average(data, axis=0)
                
#                np.average(data, axis=0)
#                [a*b for a,b in zip(lista,listb)]
                
                xx = tmp_trk.x_state.T[0].tolist()
                xx =[xx[0], xx[2], xx[4], xx[6]]
                
                x_box[trk_idx] = xx
                tmp_trk.box = xx
                tmp_trk.hits += 1
                tmp_trk.no_losses = 0
                tmp_trk.buffer_loss = 0
                tmp_trk.confidence = tmp_trk.prob

        # Deal with unmatched detections
        if len(unmatched_dets)>0:
            for idx in unmatched_dets:
                z = z_box[idx]
                z = np.expand_dims(z, axis=0).T
                x = np.array([[z[0], 0, z[1], 0, z[2], 0, z[3], 0]]).T

                tmp_trk = KF_Tracker(obj_class[idx]) # Create a new tracker
                
                tmp_trk.obj_class = obj_class[idx]
                tmp_trk.x_state = x 
#                tmp_trk.hist_feat = self.Compute_HSI_Histogram(img,z_box[idx])
                hist_feat = self.Compute_HSI_Histogram(img,z_box[idx])
                tmp_trk.hist_collection.appendleft(hist_feat)    
                hist_data = np.array(tmp_trk.hist_collection)                
                data = [a*b for a,b in zip(hist_data,cfg.avg_hist_weights)]
                tmp_trk.hist_feat = np.average(data, axis=0)
                
                tmp_trk.predict_only(sal_img)
                xx = tmp_trk.x_state
                xx = xx.T[0].tolist()
                xx =[xx[0], xx[2], xx[4], xx[6]]

                tmp_trk.box = xx
                tmp_trk.id = track_id_list.popleft() # assign an ID for the tracker
                self.tracker_list.append(tmp_trk)
                x_box.append(xx)
                
                x_cv2 = tmp_trk.box
                Track_Info.append(x_cv2)
                Track_ID.append(tmp_trk.id)
                Track_Class.append(tmp_trk.obj_class)

        # Deal with unmatched tracks
        if len(unmatched_trks)>0:
            for trk_idx in unmatched_trks:
                tmp_trk = self.tracker_list[trk_idx]
#                tmp_trk.no_losses += 1
                tmp_trk.predict_only(sal_img)
                tmp_trk.no_losses += 1
                if tmp_trk.sal_region == 1 and ((tmp_trk.no_losses > max_age) and (tmp_trk.buffer_loss <= 2)):
                    tmp_trk.no_losses = 2
                    tmp_trk.buffer_loss += 1
                elif tmp_trk.sal_region == 0:
                    tmp_trk.no_losses += 1
                
                xx = tmp_trk.x_state
                xx = xx.T[0].tolist()

                xx =[xx[0], xx[2], xx[4], xx[6]]

                tmp_trk.box =xx
                x_box[trk_idx] = xx


        # The list of tracks to be annotated
        good_tracker_list =[]
        for trk in self.tracker_list:
            if ((trk.hits >= min_hits) and (trk.no_losses <= max_age)):
                 good_tracker_list.append(trk)
                 x_cv2 = trk.box
                 label = '%s' % trk.id
                 if debug:
                     print('updated box: {}\t{}\t{}\t{}'.format(self.frame_count,x_cv2,label,trk.prob))
                     print()
                 Track_Info.append(x_cv2)
                 Track_ID.append(trk.id)
                 Track_Class.append(trk.obj_class)
                 width = x_cv2[2] - x_cv2[0]
#                 if(((x_cv2[0] >= width/8) and (x_cv2[2] <= image_w - width/8)) and trk.cornercount < 2):
                
                 self.org_img = Draw().draw_box_label(self.org_img, x_cv2,(0, 0, 255))
#                 self.org_img = self.draw_box_label(self.org_img, trk.pBox,(255, 255, 255))
                 labelSize, baseLine = cv2.getTextSize(str(label), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
                 cv2.rectangle(self.org_img, (x_cv2[0], x_cv2[1] - labelSize[1]), (x_cv2[0] + labelSize[0], x_cv2[1] + baseLine), (255, 255, 255), cv2.FILLED)
                 cv2.putText(self.org_img, label, (x_cv2[0], x_cv2[1] ), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
#                 else:
#                    trk.cornercount += 1
                    

#        deleted_tracks = filter(lambda x: ((x.no_losses > max_age) and (x.no_losses < 0.75)), tracker_list)
#        deleted_tracks = filter(lambda x: x.no_losses > max_age and x.cornercount >= 2, tracker_list)
        deleted_tracks = filter(lambda x: x.no_losses > max_age, self.tracker_list)

        for trk in deleted_tracks:
            track_id_list.append(trk.id)

#        tracker_list = [x for x in tracker_list if (((x.no_losses<=max_age) or ((x.no_losses > max_age) and (trk.prob >= 0.75))) and (trk.no_losses > max_age))]
        self.tracker_list = [x for x in self.tracker_list if x.no_losses<=max_age]
#
        self.prev_img = img.copy()
        self.frame_count+=1

        return self.org_img,Track_Info,Track_ID,Track_Class
    
    
#    def PPHD_pipeline(self,z_box,obj_class,img, sal_img):
#        '''
#        Pipeline function for detection and tracking
#        '''
#        global debug
#
##        global Track_Info
##        global Track_Label
#
#        Track_Info  = []
#        Track_ID    = []
#        Track_Class = []
#
#        if self.frame_count==0:
#            self.prev_img = img.copy()
#            self.Previous_Cord = collections.defaultdict(list)
#
#        self.org_img = img.copy()
#        img_dims = np.shape(self.org_img)
#        
#        x_box =[]
#        x_hist_feat = []
#        if len(self.tracker_list) > 0:
#            for trk in self.tracker_list:
#                x_box.append(trk.box)
#                x_hist_feat.append(trk.hist_feat)
#
#        matched, unmatched_dets, unmatched_trks \
#        = self.assign_detections_to_trackers_hist(self.prev_img, x_box, x_hist_feat, self.org_img, z_box, iou_thrd = 0.4)
##        matched, unmatched_dets, unmatched_trks \
##        = self.assign_detections_to_trackers(x_box, z_box, iou_thrd = 0.3)
#
#        
#        if debug:
#            print('Frame : ',self.frame_count)
#            print('Detection: ', z_box)
#            print('x_box: ', x_box)
#            print('matched:', matched)
#            print('unmatched_det:', unmatched_dets)
#            print('unmatched_trks:', unmatched_trks)
#
#            print('\n')
#
#        # Deal with matched detections
#        if matched.size > 0:
#            for trk_idx, det_idx in matched:
#                z = z_box[det_idx]
#                z = np.expand_dims(z, axis=0).T
#                
#                tmp_trk= self.tracker_list[trk_idx]                
#                
#                tmp_trk.obj_class = obj_class[det_idx]                
#                tmp_trk.kalman_filter(z)
##                tmp_trk.hist_feat = self.Compute_HSI_Histogram(img,z_box[det_idx])
#                hist_feat = self.Compute_HSI_Histogram(img,z_box[det_idx])
#                tmp_trk.hist_collection.appendleft(hist_feat)    
#                hist_data = np.array(tmp_trk.hist_collection)                
#                data = [a*b for a,b in zip(hist_data,cfg.avg_hist_weights)]
#                tmp_trk.hist_feat = np.average(data, axis=0)
#                
##                np.average(data, axis=0)
##                [a*b for a,b in zip(lista,listb)]
#                
#                xx = tmp_trk.x_state.T[0].tolist()
#                xx =[xx[0], xx[2], xx[4], xx[6]]
#                
#                x_box[trk_idx] = xx
#                tmp_trk.box = xx
#                tmp_trk.hits += 1
#                tmp_trk.no_losses = 0
#                tmp_trk.buffer_loss = 0
#                tmp_trk.confidence = tmp_trk.prob
#
#        # Deal with unmatched detections
#        if len(unmatched_dets)>0:
#            for idx in unmatched_dets:
#                z = z_box[idx]
#                z = np.expand_dims(z, axis=0).T
#                x = np.array([[z[0], 0, z[1], 0, z[2], 0, z[3], 0]]).T
#
#                tmp_trk = KF_Tracker() # Create a new tracker
#                
#                tmp_trk.obj_class = obj_class[idx]
#                tmp_trk.x_state = x 
##                tmp_trk.hist_feat = self.Compute_HSI_Histogram(img,z_box[idx])
#                hist_feat = self.Compute_HSI_Histogram(img,z_box[idx])
#                tmp_trk.hist_collection.appendleft(hist_feat)    
#                hist_data = np.array(tmp_trk.hist_collection)                
#                data = [a*b for a,b in zip(hist_data,cfg.avg_hist_weights)]
#                tmp_trk.hist_feat = np.average(data, axis=0)
#                
#                tmp_trk.predict_only(sal_img)
#                xx = tmp_trk.x_state
#                xx = xx.T[0].tolist()
#                xx =[xx[0], xx[2], xx[4], xx[6]]
#
#                tmp_trk.box = xx
#                tmp_trk.id = track_id_list.popleft() # assign an ID for the tracker
#                self.tracker_list.append(tmp_trk)
#                x_box.append(xx)
#
#        # Deal with unmatched tracks
#        if len(unmatched_trks)>0:
#            for trk_idx in unmatched_trks:
#                tmp_trk = self.tracker_list[trk_idx]
##                tmp_trk.no_losses += 1
#                tmp_trk.predict_only(sal_img)
#                tmp_trk.no_losses += 1
#                if tmp_trk.sal_region == 1 and ((tmp_trk.no_losses > max_age) and (tmp_trk.buffer_loss <= 2)):
#                    tmp_trk.no_losses = 2
#                    tmp_trk.buffer_loss += 1
#                elif tmp_trk.sal_region == 0:
#                    tmp_trk.no_losses += 1
#                
#                xx = tmp_trk.x_state
#                xx = xx.T[0].tolist()
#
#                xx =[xx[0], xx[2], xx[4], xx[6]]
#
#                tmp_trk.box =xx
#                x_box[trk_idx] = xx
#
#
#        # The list of tracks to be annotated
#        good_tracker_list =[]
#        for trk in self.tracker_list:
#            if ((trk.hits >= min_hits) and (trk.no_losses <= max_age)):
#                 good_tracker_list.append(trk)
#                 x_cv2 = trk.box
#                 label = '%s' % trk.id
#                 if debug:
#                     print('updated box: {}\t{}\t{}\t{}'.format(self.frame_count,x_cv2,label,trk.prob))
#                     print()
#                 Track_Info.append(x_cv2)
#                 Track_ID.append(trk.id)
#                 Track_Class.append(tmp_trk.obj_class)
#                 width = x_cv2[2] - x_cv2[0]
##                 if(((x_cv2[0] >= width/8) and (x_cv2[2] <= image_w - width/8)) and trk.cornercount < 2):
#                 self.org_img = Draw().draw_box_label(self.org_img, x_cv2,(0, 0, 255))
##                 self.org_img = self.draw_box_label(self.org_img, trk.pBox,(255, 255, 255))
#                 labelSize, baseLine = cv2.getTextSize(str(label), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
#                 cv2.rectangle(self.org_img, (x_cv2[0], x_cv2[1] - labelSize[1]), (x_cv2[0] + labelSize[0], x_cv2[1] + baseLine), (255, 255, 255), cv2.FILLED)
#                 cv2.putText(self.org_img, label, (x_cv2[0], x_cv2[1] ), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
##                 else:
##                    trk.cornercount += 1
#                    
#
##        deleted_tracks = filter(lambda x: ((x.no_losses > max_age) and (x.no_losses < 0.75)), tracker_list)
##        deleted_tracks = filter(lambda x: x.no_losses > max_age and x.cornercount >= 2, tracker_list)
#        deleted_tracks = filter(lambda x: x.no_losses > max_age, self.tracker_list)
#
#        for trk in deleted_tracks:
#            track_id_list.append(trk.id)
#
##        tracker_list = [x for x in tracker_list if (((x.no_losses<=max_age) or ((x.no_losses > max_age) and (trk.prob >= 0.75))) and (trk.no_losses > max_age))]
#        self.tracker_list = [x for x in self.tracker_list if x.no_losses<=max_age]
##
#        self.prev_img = img.copy()
#        self.frame_count+=1
#
#        return self.org_img,Track_Info,Track_ID,Track_Class
#    
    
    def box_iou2(self,a, b):
        '''
        Helper funciton to calculate the ratio between intersection and the union of
        two boxes a and b
        a[0], a[1], a[2], a[3] <-> left, up, right, bottom
        '''

        w_intsec = np.maximum (0, (np.minimum(a[2], b[2]) - np.maximum(a[0], b[0])))
        h_intsec = np.maximum (0, (np.minimum(a[3], b[3]) - np.maximum(a[1], b[1])))
        s_intsec = w_intsec * h_intsec
        s_a = (a[2] - a[0])*(a[3] - a[1])
        s_b = (b[2] - b[0])*(b[3] - b[1])

        if (s_a + s_b -s_intsec) >= 1:
            return float(s_intsec)/(s_a + s_b -s_intsec)
        else:
            return 0


    def assign_detections_to_trackers(self,trackers, detections, iou_thrd = 0.3):
        '''
        From current list of trackers and new detections, output matched detections,
        unmatchted trackers, unmatched detections.
        '''

        IOU_mat= np.zeros((len(trackers),len(detections)),dtype=np.float32)
        for t,trk in enumerate(trackers):
            #trk = convert_to_cv2bbox(trk)
            for d,det in enumerate(detections):
             #   det = convert_to_cv2bbox(det)
                IOU_mat[t,d] = self.box_iou2(trk,det)

        # Produces matches
        matched_idx = linear_assignment(-IOU_mat)

        unmatched_trackers, unmatched_detections = [], []
        for t,trk in enumerate(trackers):
            if(t not in matched_idx[:,0]):
                unmatched_trackers.append(t)

        for d, det in enumerate(detections):
            if(d not in matched_idx[:,1]):
                unmatched_detections.append(d)
        matches = []

        # For creating trackers we consider any detection with an
        # overlap less than iou_thrd to signifiy the existence of
        # an untracked object

        for m in matched_idx:
            if(IOU_mat[m[0],m[1]]<iou_thrd):
                unmatched_trackers.append(m[0])
                unmatched_detections.append(m[1])
            else:
                matches.append(m.reshape(1,2))

        if(len(matches)==0):
            matches = np.empty((0,2),dtype=int)
        else:
            matches = np.concatenate(matches,axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

    def assign_detections_to_trackers_hist(self, previous_frame, trackers, trk_features, current_frame, detections, iou_thrd = 0.3):
        '''
        From current list of trackers and new detections, output matched detections,
        unmatchted trackers, unmatched detections.
        '''
        IOU_mat= np.zeros((len(trackers),len(detections)),dtype=np.float32)
        index = 0
        for t,trk in enumerate(trackers):
            #trk = convert_to_cv2bbox(trk)
#            hist1 = self.Compute_HSI_Histogram(previous_frame,trk)
            hist1 = trk_features[index]
            for d,det in enumerate(detections):
             #   det = convert_to_cv2bbox(det)
                hist2 = self.Compute_HSI_Histogram(current_frame,det)
                score1 = 0.5 * self.box_iou2(trk,det)
                score2 = 0.5 * (1./(1. +self.returnHistogramComparison(hist1,hist2,'bhattacharyya')))
                IOU_mat[t,d] = score1 + score2 #self.box_iou2(trk,det)
            index += 1

        # Produces matches
        matched_idx = linear_assignment(-IOU_mat)

        unmatched_trackers, unmatched_detections = [], []
        for t,trk in enumerate(trackers):
            if(t not in matched_idx[:,0]):
                unmatched_trackers.append(t)

        for d, det in enumerate(detections):
            if(d not in matched_idx[:,1]):
                unmatched_detections.append(d)
        matches = []

        # For creating trackers we consider any detection with an
        # overlap less than iou_thrd to signifiy the existence of
        # an untracked object

        for m in matched_idx:
            if(IOU_mat[m[0],m[1]]<iou_thrd):
                unmatched_trackers.append(m[0])
                unmatched_detections.append(m[1])
            else:
                matches.append(m.reshape(1,2))

        if(len(matches)==0):
            matches = np.empty((0,2),dtype=int)
        else:
            matches = np.concatenate(matches,axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
