#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 14:48:29 2020

@author: venkatesh
"""
import cv2
import numpy as np
import os
import sys
import torchvision

import constants
import object_tracking as Tracker
import helper as helper
import io_helper as io
import image_helper as image_helper

from tqdm import tqdm
from config import config as cfg

def main(args=None):
    files = []
    for filename in os.listdir('/Users/adityabansal/Downloads/detection_files/'):
        if filename.endswith('.txt'):
            file_name = filename.split('.')
            files.append(file_name[0])


    for filename in files:
        obj_Tracking    = Tracker.Object_Tracking()
        img_Draw        = Tracker.Draw()
        #setting up directories of input and output----------------
        sub_output_Folder_Path = filename + '_age_2_thresh_0.4/'
        video_path = constants.dataset_directory_path + filename + '.mp4'
        Output_Folder_Path = constants.main_output_folder_path + sub_output_Folder_Path
        video_no = 0
        print(filename)
        if len(filename) == 5:
            video_no = int(filename[-1])
        else:
            if len(filename) == 6:
                video_no = int(filename[-2:])
            else:
                video_no = int(filename[4])
        roi_path = constants.roi_directory_path + 'cam_' + str(video_no) + '.txt'
        video_directory = 'cam_' + str(video_no)
        print(filename, video_path, Output_Folder_Path)
        video_reader = cv2.VideoCapture(video_path)
        num_files = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        all_boxes = io.read_all_boxes(constants.detection_input_path + filename + '.txt',num_files)
        io.make_required_directories(Output_Folder_Path)

        #setting up required variables for counting algorithm
        carInOrOut = np.zeros((10000,1), dtype=int)
        direction_entry = np.zeros((10000,1), dtype=int)
        direction_exit  = np.zeros((10000,1), dtype=int)
        for i in range(10000):
            direction_entry[i] = -1
            direction_exit[i] = -1
            carInOrOut[i] = -1
        MOI_count = {}
        #extracting polygon cordinates and edges from ROI file
        polygon = io.read_roi_file(roi_path)
        lines = helper.extract_lines_from_polygon(polygon)
        with tqdm(total= num_files, file=sys.stdout) as pbar:
                ImageFilename = []
                fCount = 0
                for count in range(num_files):

                    _, image = video_reader.read() 

                    cam_image_h, cam_image_w, _ = image.shape
                    org_img = image.copy()
                    temp_img = org_img.copy()
                    obj_bbox, obj_scores, obj_class = helper.filter_useful_boxes(all_boxes[fCount + 1], polygon, temp_img)
                    unique_obj_bbox, unique_obj_scores, unique_obj_class = helper.non_max_supression(obj_bbox, obj_scores, obj_class, constants.iou_threshold, obj_Tracking)
                    predict_image, Track_Info,Track_ID, Track_Class = obj_Tracking.KF_pipeline(unique_obj_bbox,
                                                                                           unique_obj_class,
                                                                                           org_img,
                                                                                           '')
                 
                    predict_image = img_Draw.draw_roi_boxes(predict_image,unique_obj_bbox,linewidth = 1)
                    f = open(Output_Folder_Path + 'Text/' + str(fCount + 1) + '.txt', 'w+')
                    Track_Info,Track_ID,Track_Class = io.read_tracking_output_file(video_directory, fCount + 1)

                    # Vehicle Counting initialisation
                    for index in range(len(Track_ID)): 
                        vehicleClass = Track_Class[index]
                        vehicleId = Track_ID[index]
                        vehicleStatus = carInOrOut[vehicleId]
                        vehicleBoundingBox = Track_Info[index].copy()
                        vehicleBoundingBox[2] = vehicleBoundingBox[2] - vehicleBoundingBox[0]
                        vehicleBoundingBox[3] = vehicleBoundingBox[3] - vehicleBoundingBox[1]
                        vehicleBoundingBox = helper.modify_if_at_corners(vehicleBoundingBox, cam_image_h, cam_image_w)
                        points_inside_the_polygon = helper.checkIfAnyVerticeLiesInsidePolygon(vehicleBoundingBox, polygon)
                        if (points_inside_the_polygon == 4):
                            vehicleResult = 1
                        else:
                            vehicleResult = 0
                        f.write(str(vehicleId) + "," + str(vehicleClass) + "," + str(vehicleStatus) + "," + str(vehicleBoundingBox[0]) + "," + str(vehicleBoundingBox[1]) + "," + str(vehicleBoundingBox[2]) + "," + str(vehicleBoundingBox[3]) + "," + str(vehicleResult) + "," + str(direction_entry[vehicleId]) + "," + str(direction_exit[vehicleId]) + '\n')
                        vehicleMidPoint = helper.getMidPointOfBoundingBox(vehicleBoundingBox)
                        #vehicleResult = helper.checkIfPointLiesInsidePolygon(vehicleMidPoint, polygon)
                        image_helper.draw_caption(temp_img, 
                                  (vehicleBoundingBox[0],vehicleBoundingBox[1] + vehicleBoundingBox[3],-1,-1),
                                  str(Track_ID[index]) + "," + str(vehicleStatus) + "," + str(vehicleResult) + "," + str(direction_entry[vehicleId]) + "," + str(direction_exit[vehicleId])
                                  )

                        image_helper.draw_caption(temp_img, (vehicleMidPoint[0], vehicleMidPoint[1], -1,-1), "MP")
                        if vehicleStatus == 2:
                            continue

                        if (vehicleStatus == -1) and (vehicleResult == 0):
                            carInOrOut[vehicleId] = 0
                            direction_entry_detected = helper.find_line_number_of_intersection(vehicleMidPoint, lines)
                            direction_entry[vehicleId] = direction_entry_detected
                            continue
                     
                        if (vehicleStatus == -1) and (vehicleResult == 1):
                            carInOrOut[vehicleId] = 1
                            direction_entry[vehicleId] = -1 
                            continue
                      
                      
                        if (vehicleStatus == 0) and (vehicleResult == 0):
                            continue
                          
                        if (vehicleStatus == 0) and (vehicleResult == 1):
                            carInOrOut[vehicleId] = 1
                            continue
                          
                        if (vehicleStatus == 1) and (vehicleResult == 0):
                            direction_exit_detected = helper.find_line_number_of_intersection(vehicleMidPoint, lines)
                            if direction_exit_detected == direction_entry[vehicleId]:
                                continue  
                            direction_exit[vehicleId] = direction_exit_detected
                            moi_key = helper.convert_directions_to_moi_key(direction_entry[vehicleId], direction_exit[vehicleId])
                            if moi_key in MOI_count:
                                MOI_count[moi_key] = MOI_count[moi_key] + 1
                            else:
                                MOI_count[moi_key] = 1
                            carInOrOut[vehicleId] = 2
                          
                        if vehicleStatus == 1 and vehicleResult == 1:
                            continue
                    f.close()
                    #image_helper.draw_roi_with_line_numbers_on_image(lines,temp_img)
                    #image_helper.display_results_on_image(MOI_count, temp_img)
                    # Display the resulting frame 
                    pts = np.array(polygon, np.int32)
                    pts = pts.reshape((-1,1,2))
                    cv2.polylines(temp_img,[pts],True,(0,255,255))

                    #cv2.imwrite(os.path.join(Output_Folder_Path + 'Count','Frame_{}.png'.format(fCount + 1)), temp_img)
                    #cv2.imwrite(os.path.join(Output_Folder_Path + 'Tracking','Frame_{}.png'.format(fCount + 1)), predict_image)
                    #cv2.imshow('Tracking', predict_image) 
                    #cv2.imshow('Detection', temp_img)
                    #cv2.waitKey(200) 
                    if cv2.waitKey(25) | 0xFF == ord('q'): 
                        cv2.destroyAllWindows()
                        break
                  
                    fCount += 1
                    pbar.set_description('processed: %d' % (fCount))
                    pbar.update(1)
                    #if fCount == 1000:
                    #    break
                

if __name__ == '__main__':
 main()