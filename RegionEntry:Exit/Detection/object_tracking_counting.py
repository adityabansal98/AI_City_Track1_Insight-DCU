#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 14:48:29 2020

@author: Multiple
"""

import cv2
import numpy as np
import os
import sys
import csv
import glob
import time
import re

from tqdm import tqdm
from config import config as cfg

from shapely.geometry import Polygon, LineString, Point
from itertools import groupby

from utils.sort import *

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = " "  # specify which GPU(s) to be used

history = {}
track_history_log_dir = ''

draw_on_image = False
show_image = False
save_image = False


def read_roi_file(moi_cordinates_file):
    # read polygon cordinates from file
    polygon = []
    f = open(moi_cordinates_file)
    g = f.readlines()
    countArea = []
    for h in g:
        cordinates = h.split(",")
        cordinates[0] = int(cordinates[0])
        cordinates[1] = int(cordinates[1][:-1])
        vertice = []
        vertice.append(cordinates[0])
        vertice.append(cordinates[1])
        countArea.append([cordinates[0], cordinates[1]])
        polygon.append(vertice)
    # print(polygon)

    indexLines = 1
    num_of_cords = len(countArea)
    polyside = []
    for index in range(0, num_of_cords-1):
        polyside.append([countArea[index], countArea[index+1]])

    polyside.append([countArea[num_of_cords-1], countArea[0]])

    return polygon, countArea, polyside

def read_detection_data(filepath):
    valid_class_labels = cfg.class_labels
    valid_class_labels = [x.lower() for x in valid_class_labels]

    valid_class_id = [valid_class_labels.index(label) for label in valid_class_labels]

    data = []
    f = open(filepath, "r")
    for line in f:
        # data format used for logging the detection output:
        # (frame,objectType,x1,y1,x2,y2,objectScore)

        line = line.strip()
        fields = line.split(" ")
        # get fields from table
        frame = int(fields[0])  # frame : frame number starts with 1
        obj_type = int(fields[1])  # object type [1:car, 2:truck]
        x1 = float(fields[2])  # left   [px]
        y1 = float(fields[3])  # top    [px]
        x2 = float(fields[4])  # right  [px]
        y2 = float(fields[5])  # bottom [px]
        x = int((x2 + x1) / 2)
        y = int((y1 + y2) / 2)
        w = x2 - x1
        h = y2 - y1
        obj_score = float(fields[6])

        if obj_type in valid_class_id:
            # obj_class = valid_class_labels.index(obj_type)
            data.append((frame, obj_type, x, y, w, h, obj_score))

    return np.array(data)

def read_video_info(video_info_filepath):
    video_info = []
    with open(video_info_filepath, mode = 'r') as text_file:
        for line in text_file.readlines():
            line = line.strip()
            id, videoname = re.split(r',| ', line)
            video_info.append(videoname)

    return video_info

def main(args=None):

    ai_city_data_dir = '/home/venkatesh/Desktop/AIC20_track1'
    data_dir_path = os.path.join(ai_city_data_dir, 'Dataset_A')
    video_file_info = '/home/venkatesh/Desktop/AIC20_track1/Dataset_A/list_video_id.txt'
    trainedWidth = 416
    trainedHeight = 416
    Output_Folder_Path = 'Results'

    detection_log_dir = os.path.join(Output_Folder_Path, 'detection')

    # video_file_list = glob.glob(os.path.join(data_dir_path, '*.mp4'))
    video_file_list = read_video_info(video_file_info)

    # Save_Output_Frame = os.path.join(Output_Folder_Path)
    if not os.path.exists(Output_Folder_Path):
        os.makedirs(Output_Folder_Path)

    tracking_log_dir = os.path.join(Output_Folder_Path, 'tracking')
    if not os.path.exists(tracking_log_dir):
        os.makedirs(tracking_log_dir)

    tracking_image_dir = os.path.join(Output_Folder_Path, 'tracking_img')

    if not os.path.exists(tracking_image_dir):
        os.makedirs(tracking_image_dir)

    global track_history_log_dir

    track_history_log_dir = os.path.join(Output_Folder_Path, 'track_history')
    if not os.path.exists(track_history_log_dir):
        os.makedirs(track_history_log_dir)

    ComputationTime_Filename = 'tracking_computation_time.txt'
    ComputationTimeFile = open(os.path.join(tracking_log_dir, ComputationTime_Filename), "w+")
    ComputationTimeFile.write('filename total_time_msec avg_time_msec\n')
    ComputationTimeFile.close()

    for video_file in video_file_list:

        video_path = os.path.join(data_dir_path, video_file)

        det_filename = os.path.splitext(os.path.basename(video_file))[0] + '.txt'
        roi_filename = os.path.splitext(os.path.basename(video_file))[0]
        temp_text = roi_filename.split("_")
        if len(temp_text) >= 2:
            roi_filename = temp_text[0] + '_' + temp_text[1] + '.txt'

        roi_path = os.path.join(ai_city_data_dir, 'ROIs', roi_filename)

        DectionFilename = os.path.join(detection_log_dir, det_filename)

        TrackingFile = open(os.path.join(tracking_log_dir, det_filename), "w+")
        TrackingFile .close()
        ComputationTimeFile = open(os.path.join(tracking_log_dir, ComputationTime_Filename), "a+")

        # file.close()


        video_reader = cv2.VideoCapture(video_path)
        num_files = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

        videoName = os.path.basename(video_path)
        filename = os.path.splitext(videoName)[0] + '.csv'
        history_filename = os.path.join(track_history_log_dir, filename)
        history_file = csv.writer(open(history_filename, "w"))

        tracking_image_dir_videoname = os.path.join(tracking_image_dir, videoName)
        if not os.path.exists(tracking_image_dir_videoname):
            os.makedirs(tracking_image_dir_videoname)

        # read the detection results
        detection_db = read_detection_data(DectionFilename)

        global history
        history = {}

        time.sleep(2)

        with tqdm(total=num_files, file=sys.stdout) as pbar:
            fCount=0
            tracking_computation_time = []
            mot_tracker = Sort(max_age=10, min_hits=2)

            polygon, countArea, polyside = read_roi_file(roi_path)
            pts = np.array(polygon, np.int32)
            pts = pts.reshape((-1, 1, 2))

            # restart count for each video
            KalmanBoxTracker.count = 0
            TrackingFile = open(os.path.join(tracking_log_dir, det_filename), "a+")
            for count in range(num_files):
                _, image = video_reader.read()
                cam_image_h, cam_image_w, _ = image.shape

                org_img = image.copy()
                image_h, image_w, _ = org_img.shape

                # prepare the data for tracking and counting
                obj_info = []
                obj_bbox = []
                det_in_frame = np.where(detection_db[:, 0] == (fCount + 1))[0]
                det_in_frame_data = detection_db[det_in_frame, :]
                obj_bbox = []
                obj_class = []
                scaled_obj_bbox = []
                for data in det_in_frame_data:
                    if data[6] > 0.4:

                        x = int(data[2])
                        y = int(data[3])
                        w = int(data[4])
                        h = int(data[5])
                        x1 = int(x - w / 2)
                        y1 = int(y - h / 2)
                        x2 = int(x + w / 2)
                        y2 = int(y + h / 2)
                        # boxColor = (255, 0, 255)

                        polya = Polygon(polygon)
                        bbox = [(x1, y1), (x1 + w, y1), (x2, y2), (x1, y1 + h)]
                        polyb = Polygon(bbox)

                        # cv2.rectangle(org_img, (x1, y1), (x2, y2), boxCol)or, thickness=2)
                        if polya.contains(polyb) or polya.intersects(polyb):
                            obj_info = []
                            obj_info.append(int(data[1]))
                            obj_info.append(float(data[6]))
                            obj_info.append([data[2], data[3], data[4], data[5]])
                            obj_bbox.append(obj_info)

                tracking_time_start = time.clock()

                videoName = os.path.basename(video_path)

                if len(obj_bbox):
                    frame = track_and_count(count, org_img, obj_bbox, cfg.class_labels, polyside, np.array(countArea), mot_tracker, videoName, TrackingFile)

                tracking_time_elapsed = (time.clock() - tracking_time_start)
                tracking_computation_time.append(tracking_time_elapsed)

                if draw_on_image:
                    cv2.polylines(frame, [pts], False, (255, 0, 0), thickness=2)
                    for side_index, line in enumerate(polyside):
                        cv2.putText(org_img, str(side_index), (line[0][0], line[0][1]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

                if save_image:
                    resize_img = cv2.resize(org_img, (int(image_w/4), int(image_h/4)), interpolation = cv2.INTER_AREA)
                    cv2.imwrite(os.path.join(tracking_image_dir_videoname, 'Frame_{}.png'.format(fCount + 1)), resize_img)

                if show_image:
                    cv2.imshow('output', org_img)

                if cv2.waitKey(25) | 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break

                fCount += 1
                pbar.set_description('processed %s: %d' % (videoName, fCount))
                pbar.update(1)

        ComputationTimeFile.write('{} {:0.2f} {:0.2f}\n'.format(video_file, np.sum(tracking_computation_time) * 1000,
                                                                (np.mean(tracking_computation_time)) * 1000))
        TrackingFile.close()
        ComputationTimeFile.close()

        break

def pointInCountArea(painting, AreaBound, point):
    h,w = painting.shape[:2]
    point = np.array(point)
    point = point - AreaBound[:2]
    if point[0] < 0 or point[1] < 0 or point[0] >= w or point[1] >= h:
        return 0
    else:
        return painting[point[1],point[0]]

def filiter_out_repeat(objects):
    objects = sorted(objects,key=lambda x: x[1])
    l = len(objects)
    new_objects = []
    if l > 1:
        for i in range(l-1):
            flag = 0
            for j in range(i+1,l):
                x_i, y_i, w_i, h_i = objects[i][2]
                x_j, y_j, w_j, h_j = objects[j][2]
                box1 = [int(x_i - w_i / 2), int(y_i - h_i / 2), int(x_i + w_i / 2), int(y_i + h_i / 2)]
                box2 = [int(x_j - w_j / 2), int(y_j - h_j / 2), int(x_j + w_j / 2), int(y_j + h_j / 2)]
                if cal_iou(box1,box2) >= 0.7:
                    flag = 1
                    break
            #if no repeat
            if not flag:
                new_objects.append(objects[i])
        #add the last one
        new_objects.append(objects[-1])
    else:
        return objects

    return list(tuple(new_objects))

def get_objName(item,objects):
    iou_list = []
    for i,object in enumerate(objects):
        x, y, w, h = object[2]
        x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
        iou_list.append(cal_iou(item[:4],[x1,y1,x2,y2]))
    max_index = iou_list.index(max(iou_list))
    return objects[max_index][0], objects[max_index][2]

def cal_iou(box1,box2):
    x1 = max(box1[0],box2[0])
    y1 = max(box1[1],box2[1])
    x2 = min(box1[2],box2[2])
    y2 = min(box1[3],box2[3])
    i = max(0,(x2-x1))*max(0,(y2-y1))
    u = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) -  i
    iou = float(i)/float(u)
    return iou

def track_and_count(frame_no, frame, objects, valid_class, polygonSides, CountArea, mot_tracker, videoName, File):

    # filter out repeat bbox
    objects = filiter_out_repeat(objects)

    detections = []
    for item in objects:
        detections.append([int(item[2][0] - item[2][2] / 2),
                           int(item[2][1] - item[2][3] / 2),
                           int(item[2][0] + item[2][2] / 2),
                           int(item[2][1] + item[2][3] / 2),
                           item[1]])


    track_bbs_ids = mot_tracker.update(np.array(detections))

    if len(track_bbs_ids) > 0:
        for bb in track_bbs_ids:  # add all bbox to history
            id = int(bb[-1])
            objectName, objectPosition = get_objName(bb, objects)
            if id not in history.keys():  # add new id
                history[id] = {}
                history[id]["no_update_count"] = 0
                history[id]["his"] = []
                history[id]["his"].append(objectName)
                history[id]["pos"] = []
                history[id]["pos"].append(objectPosition)
                x, y, w, h = objectPosition
                File.write('{} {} {} {} {} {} {}\n'.format(frame, id, objectName, x, y, w, h))
            else:
                history[id]["no_update_count"] = 0
                history[id]["his"].append(objectName)
                history[id]["pos"].append(objectPosition)
                x, y, w, h = objectPosition
                File.write('{} {} {} {} {} {} {}\n'.format(frame, id, objectName, x, y, w, h))

    for i, item in enumerate(track_bbs_ids):
        bb = list(map(lambda x: int(x), item))
        id = bb[-1]
        x1, y1, x2, y2 = bb[:4]

        his = history[id]["his"]
        result = {}
        for i in set(his):
            result[i] = his.count(i)
        res = sorted(result.items(), key=lambda d: d[1], reverse=True)
        objectName = valid_class[res[0][0]].lower()

        boxColor = cfg.colorDict[objectName]
        cv2.rectangle(frame, (x1, y1), (x2, y2), boxColor, thickness=2)
        cv2.putText(frame, str(id) + "_" + objectName, (x1 - 1, y1 - 3), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                    boxColor,
                    thickness=2)

    counter_results = []
    # videoName = videoName.split('/')[-1]
    removed_id_list = []
    for id in history.keys():  # extract id after tracking
        history[id]["no_update_count"] += 1
        if history[id]["no_update_count"] > 5:
            his = history[id]["his"]
            result = {}
            for i in set(his):
                result[i] = his.count(i)
            res = sorted(result.items(), key=lambda d: d[1], reverse=True)
            objectName = valid_class[res[0][0]].lower()
            counter_results.append([videoName, id, objectName])
            # del id
            removed_id_list.append(id)

    filename = os.path.splitext(videoName)[0] + '.csv'
    history_filename = os.path.join(track_history_log_dir, filename)
    file = csv.writer(open(history_filename, "a+"))
    for id in removed_id_list:
        if len(history[id]["pos"]) > 5:
            his = history[id]["his"]
            result = {}
            for i in set(his):
                result[i] = his.count(i)
            res = sorted(result.items(), key=lambda d: d[1], reverse=True)
            objectName = valid_class[res[0][0]].lower()
            file.writerow([id, objectName, history[id]['pos']])
        _ = history.pop(id)

    # if len(counter_results):
    #     self.sin_counter_results.emit(counter_results)



    return frame

if __name__ == '__main__':
    main()