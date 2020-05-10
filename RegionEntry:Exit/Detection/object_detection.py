#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 06:50:40 2020

@author: venkatesh
"""

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 14:48:29 2020

@author: venkatesh
"""

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 12:32:07 2020

@author: venkatesh
"""

import numpy as np
import torchvision
import time
import os
import copy
import pdb
import time
import argparse

import sys
import cv2
import glob

import csv

import skimage.io
import skimage.transform
import skimage.color
import skimage

import tensorflow as tf
import tensornets as nets

import matplotlib.lines as mlines
import matplotlib.path as mpltPath
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import object_tracking as Tracker

from tqdm import tqdm
from config import config as cfg

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # specify which GPU(s) to be used

from six import raise_from


def _open_for_csv(path):
    """
    Open a file with flags suitable for csv.reader.
    This is different for python2 it means with mode 'rb',
    for python3 this means 'r' with "universal newlines".
    """
    if sys.version_info[0] < 3:
        return open(path, 'rb')
    else:
        return open(path, 'r', newline='')


def _parse(value, function, fmt):
    """
    Parse a string into a value, and format a nice ValueError if it fails.
    Returns `function(value)`.
    Any `ValueError` raised is catched and a new `ValueError` is raised
    with message `fmt.format(e)`, where `e` is the caught `ValueError`.
    """
    try:
        return function(value)
    except ValueError as e:
        raise_from(ValueError(fmt.format(e)), None)


def load_classes(csv_reader):
    result = {}

    for line, row in enumerate(csv_reader):
        line += 1

        try:
            class_name, class_id = row
        except ValueError:
            raise_from(ValueError('line {}: format should be \'class_name,class_id\''.format(line)), None)
        class_id = _parse(class_id, int, 'line {}: malformed class ID: {{}}'.format(line))

        if class_name in result:
            raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
        result[class_name] = class_id
    return result


def read_csv_classes(class_list):
    with _open_for_csv(class_list) as file:
        classes = load_classes(csv.reader(file, delimiter=','))

    labels = {}
    for key, value in classes.items():
        labels[value] = key

    return labels


def load_image(image_path):
    img = skimage.io.imread(image_path)

    if len(img.shape) == 2:
        img = skimage.color.gray2rgb(img)

    return img.astype(np.float32) / 255.0


def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    return image


def read_roi_file(moi_cordinates_file):
    # read polygon cordinates from file
    polygon = []
    f = open(moi_cordinates_file)
    g = f.readlines()
    for h in g:
        cordinates = h.split(",")
        cordinates[0] = int(cordinates[0])
        cordinates[1] = int(cordinates[1][:-1])
        vertice = []
        vertice.append(cordinates[0])
        vertice.append(cordinates[1])
        polygon.append(vertice)
    print(polygon)

    return polygon


def extract_lines_from_polygon(polygon):
    lines = []
    for i in range(len(polygon)):
        start_point = polygon[i]
        end_point = polygon[(i + 1) % len(polygon)]
        lines.append((start_point, end_point))

    return lines


# Assumption - polygon is in the format -> [(x1,y1),(x2,y2)...(xn,yn)]
def checkIfPointLiesInsidePolygon(point, polygon):
    path = mpltPath.Path(polygon)  # optimize this, no need to call it again and again
    return path.contains_points([point])


# Assumption - boundingBox = [left, top, width, height]
def getMidPointOfBoundingBox(boundingBox):
    x_mid = boundingBox[0] + boundingBox[2] / 2
    y_mid = boundingBox[1] + boundingBox[3] / 2
    return [x_mid, y_mid]


def getPerpendicularDistance(line, point):
    point = np.asarray(point)
    line = np.asarray(line)
    dist = np.abs(np.linalg.norm(np.cross(line[1] - line[0], line[0] - point)) / np.linalg.norm(line[1] - line[0]))
    return float(dist)


def convert_directions_to_moi_key(entry, exit):
    return str(entry) + ":" + str(exit)


def remove_duplicates(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    #	# if the bounding boxes integers, convert them to floats --
    #	# this is important since we'll be doing a bunch of divisions
    #	if boxes.dtype.kind == "i":
    #		boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    boxes = np.array(boxes)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int"), pick

def read_video_info(video_info_filepath):
    video_info = []
    with open(video_info_filepath, mode = 'r') as text_file:
        for line in text_file.readlines():
            line = line.strip()
            id, videoname = re.split(r',| ', line)
            video_info.append(videoname)

    return video_info

def main(args=None):
    data_dir_path = '/media/venkatesh/HDD_2/data/AI_city/AIC20_track1/Dataset_A'
    csv_val = 'data/val.csv'
    csv_classes = 'data/class_list.csv'
    model = 'model_files/model_final.pt'
    video_file_info = '/home/venkatesh/Desktop/AIC20_track1/Dataset_A/list_video_id.txt'
    trainedWidth = 416
    trainedHeight = 416
    Output_Folder_Path = 'Results'
    video_path = 'input/cam_1.mp4'

    roi_path = '/media/venkatesh/HDD_2/data/AI_city/AIC20_track1/ROIs/cam_1.txt'

    classes = {'0': 'person', '1': 'bicycle', '2': 'car', '3': 'bike', '5': 'bus', '7': 'truck'}
    #    valid_class = cfg.class_labels
    valid_class = [label.lower() for label in cfg.class_labels]
    list_of_classes = [2, 7]

    video_file_list = read_video_info(video_file_info) #glob.glob(os.path.join(data_dir_path, '*.mp4'))
    #

    # Save_Output_Frame = os.path.join(Output_Folder_Path)
    if not os.path.exists(Output_Folder_Path):
        os.makedirs(Output_Folder_Path)

    detection_log_dir = os.path.join(Output_Folder_Path, 'detection')
    if not os.path.exists(detection_log_dir):
        os.makedirs(detection_log_dir)

    #    dataset_val    = CSVDataset(data_dir = data_dir_path, train_file=csv_val, class_list=csv_classes,
    #                             transform=transforms.Compose([Normalizer(), Resizer()]))
    #
    #    sampler_val    = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
    #    dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)

    inputs = tf.placeholder(tf.float32, [None, trainedWidth, trainedHeight, 3])
    model = nets.YOLOv3COCO(inputs, nets.Darknet19)

    Threshold_obj_score = 0.1

    ComputationTime_Filename = 'detection_computation_time.txt'
    ComputationTimeFile = open(os.path.join(detection_log_dir, ComputationTime_Filename), "w+")
    ComputationTimeFile.write('filename total_time_msec avg_time_msec\n')
    ComputationTimeFile.close()

    with tf.Session() as sess:
        sess.run(model.pretrained())

        for video_file in video_file_list:

            video_path = os.path.join(data_dir_path, video_file)

            det_filename = os.path.splitext(os.path.basename(video_file))[0] + '.txt'
            DectionFile = open(os.path.join(detection_log_dir, det_filename), "w+")
            ComputationTimeFile = open(os.path.join(detection_log_dir, ComputationTime_Filename), "a+")

            video_reader = cv2.VideoCapture(video_path)
            num_files = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

            with tqdm(total=num_files, file=sys.stdout) as pbar:
                fCount = 0
                DP_computation_Time = []
                for count in range(num_files):
                    _, image = video_reader.read()
                    cam_image_h, cam_image_w, _ = image.shape

                    org_img = image.copy()
                    image_h, image_w, _ = org_img.shape
                    scale_w = cam_image_w / trainedWidth
                    scale_h = cam_image_h / trainedHeight

                    img = cv2.resize(image, (trainedWidth, trainedHeight))
                    _data = np.array(img).reshape(-1, trainedWidth, trainedHeight, 3)

                    DP_time_start = time.clock()

                    # Detection module call
                    preds = sess.run(model.preds, {inputs: model.preprocess(_data)})
                    boxes = model.get_boxes(preds, _data.shape[1:3])

                    DP_time_elapsed = (time.clock() - DP_time_start)
                    DP_computation_Time.append(DP_time_elapsed)

                    temp_boxes = np.array(boxes)
                    obj_bbox = []
                    obj_class = []
                    for obj_class_id in list_of_classes:
                        if str(obj_class_id) in classes:
                            label_name = classes[str(obj_class_id)]
                        if len(temp_boxes) != 0:
                            for i in range(len(temp_boxes[obj_class_id])):
                                bbox = temp_boxes[obj_class_id][i]
                                obj_confidence_score = bbox[4]
                                if obj_confidence_score >= Threshold_obj_score:
                                    x1 = int(bbox[0] * scale_w)
                                    y1 = int(bbox[1] * scale_h)
                                    x2 = int(bbox[2] * scale_w)
                                    y2 = int(bbox[3] * scale_h)
                                    obj_bbox.append([x1, y1, x2, y2])
                                    #                            label_name = labels[int(classification[idxs[0][j]])]
                                    obj_index = valid_class.index(label_name)
                                    obj_class.append(obj_index)
                                    DectionFile.write(
                                        '{} {} {} {} {} {} {:0.4f}\n'.format(fCount + 1, obj_index, x1, y1, x2, y2,
                                                                             obj_confidence_score))

                    if cv2.waitKey(25) | 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        break

                    fCount += 1
                    pbar.set_description('processed: %d' % (fCount))
                    pbar.update(1)

            ComputationTimeFile.write(
                '{} {:0.2f} {:0.2f}\n'.format(os.path.basename(video_file), np.sum(DP_computation_Time) * 1000,
                                              (np.mean(DP_computation_Time)) * 1000))
            DectionFile.close()
            ComputationTimeFile.close()


if __name__ == '__main__':
    main()