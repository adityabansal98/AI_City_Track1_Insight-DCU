import cv2 
import matplotlib.path as mpltPath
import numpy as np

import constants
import image_helper as image_helper
import object_tracking as Tracker

def modify_if_at_corners(box,image_h, image_w):
    x_left = box[0]
    x_right = box[0] + box[2]
    y_top = box[1]
    y_bottom = box[1] + box[3]
    if(x_left < 20):
        box[0] = 0
    if(x_right > image_w - 20):
        box[2] += 20
    if(y_top < 20):
        box[1] = 0
    if(y_bottom > image_h - 20):
        box[3] += 20
    return box

#pass the closest point instead of mid point of vehicle
def find_line_number_of_intersection(vehicleMidPoint, lines):
    distances_from_lines = np.zeros((len(lines),1),dtype=int)
    for i in range(len(lines)):
        distances_from_lines[i] = getPerpendicularDistance(lines[i],(vehicleMidPoint))
        #min_distance_line = np.where(distances_from_lines == np.amin(distances_from_lines))[0] #+ 1
        min_distance = 1000 #to be changed
        min_distance_line = -1
        for i in range(len(distances_from_lines)):
            if distances_from_lines[i] < min_distance:
                min_distance = distances_from_lines[i]
                min_distance_line = i
    return min_distance_line + 1

def filter_useful_boxes(boxes,polygon, temp_img):
    obj_bbox = []
    obj_scores = []
    obj_class = []
    for i in range(len(boxes)):
                bbox = boxes[i]
                print(bbox)
                obj_confidence_score = bbox[4]
                if obj_confidence_score >= constants.Threshold_obj_score:
                    x1 = int(bbox[0])
                    y1 = int(bbox[1])
                    x2 = int(bbox[2])
                    y2 = int(bbox[3])
                    if (checkIfAnyVerticeLiesInsidePolygon([x1,y1,x2 - x1,y2 - y1], polygon) != 0):
                        obj_bbox.append([x1,y1,x2,y2])
                        obj_scores.append(obj_confidence_score)
                        obj_class.append(bbox[5])
                        label_name = ''
                        if bbox[5] == 1:
                            label_name = 'car'
                        else:
                            label_name = 'truck'
                        image_helper.draw_caption(temp_img, (x1, y1, x2, y2), label_name + '-'+ str(obj_confidence_score))
                        cv2.rectangle(temp_img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
    return obj_bbox, obj_scores, obj_class
def extract_objects_from_predictions(temp_boxes, temp_img, scale_w, scale_h, polygon, valid_class):
    obj_bbox = []
    obj_scores = []
    obj_class = []
    for obj_class_id in constants.list_of_classes:
        if str(obj_class_id) in constants.classes:
            label_name=constants.classes[str(obj_class_id)]
        if len(temp_boxes) !=0:
            for i in range(len(temp_boxes[obj_class_id])):
                bbox = temp_boxes[obj_class_id][i]
                obj_confidence_score = bbox[4]
                if obj_confidence_score >= constants.Threshold_obj_score:
                    x1 = int(bbox[0]*scale_w)
                    y1 = int(bbox[1]*scale_h)
                    x2 = int(bbox[2]*scale_w)
                    y2 = int(bbox[3]*scale_h)
                    if (checkIfAnyVerticeLiesInsidePolygon([x1,y1,x2 - x1,y2 - y1], polygon) != 0):
                        obj_bbox.append([x1,y1,x2,y2])
                        obj_scores.append(obj_confidence_score)
                        obj_index = valid_class.index(label_name)
                        obj_class.append(obj_index)
                        image_helper.draw_caption(temp_img, (x1, y1, x2, y2), label_name + '-'+ str(obj_confidence_score))
                        cv2.rectangle(temp_img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
    return obj_bbox, obj_scores, obj_class

def get_predictions(model,sess,image,inputs):
    img  = cv2.resize(image,(constants.trainedWidth,constants.trainedHeight))
    _data = np.array(img).reshape(-1,constants.trainedWidth,constants.trainedHeight,3)
    preds = sess.run(model.preds, {inputs: model.preprocess(_data)})
    boxes = model.get_boxes(preds, _data.shape[1:3])
    return np.array(boxes)

def convert_directions_to_moi_key(entry, exit):
    return str(entry) + ":" + str(exit)

def extract_lines_from_polygon(polygon):
    lines = []
    for i in range(len(polygon)):
        start_point = polygon[i]
        end_point   = polygon[(i + 1)%len(polygon)]
        lines.append((start_point, end_point))
        
    return lines

#Assumption - boundingBox = [left, top, width, height]
def getMidPointOfBoundingBox(boundingBox):
    x_mid = boundingBox[0] + boundingBox[2]/2
    y_mid = boundingBox[1] + boundingBox[3]/2
    return [x_mid,y_mid]

def getPerpendicularDistance(line, point):
    point = np.asarray(point)
    line = np.asarray(line)
    dist = np.abs(np.linalg.norm(np.cross(line[1] - line[0], line[0] - point))/np.linalg.norm(line[1] - line[0]))
    return float(dist)

#Assumption - polygon is in the format -> [(x1,y1),(x2,y2)...(xn,yn)]
def checkIfPointLiesInsidePolygon(point, polygon):
    path = mpltPath.Path(polygon) #optimize this, no need to call it again and again
    return path.contains_points([point])

def checkIfAnyVerticeLiesInsidePolygon(boundingBox, polygon):
    left = boundingBox[0]
    top = boundingBox[1]
    right = boundingBox[0] + boundingBox[2]
    bottom = boundingBox[1] + boundingBox[3]
    count = 0
    if(checkIfPointLiesInsidePolygon([left, top], polygon)):
        count = count + 1
    if(checkIfPointLiesInsidePolygon([left, bottom], polygon)):
        count = count + 1
    if(checkIfPointLiesInsidePolygon([right, top], polygon)):
        count = count + 1
    if(checkIfPointLiesInsidePolygon([right, bottom], polygon)):
        count = count + 1
    return count

def remove_boxes_not_in_roi(boxes, polygon):
    boxes_in_roi = []
    for box in boxes:
        if(checkIfAnyVerticeLiesInsidePolygon(box,polygon) != [-1,-1]):
            boxes_in_roi.append(box)
    return boxes_in_roi

def remove_overlapping_bounding_box(b_box_list, threshold, obj_Tracking):
    b_box_list.sort(key=lambda x: x[0])
    new_b_box_list = []
    done = np.zeros((len(b_box_list),1),dtype=bool)
    for i in range(len(b_box_list)):
        if done[i] == True:
            continue
        x1_beg = b_box_list[i][0]
        x1_end = b_box_list[i][2]
        flag = False
        for j in range(i + 1, len(b_box_list)):
            x2_beg = b_box_list[j][0]
            if(x2_beg > (x1_end - x1_beg)*threshold):
                break
            iou = obj_Tracking.box_iou2(b_box_list[i], b_box_list[j])
            #print(iou)
            if obj_Tracking.box_iou2(b_box_list[i], b_box_list[j]) >= threshold:
                if b_box_list[i][4] > b_box_list[j][4]:
                    done[j] = True
                    continue
                else:
                    flag = True
                    break
        done[i] = True
        if flag == False:
            new_b_box_list.append(b_box_list[i])
    return new_b_box_list

def non_max_supression(boxes, scores, classes, threshold, obj_Tracking):
    done = np.zeros((len(boxes),1),dtype=bool)
    new_boxes = []
    new_scores = []
    new_classes = []
    sorted_indices = sorted(range(len(scores)), key = lambda k: scores[k], reverse = True)
    for i in range(len(boxes)):
        if done[i] == True:
            continue
        index_i = sorted_indices[i]
        new_boxes.append(boxes[index_i])
        new_scores.append(scores[index_i])
        new_classes.append(classes[index_i])
        for j in range(i + 1, len(boxes)):
            if done[j] == True:
                continue
            index_j = sorted_indices[j]
            iou = obj_Tracking.box_iou2(boxes[index_i], boxes[index_j])
            if iou >= threshold:
                done[j] = True
    return new_boxes, new_scores, new_classes


def remove_duplicates(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
 
    # # if the bounding boxes integers, convert them to floats --
    # # this is important since we'll be doing a bunch of divisions
    # if boxes.dtype.kind == "i":
    #   boxes = boxes.astype("float")
 
    # initialize the list of picked indexes 
    pick = []
 
    # grab the coordinates of the bounding boxes
    boxes  = np.array(boxes)
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
 
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
        idxs = np.delete(idxs, np.concatenate(([last],np.where(overlap > overlapThresh)[0])))
 
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int"), pick