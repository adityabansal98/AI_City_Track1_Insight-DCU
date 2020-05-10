import cv2
import numpy as np
import os
import skimage.io
import skimage.transform
import skimage.color
import skimage

def draw_caption(image, box, caption):

    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
    
    return image

def load_image(image_path):
    img = skimage.io.imread(image_path)

    if len(img.shape) == 2:
        img = skimage.color.gray2rgb(img)

    return img.astype(np.float32)/255.0

def read_roi_file(moi_cordinates_file):
    #read polygon cordinates from file
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
    
    return polygon, countArea

def make_required_directories(Output_Folder_Path):
    if not os.path.exists(Output_Folder_Path):
        os.makedirs(Output_Folder_Path)
        os.makedirs(Output_Folder_Path + 'Count')
        os.makedirs(Output_Folder_Path + 'Tracking')
        os.makedirs(Output_Folder_Path + 'Text')

'''

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
'''