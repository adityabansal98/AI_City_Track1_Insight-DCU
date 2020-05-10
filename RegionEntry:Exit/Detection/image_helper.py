import cv2
import numpy as np

def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
    
    return image

def draw_roi_with_line_numbers_on_image(lines, image):
    line_no = 1
    for i in lines:
        start = i[0]
        end   = i[1]
        mid_point_x = int((start[0] + end[0])/2)
        mid_point_y = int((start[1] + end[1])/2)
        draw_caption(image, (mid_point_x, mid_point_y, -1, -1), "Line " + str(line_no))
        line_no = line_no + 1

def display_results_on_image(result_dict, image):
    linespace = 0
    image_h, image_w, _ = image.shape
    for i in result_dict:
        to_write = str(i) + " " + str(result_dict[i])
        draw_caption(image, (image_w/2, image_h/2 + linespace, -1, -1), to_write)
        linespace += 12