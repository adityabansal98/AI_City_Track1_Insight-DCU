
# coding: utf-8

# In[95]:


import cv2
import numpy as np
import os
from matplotlib import pyplot as plt


# In[122]:


f = open('/Volumes/BANSAL/modified_outputs/cam_5_rain/nodes.txt')
f_lines = f.readlines()
f.close()


# In[123]:


g = open('/Volumes/BANSAL/modified_outputs/cam_5_rain/vehicles.txt')
g_lines = g.readlines()
g.close()


# In[124]:


img = cv2.imread('/Users/adityabansal/Downloads/AIC20_track1/screen_shot_with_roi_and_movement/cam_5.jpg')


# In[125]:


video_id = 12


# In[126]:


vehicles = []
nodes = []


# In[127]:


for line in f_lines:
    split = line.split(' ')
    node = []
    for i in range(4):
        node.append(int(float(split[i + 1])))
    nodes.append(node)


# In[128]:


for line in g_lines:
    split = line.split(' ')
    vehicle = []
    for i in range(4):
        vehicle.append(int(split[i]))
    vehicles.append(vehicle)


# In[129]:


#print(vehicles[6000])


# In[130]:


#print(len(nodes), len(vehicles))


# In[131]:


entry_exit_map = {}
missed_map = {}


# In[132]:


for vehicle in vehicles:
    node_entry = vehicle[0]
    node_exit  = vehicle[1]
    key = str(node_entry) + ":" + str(node_exit)
    if node_exit != -1:
        if key in entry_exit_map:
            entry_exit_map[key] = entry_exit_map[key] + 1
        else:
            entry_exit_map[key] = 1
    else:
        if key in missed_map:
            missed_map[key] = missed_map[key] + 1
        else:
            missed_map[key] = 1


# In[133]:


for i in entry_exit_map:
    print (i, entry_exit_map[i])


# In[134]:


def draw_region(img, node_entry, node_exit,count):
    color_entry = (255,255,0)
    cv2.putText(img, 'entry', (nodes[node_entry][0], nodes[node_entry][1] ), cv2.FONT_HERSHEY_SIMPLEX, 1, (120, 0, 0), 1)
    cv2.rectangle(img, (nodes[node_entry][0],nodes[node_entry][1]) ,(nodes[node_entry][0] + nodes[node_entry][2],nodes[node_entry][1] + nodes[node_entry][3]),color_entry,1)
    color_exit = (255,0,0)
    cv2.putText(img, 'exit', (nodes[node_exit][0], nodes[node_exit][1] ), cv2.FONT_HERSHEY_SIMPLEX, 1, (120, 0, 0), 1)
    cv2.rectangle(img, (nodes[node_exit][0],nodes[node_exit][1]) ,(nodes[node_exit][0] + nodes[node_exit][2],nodes[node_exit][1] + nodes[node_exit][3]),color_exit,1)
    cv2.imwrite(os.path.join('/Volumes/BANSAL/modified_outputs/cam_5_rain/node_pics/',str(node_entry) + ':' + str(node_exit) + ':' + str(count) +  '.jpg'),img)


# In[135]:


for i in entry_exit_map:
    split = i.split(':')
    temp_img = img.copy()
    node_entry = int(split[0])
    node_exit = int(split[1])
    draw_region(temp_img, node_entry, node_exit,entry_exit_map[i])
    #plt.imshow(temp_img)


# In[136]:


plt.imshow(img) 


# In[137]:


h = open('/Volumes/BANSAL/modified_outputs/cam_5_rain/nodes_to_moi.txt')
h_lines = h.readlines()
h.close()


# In[138]:


entry_exit_to_MOI = {}


# In[139]:


for line in h_lines:
    line_split = line.split(',')
    node_entry = int(line_split[0])
    node_exit = int(line_split[1])
    moi = int(line_split[2])
    entry_exit_to_MOI[str(node_entry) + ':' + str(node_exit)] = moi


# In[152]:


h = open('/Volumes/BANSAL/modified_outputs/cam_5_rain/results.txt','w+')


# In[153]:


print(vehicles[5])


# In[154]:


for vehicle in vehicles:
    node_entry = vehicle[0]
    node_exit  = vehicle[1]
    key = str(node_entry) + ":" + str(node_exit)
    if key in entry_exit_to_MOI:
#         if entry_exit_to_MOI[key] == 2:
#             h.write(str(video_id) + ' ' + str(vehicle[2]) + ' ' + str(entry_exit_to_MOI[key] - 2) + ' ' + str(vehicle[3]) + '\n')
#         else:
        h.write(str(video_id) + ' ' + str(vehicle[2]) + ' ' + str(entry_exit_to_MOI[key]) + ' ' + str(vehicle[3]) + '\n')

