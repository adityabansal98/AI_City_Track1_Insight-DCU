#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Multiple
"""

import os
import glob
import matplotlib.pyplot as plt
import csv
import re
import numpy as np
from ast import literal_eval
from dipy.tracking.streamline import set_number_of_points

# from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import ResampleFeature
from dipy.segment.metric import AveragePointwiseEuclideanMetric
# from dipy.segment.clustering import QuickBundles


from dipy.io.streamline import load_tractogram
from dipy.tracking.streamline import Streamlines
from dipy.segment.clustering import QuickBundles
from dipy.io.pickles import save_pickle
from dipy.data import get_fnames
from dipy.viz import window, actor

import pandas as pd
from sklearn.cluster import KMeans


def read_track_trajectories(filepath):
    track_info = {}
    with open(filepath, mode = 'r') as csv_file:
        reader = csv.reader(csv_file)
        linecount = 0
        for row in reader:
            obj_ID, obj_class, obj_trajectory = row
            obj_ID = int(obj_ID)
            track_info[obj_ID] = {}
            track_info[obj_ID]['obj_class'] = obj_class
            obj_trajectory = literal_eval(obj_trajectory)
            track_info[obj_ID]['traj'] = obj_trajectory

            track_info[obj_ID]['traj_point'] = []
            for pos in obj_trajectory:
                x = int(pos[0] + pos[2]/2)
                y = int(pos[1] + pos[3]/2)
                track_info[obj_ID]['traj_point'].append([x,y])

            linecount += 1

    return track_info

def main(args=None):

    # fname = get_fnames('fornix')
    #
    # fornix = load_tractogram(fname, 'same', bbox_valid_check=False)
    # streamlines = fornix.streamlines
    #
    # temp_data = streamlines.data[0]
    #
    # print(temp_data)
    #
    # print(len(streamlines.data))
    #
    # qb = QuickBundles(threshold=10.)
    # clusters = qb.cluster(streamlines)

    filepath = '/home/venkatesh/Desktop/Vehicle_counting_pipeline (extract.me)/Results/track_history/cam_7_dawn.csv'
    track_information = read_track_trajectories(filepath)

    plt.figure(figsize=(6, 6))

    obj_trajectory = []
    x_pos = []
    y_pos = []
    data = []
    for obj_id in track_information:
        cord_points = np.array(track_information[obj_id]['traj_point'])

        # data = []
        # x_pos = []
        # y_pos = []
        for index in range(len(cord_points)):
            x_pos.append(cord_points[index, 0])
            y_pos.append(cord_points[index, 1])
            data.append([cord_points[index, 0], cord_points[index, 1]])


        obj_trajectory.append(np.asarray(data, dtype = 'int32'))

    # print(obj_trajectory[0])

    # qb = QuickBundles(threshold=4.)
    streamlines = set_number_of_points(obj_trajectory, nb_points=50)

    # print(streamlines[0])
    # clusters = qb.cluster(streamlines)

    # Streamlines will be resampled to 24 points on the fly.
    # feature = ResampleFeature(nb_points=24)
    # metric = AveragePointwiseEuclideanMetric(feature=streamlines)  # a.k.a. MDF
    qb = QuickBundles(threshold=200.)
    clusters = qb.cluster(streamlines)

    print("Nb. clusters:", len(clusters))
    print("Cluster sizes:", list(map(len, clusters)))

    # Enables/disables interactive visualization
    interactive = False

    ren = window.Renderer()
    ren.SetBackground(1, 1, 1)
    ren.add(actor.streamtube(streamlines, window.colors.white))
    window.record(ren, out_path='fornix_initial.png', size=(600, 600))
    if interactive:
        window.show(ren)
    # wcss = []
    # for i in range(1, 11):
    #     kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    #     kmeans.fit(streamlines)
    #     wcss.append(kmeans.inertia_)
    # plt.plot(range(1, 11), wcss)
    # plt.title('Elbow Method')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('WCSS')
    # plt.show()
    #
    # kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
    # pred_y = kmeans.fit_predict(obj_trajectory)
    # plt.plot(cord_points[:, 0], cord_points[:, 1])
    # plt.plot(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1])
    # # plt.plot(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
    # plt.show()

    return 0



if __name__ == '__main__':
    main()