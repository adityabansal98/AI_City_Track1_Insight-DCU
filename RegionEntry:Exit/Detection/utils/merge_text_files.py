#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Python program to
# merging of multiple files

import os
import glob


# path to the saved results directory
input_dir = '/home/venkatesh/Desktop/vehicle_outputs_m1+m2+nms'
output_dir = os.path.join(input_dir, 'Merged_Results')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# get all the folders in the directory
sub_folder_list = os.listdir(input_dir)

for folder in sub_folder_list:
    # retrive all the text files information in a folder to merge to single file
    path = os.path.join(input_dir, folder, 'Text')
    # Creating a list of filenames
    text_file_list = glob.glob(os.path.join(path, "*.txt"))

    print('{} : {}'.format(folder, len(text_file_list)))

    merged_text_file = os.path.join(output_dir, folder + '.txt')

    # Open file3 in write mode
    with open(merged_text_file, 'w') as outfile:
        # Iterate through list
        for file in text_file_list:
            # Open each file in read mode
            with open(file) as infile:
                # read the data from file1 and
                # file2 and write it in file3
                outfile.write(infile.read())

                # Add '\n' to enter data of file2
            # from next line
            outfile.write("\n")
