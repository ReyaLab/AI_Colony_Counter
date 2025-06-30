#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 12:17:55 2025

@author: mattc

This just flips and rotates a folder of images to increase training data for a cNN

This script is just for making training images, not needed for operation of program
"""

def visualize_segmentation(mask, image=0):
    import matplotlib.pyplot as plt
    import numpy as np
    plt.figure(figsize=(10, 5))

    if(not np.isscalar(image)):
        # Show original image if it is entered
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis("off")

    # Show segmentation mask
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap="gray")  # Show as grayscale
    plt.title("Segmentation Mask")
    plt.axis("off")

    plt.show()

import os

path = '/home/mattc/Documents/ColonyAssaySegformer/Mari_trainingset2/labels/'

file_list = ["",""]

file_list = os.listdir(path)
file_list = [x for x in file_list if (x[-4::]==".tif" or x[-5::]==".tiff")]
file_list = [x for x in file_list if ('Mask2' not in x)]

import cv2


#This section combines the necrosis masks and the colony masks
for x in file_list:
    img = cv2.imread(path + x)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img[img>0] = 1
    img2 = cv2.imread(path + x.replace("Mask", "Mask2"))
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    #the program filters out all necrosis for training data that is outside of training data colonies
    img2[img==0] = 0
    img[img2>0] = 2
    name = x.split(".")[0]
    cv2.imwrite(path+name+'tern.tif', img)
del img, img2, x, name

path = '/home/mattc/Documents/ColonyAssaySegformer/Mari_trainingset2/labels/'

file_list = ["",""]

file_list = os.listdir(path)
file_list = [x for x in file_list if (x[-4::]==".tif" or x[-5::]==".tiff")]

import cv2

#This section creates a y axis flip of each image
for x in file_list:
    img = cv2.imread(path + x)
    img =cv2.flip(img,1)
    name = x.split(".")[0]
    cv2.imwrite(path+name+'F.tif', img)


#This section creates 90, 180, and 270 degree rotations of each image    

file_list = os.listdir(path)
file_list = [x for x in file_list if (x[-4::]==".tif" or x[-5::]==".tiff")]

for x in file_list:
    img = cv2.imread(path + x)
    name = x.split(".")[0]
    img =cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(path+name+'R90.tif', img)
    img =cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(path+name+'R180.tif', img)
    img =cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(path+name+'R270.tif', img)
del x, img, name
#This section crops the 1536x2048 images into 512x512 chunks    

import os
import cv2
path = '/home/mattc/Documents/ColonyAssaySegformer/Mari_trainingset2/labels/'
file_list = os.listdir(path)
file_list = [x for x in file_list if (x[-4::]==".tif" or x[-5::]==".tiff")]
path2 = '/home/mattc/Documents/ColonyAssaySegformer/Mari_trainingset2/labels_cropped/'
for x in file_list:
    img = cv2.imread(path + x)
    name = x.split(".")[0]
    i_num = img.shape[0]/512
    j_num = img.shape[1]/512
    count = 1
    for i in range(int(i_num)):
        for j in range(int(j_num)):
            img2 = img[(512*i):(512*(i+1)), (512*j):(512*(j+1))]
            cv2.imwrite(path2+name+'_part'+str(count)+'.tif', img2)
            count +=1
    del i,j, img2, count
