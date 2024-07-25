# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 20:03:34 2022

@author: marti
"""

import numpy as np

def make_squares_from_patches(patch_list, num_patch, window_size):
    num_block = int(np.floor(np.sqrt(num_patch)))  #number of square blocks to have same H and W
    square_list = []
    W = window_size[0]
    for r in range(len(patch_list)):   #for every resolution
        for i in range(num_block):
            col = patch_list[r][i*num_block*W:(i+1)*num_block*W,:,:]
            row = np.rot90(col)
            for j in range(num_block):
                row[:, j*W:(j+1)*W,:] = np.rot90(row[:, j*W:(j+1)*W,:], -1)
            if i == 0:
                res = row
            else:
                res = np.concatenate([res, row], axis=0)   #stacking rows
        square_list.append(res)
        W = int(W/2)

    return square_list

def create_quad_tree(img_activations, labels):
    imgQuadTree = []  # list of images for each resolution layer (Quadtree)
    labelQuadTree = []  # list of classification maps for each resolution layer (Quadtree)
    
    for i in range(len(img_activations)):
        imgQuadTree.append(img_activations[i])
        labelQuadTree.append(labels[i])
    return imgQuadTree, labelQuadTree