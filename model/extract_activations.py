# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 13:04:51 2022

@author: mpastori
"""

import numpy as np
from utils.utils_dataset import *
from utils.utils import *
from torch.autograd import Variable
import torch
import cv2


def extract_activation(net, train_image, train_label, num_patch, window_size):
    train_activations = []
    train_labels = []
    W = window_size[0]
    net.eval()

    l = int(np.sqrt(num_patch))
    for img, gt in zip(train_image, train_label):
        
        gt = conn_comp(convert_from_color(gt), disk(8))
        total = count_sliding_window(img, step=256, window_size=window_size)


        for coords in grouper(total, sliding_window(img, step=256, window_size=window_size)):
            for i in range(len(coords)): #take a square portion of the image made up by 256x256 contingent patches
                  for q in range(l):  # number of lines
                    if coords[i][:2] == ((q * 256)+256,0):  # position in the original image
                        image_patches = [np.copy(img[x:x+w, y:y+h]).transpose((2,0,1)) for x,y,w,h in coords[i:i+l]]
                        label_ptc = [np.copy(gt[x:x+w, y:y+h]) for x,y,w,h in coords[i:i+l]]
                        image_patches = np.asarray(image_patches)
                        if q == 0:
                            image_ptc = image_patches
                            label_patches = label_ptc
                        else:
                            image_ptc = np.concatenate([image_ptc, image_patches], axis=0)
                            label_patches = np.concatenate([label_patches, label_ptc], axis=0)
            image_patches = Variable(torch.from_numpy(image_ptc).cuda(), volatile=True)


            # Do the inference
            train_outs, activations = net(image_patches)
            train_outs = train_outs.data.cpu().numpy()
            image_patches = image_patches.data.cpu().numpy()
            image_patches = np.concatenate((image_patches, train_outs), axis=1)
            del(train_outs)
            
    for i in range(len(activations)):
        activations[i] = activations[i].data.cpu().numpy()

    train_img = np.transpose(image_patches, [0,2,3,1])
    train_gt_img = label_patches

    train_activations.append(np.reshape(train_img, (num_patch*W, W, -1)))
    train_labels.append(np.reshape(train_gt_img, (num_patch*W, W, 1)))
    print(np.unique(train_labels[0]))

    W = int(W/2)
    for i in range(1, len(activations)+1):
        tmp = activations[-i]
        train_activations.append(np.transpose(tmp, [0,2,3,1]))
        train_activations[i] = np.reshape(train_activations[i], (num_patch*W, W, train_activations[i].shape[3]))
        train_labels.append(np.expand_dims(cv2.resize(train_labels[0], dsize=(W, num_patch*W), interpolation=cv2.INTER_NEAREST), axis=2))
        W = int(W/2)

    del(activations)

    return train_activations, train_labels