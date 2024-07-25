# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 17:37:46 2022

@author: marti
"""

import time
import numpy as np
import random
import itertools
from sklearn.metrics import confusion_matrix
import json
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt




############# Utils ###############

def note_down_computational_time(start_time):
    return print(time.time() - start_time)


def get_random_pos(img, window_shape):
    """ Extract of 2D random patch of shape window_shape in the image """
    w, h = window_shape
    W, H = img.shape[-2:]
    x1 = random.randint(0, W - w - 1)
    x2 = x1 + w
    y1 = random.randint(0, H - h - 1)
    y2 = y1 + h
    return x1, x2, y1, y2


def accuracy(input, target):
    return 100 * float(np.count_nonzero(input == target)) / target.size

def sliding_window(top, step=10, window_size=(20,20)):
    """ Slide a window_shape window across the image with a stride of step """
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            yield x, y, window_size[0], window_size[1]
            
def count_sliding_window(top, step=10, window_size=(20,20)):
    """ Count the number of windows in an image """
    c = 0
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            c += 1
    return c

def grouper(n, iterable):
    """ Browse an iterator by chunk of n elements """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk

def metrics(predictions, gts, label_values):
    cm = confusion_matrix(
            gts,
            predictions,
            range(len(label_values)))
    
    print("Confusion matrix :")
    print(cm)
    
    print("---")
    
    # Compute global accuracy
    total = sum(sum(cm))
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)
    print("{} pixels processed".format(total))
    print("Total accuracy : {}%".format(accuracy))
    
    print("---")
    
    # Compute F1 score
    F1Score = np.zeros(len(label_values))
    for i in range(len(label_values)):
        try:
            F1Score[i] = 2. * cm[i,i] / (np.sum(cm[i,:]) + np.sum(cm[:,i]))
        except:
            # Ignore exception if there is no element in class i for test set
            pass
    print("F1Score :")
    for l_id, score in enumerate(F1Score):
        print("{}: {}".format(label_values[l_id], score))

    print("---")
        
    # Compute kappa coefficient
    total = np.sum(cm)
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total*total)
    kappa = (pa - pe) / (1 - pe);
    print("Kappa: " + str(kappa))
    return accuracy

def normalize_cm(list_el, num_classes):
    normalized_list = []
    count = np.asarray(list_el).sum(axis=1)
    for i in range(num_classes):
        normalized_list.append(list_el[i] / count[i])
    return normalized_list

def conf_matr(resultQuadTree, testQuadTree):
    """ Export classification results given images of result and ground-truth
    """
    H = testQuadTree[0].shape[0]
    W = testQuadTree[0].shape[1]
    C = len(np.unique(testQuadTree[0]))
    if 6 in np.unique(testQuadTree[0]):
        C -= 1
    R = len(testQuadTree)
    confusionMatrixList = []
    labels = np.arange(C)  # [1, 2, ... , C]

    # get accuracies and confusion matrices
    accuracy = []  # list of accuracy for each resolution
    countSuccess = 0
    countTestPixel = 0
    width = W
    height = H

    for r in range(R):
        # label choosed is argmax on xs
        y_true = testQuadTree[r].ravel()
        y_pred = resultQuadTree[r].ravel()
        confusionMatrixList.append(confusion_matrix(y_true, y_pred, labels))
    return confusionMatrixList

def print_conf_matr(outQuadTree, testQuadTree, R, classes, title, name):
    cm = conf_matr(outQuadTree, testQuadTree)
    normalized_cm = normalize_cm(cm[R], len(classes))
    df_cm = pd.DataFrame(normalized_cm, index = [i for i in classes], 
                  columns = [i for i in classes])
    plt.figure(figsize = (30,25))
    plt.rcParams.update({'font.size': 22})
    plt.title(title)
    sn.heatmap(df_cm, cmap="RdPu", annot=True, fmt='.2f')
    plt.savefig(name)