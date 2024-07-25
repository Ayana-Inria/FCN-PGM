# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 23:09:47 2022

@author: mpastori
"""

import numpy as np
from tqdm import tqdm_notebook as tqdm
import pickle
from sklearn.ensemble import RandomForestClassifier

def ensemble_estim(imgQuadTree, mapQuadTree, save_path, train=True):
    randomForestList = []  # list, for each resolution, of list, for each class, of the site statistics
    P = []  # list, for each resolution, of predicted labels
    
    H = imgQuadTree[0].shape[0]
    W = imgQuadTree[0].shape[1]
    height = H
    width = W
    if 6 in np.unique(mapQuadTree[0]):
        C = len(np.unique(mapQuadTree[0]))-1
    else:
        C = len(np.unique(mapQuadTree[0]))
    R = len(imgQuadTree)
    B = []
    for r in range(R):
        B.append(imgQuadTree[r].shape[2])

    X = [] #for each r: all feature vectors in dataset
    Xtrain = [] #for each r: feature vector that has a label (training)
    Y = [] #for each r: label of training set

    for r in range(R):
        pixelFlat = imgQuadTree[r].ravel() #put pixel on a single row
        trainPixel = mapQuadTree[r].reshape(height,width) #reshape to throw away useless band axis
        
        # -------------------------------------------------
        #|   X is list of matrix [W*H][B]                 |
        #|   where B is the number of bands in layer r    |
        #|   s is the site index                          |
        #|   [s=0     b=0][s=0   b=1]   .   [s=0   b=B-1] | 
        #|   [s=1     b=0]                                |
        #|   .          .                                 | 
        #|   .                                            |
        #|   [s=W*H-1 b=0]   .     .    . [s=W*H-1 b=B-1] |
        # -------------------------------------------------
        X.append(np.empty([width*height, B[r]], dtype = type(pixelFlat[0]))) #matrix that will hold for each row values for a pixel in every band (feature vector)
        for i in tqdm(range(0, width*height)): #for every pixel
            X[r][i] = pixelFlat[(i*B[r]):(i*B[r])+B[r]] #store in i row of X values for each band
            
            
            # obs: here we are starting from first band and jumping W*H pixel, num of pixel per each band
        
        #find train samples
        m = trainPixel != 6 #get boolean matrix mask
        m = m.reshape(width*height,1) #to column
        numOfTrainSample = np.count_nonzero(m)
        print('train samples')
        print(numOfTrainSample)

        columnM0 = m
        for i in range (B[r]-1): #horizontally add another columnM0 column to match X size
            m = np.hstack((m,columnM0))
        Xtrain.append(X[r][m].reshape(-1, B[r])) #-1 -> infer dimension

        #create empty labels array
        Y.append(np.empty([numOfTrainSample, 1]))
        #fill train labels in Y
        count = 0
        for h in range(height):
            for w in range(width):
                if mapQuadTree[r][h][w] != 6:
                    Y[r][count, 0] = mapQuadTree[r][h][w][0]
                    count += 1
        #reduce sizes
        width = int(width/2)
        height = int(height/2)

        
    #---Statistics estimation using Random Forest
    width = W
    height = H
    #classifier = []
    for r in range(R):
        print('Random forest estimation started')
        clf = RandomForestClassifier(n_jobs=2, random_state=0, n_estimators=100)
        
        if train == True:
            #obs:fit(Xtrain, Y)
            #n_samples : number of feature vectors in training set
            #Xtrain : has dim [n_samples, n_features]
            #Y : has dim [n_samples, n_outputs]
            print("Training the model")
            clf.fit(Xtrain[r], Y[r].ravel())
            
            #save classifier
            pkl_filename = 'clf_{}.pkl'.format(r)
            with open(save_path / pkl_filename, 'wb') as file:
                pickle.dump(clf, file)
        
        else:
            pkl_filename = save_path / 'clf_{}.pkl'.format(r)
            with open(pkl_filename, 'rb') as file:
                clf = pickle.load(file)
        
        #find predicted labels for outputting classification result
        # Apply the classifier we trained to the training set
        print("Predicting...")
        P.append(clf.predict(X[r]))
        
        outSingleRes = [] #list for each class of the statistics for each site
        
        # predicted probabilities p has dim [n_samples, n_classes]
        predProb = clf.predict_proba(X[r])
        


        
        #fill statistics data structure
        for xs in range(0, C):
            #print('xs ->', xs)
            outSingleRes.append(np.zeros((height, width, 1)))
            
            #print(len(outSingleRes))
            #count = 0
            for h in range(0, height):
                #print('h ->', h)
                for w in range(0, width):
                    outSingleRes[xs][h][w][0] = predProb[w+h*width][xs]
                    
                    #count += 1
        randomForestList.append(outSingleRes)
        

        #reduce sizes
        width = int(width/2)
        height = int(height/2)

    return P, randomForestList