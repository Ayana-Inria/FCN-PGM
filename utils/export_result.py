# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 23:09:47 2022

@author: mpastori
"""
import numpy as np
import json
import pathlib
import os
import datetime
import struct
from math import isnan
from sklearn.metrics import confusion_matrix, cohen_kappa_score


def initialize_experiment(name):
    """It retrieves the information to run the experiment and to deal with the dataset.
    The experiment asd the dataset are described in the json files placed respective in the
    experiments and the datasets folders.
    """
    # get experiment specification
    with open('experiments/' + str(name) + '.json') as f:
        experiment = json.load(f)
    # get dataset information specification
    with open('datasets/' + str(experiment["dataset"]["name"]) + '.json') as f:
        dataset = json.load(f)

    return experiment, dataset


def get_variables(experiment, dataset):
    """
    It retrieves a tuple of variables that are used to run correctly the experiment.
    :return: R -> num of resolutions
    :return: W -> width of the biggest image
    :return: H -> height of the biggest image
    :return: C -> num od classes
    :return: _method -> method to estimate the statistics
    """
    #data_used = get_small_dataset_str(experiment)

    R = dataset["R"]
    W = dataset["data"]["W"]
    H = dataset["data"]["H"]
    C = dataset["data"]["C"]
    _method = experiment["statistic_estimation"]["method"]
    net_type = experiment["neural_architecture"]["net"]

    return R, W, H, C, _method, net_type


def get_small_dataset_str(experiment):
    """It returns a string to treat the small dataset version, if any.
    """
    if experiment["dataset"]["small"]:
        data_used = 'small_data'
    else:
        data_used = 'data'
    return data_used

def set_output_location(experiment_name, run_path, net_type):
    #output_path = 'output/' + str(experiment_name)
    output_path = run_path / str(experiment_name)
    # check if directory for output exist, if not creates it
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    # write in file the starting time
    print('>>>Starting the experiment: ', experiment_name)
    f = open(output_path / 'result.txt', "a+")
    f.write('>>>Starting the experiment: ' + experiment_name + ', net used: ' + net_type + '\n')
    f.write('Results at time: ' + str(datetime.datetime.now()) + '\n\r')
    f.close()  # close file
    return output_path

#Store graylevel in a file, numByte byte for graylevel, mode:bsq
def store(output_image, filePath, numByte):
    print('storing image in -> "' , filePath, '"')
    if numByte == 1:
        with open(filePath, 'wb') as outputFile: #open as write string
            for p in np.nditer(output_image): # loop every element of array in read only mode
                outputFile.write(int(p).to_bytes(1, byteorder='little')) #write value
    elif numByte == 2 or numByte == 4 or numByte == 8: #expect that pixel are float (16, 32, 64bit)
        if numByte == 2:
            fmt = 'h'
        if numByte == 4:
            fmt = 'f'
        if numByte == 8:
            fmt = 'd'
        with open(filePath, 'wb') as outputFile: #open as write string
            for p in np.nditer(output_image): # loop every element of array in read only mode
                bArray = bytearray(struct.pack(fmt, p)) #gives array of fmt byte
                for bArrayElement in bArray:
                    outputFile.write(bArrayElement.to_bytes(1, byteorder='little')) #write value
    else:
        print('image.store(), unsupported option')

    '''
    if numByte == 8: #expect that pixel are float (64bit)
    with open(filePath, 'wb') as outputFile: #open as write string
      for p in np.nditer(self.pixel): # loop every element of array in read only mode
        bArray = bytearray(struct.pack("d", p)) #gives array of 8 byte
        for bArrayElement in bArray:
          outputFile.write(bArrayElement.to_bytes(1, byteorder='little')) #write value
    elif numByte == 4: #expect that pixel are float (32bit)
    with open(filePath, 'wb') as outputFile: #open as write string
      for p in np.nditer(self.pixel): # loop every element of array in read only mode
        bArray = bytearray(struct.pack("f", p)) #gives array of 4 byte
        for bArrayElement in bArray:
          outputFile.write(bArrayElement.to_bytes(1, byteorder='little')) #write value
    elif numByte == 2: #expect that pixel are float (16bit)
    with open(filePath, 'wb') as outputFile: #open as write string
      for p in np.nditer(self.pixel): # loop every element of array in read only mode
        bArray = bytearray(struct.pack("h", p)) #gives array of 4 byte
        for bArrayElement in bArray:
          outputFile.write(bArrayElement.to_bytes(1, byteorder='little')) #write value
    '''


def img_from_data(prob, H, W, R, output_path, img_name, _from='prob', export=False):
    """ Create images of predicted labels on every resolution
      prob -> probability of each class, format expected [R][C][H][W]
      label -> label already given
    """
    _result_img = []  # list of result images
    height = H
    width = W
    if _from == 'prob':
        C = len(prob)

    for r in range(R):
        #imageResult = Image(width, height, pixelType='uint8')  # result of classification image
        imageResult = np.zeros((height, width, 1), dtype='uint8')
        if _from == 'prob':
          # label choosed is argmax on xs of partialPost
          # obs:
          #	done in a numpy way would be
          #	imageResult = np.argmax(partialPost, axis=1)
            for h in range(height):
                for w in range(width):
                    label = 0
                    maxValue = 0
                    for xs in range(C):
                        if isnan(prob[r][xs][h][w]):
                            print('WARNING: nan found')
                        else:
                            if prob[r][xs][h][w] > maxValue:
                                label = xs #+ 1   #potresti dover cambiare questo
                                maxValue = prob[r][xs][h][w]
                    # store label
                    #imageResult.pixel[0][h][w] = label
                    imageResult[h][w][0] = label

                    # set to background the first row and first column
                    """
                    if h == 0 or w == 0 or h == height - 1 or w == width - 1:
                        imageResult[h][w][0] = 0
                    """
        elif _from == 'label':
            PSingleRes = prob[r].reshape(height, width)  # reshape the array to have the right dimension
            # fill result image and compute accuracy
            for h in range(0, height):
                for w in range(0, width):
                    #imageResult.pixel[0][h][w] = PSingleRes[h][w]
                    imageResult[h][w][0] = PSingleRes[h][w]
        else:
            print('ERROR: unsupported option _from: ', _from)

        #putting bands first
        imageResult = np.transpose(imageResult, [2, 0, 1])
      
        if export:

            #imageResult.store(output_path + '/' + str(img_name) + '_r_' + str(r) + '.raw', 1)
            store(imageResult, output_path / str(str(img_name) + '_r_' + str(r) + '.raw'), 1)

            # --- taking care of the .hdr file
            # if .hdr file not exist create it
            if not os.path.isfile(output_path / str(str(img_name) + '_r_' + str(r) + '.hdr')):
                f = open(output_path / str(str(img_name) + '_r_' + str(r) + '.hdr'), "a+")
                hdr = """ENVI
                samples = {width}
                lines   = {height}
                bands   = 1
                header offset = 0
                file type = ENVI Standard
                data type = 1
                interleave = bsq""".format(width=width, height=height)
                f.write(hdr)
                f.close()  # close file
        _result_img.append(imageResult)
        # update sizes
        width = int(width / 2)
        height = int(height / 2)

    return _result_img

def create_partial_post_folder(experiment, dataset, output_path, _method):
    add_str = ''  # additional string to allow different dataset's versions
    #if experiment["dataset"]["small"]:
    #    add_str = '/small'
    #partial_folder = 'datasets/' + dataset["name"] + '/' + add_str + '/partial_post'
    partial_folder = output_path / 'partial_post'
    _file = partial_folder / str(str(_method) + '_bott-up.npy')
    return partial_folder, _file


def note_down_computation_time(output_path, start_time):
    # write computation time in result file
    f = open(output_path + '/result.txt', "a+")
    f.write('computing time: ' + str(time.time() - start_time) + ' sec' + '\n\r\n\r')
    f.close()  # close file
    
    
def export_results(resultQuadTree, testQuadTree, run_path, experiment_name,
                   confusionMat=None,
                   prodAccuracy=None,
                   averageAccuracy=None,
                   kappaCoeff=None,
                   title=''):
    """ Export classification results given images of result and ground-truth
    """
    H = testQuadTree[0].shape[0]
    W = testQuadTree[0].shape[1]
    C = len(np.unique(testQuadTree[0]))
    R = len(testQuadTree)
    confusionMatrixList = []
    cohenKappaScoreList = []
    producerAccuracyList = []
    userAccuracyList = []
    labels = np.arange(C)  # [1, 2, ... , C]

    output_path = run_path / str(str(experiment_name) + '/result.txt')

    # get accuracies and confusion matrices
    accuracy = []  # list of accuracy for each resolution
    countSuccess = 0
    countTestPixel = 0
    width = W
    height = H

    for r in range(R):
        # label choosed is argmax on xs
        for h in range(height):
            for w in range(width):
                # do computation for getting accuracy
                if testQuadTree[r][h][w] != 6:
                    countTestPixel += 1
                    if testQuadTree[r][h][w][0] == resultQuadTree[r][0][h][w]: #resultQuadTree[r][0][h][w]:
                        countSuccess += 1
        accuracy.append(countSuccess / countTestPixel)
        # reset sizes
        height = int(height / 2)
        width = int(width / 2)
        # reset counters
        countSuccess = 0
        countTestPixel = 0

        y_true = testQuadTree[r].ravel()
        y_pred = resultQuadTree[r].ravel()
        if confusionMat is not None:
            confusionMatrixList.append(confusion_matrix(y_true, y_pred, labels))
        if kappaCoeff is not None:
            cohenKappaScoreList.append(cohen_kappa_score(y_true, y_pred, labels))

    if prodAccuracy is not None:
        # compute producers accuracies
        for r in range(R):
            singleResProducerAccuracies = []
            singleResUserAccuracies = []

            for c in range(C):
                # print(confusionMatrixList[r][c][c])
                singleResProducerAccuracies.append(confusionMatrixList[r][c][c])
                singleResUserAccuracies.append(confusionMatrixList[r][c][c])

            for c1 in range(C):
                countP = 0
                countU = 0
                for c2 in range(C):
                    countP += confusionMatrixList[r][c1][c2]
                    countU += confusionMatrixList[r][c2][c1]
                    # count += confusionMatrixList[r][c2][c1] #for user accuracies
                singleResProducerAccuracies[c1] /= countP
                singleResUserAccuracies[c1] /= countU
                
            userAccuracyList.append(singleResUserAccuracies)
            producerAccuracyList.append(singleResProducerAccuracies)

    if averageAccuracy is not None:
        averageAccuracy = []
        averageUserAccuracy = []
        for r in range(R):
            _sum = 0
            _sumUser = 0
            for c in range(C):
                _sum += producerAccuracyList[r][c]
                _sumUser += userAccuracyList[r][c]
            _sum /= C
            _sumUser /= C
            averageAccuracy.append(_sum)
            averageUserAccuracy.append(_sumUser)

    # write accuracies in file
    #output_path = '/content/drive/My Drive/Colab Notebooks/tmp/result.txt'
    f = open(output_path, "a+")
    f.write(title + '\n\r')
    f.write('overall accuracy\n\r')
    for r in reversed(range(R)):
        print('overall accuracy in r = ', r, ' -> ', accuracy[r])
        f.write('r = ' + str(r) + ' -> ' + str(accuracy[r]) + '\n')
    f.write('\n')
    # close file
    f.close()

    if prodAccuracy is not None:
        # write producer accuracies in file
        f = open(output_path, "a+")
        f.write('producer accuracies\n\r')
        for r in reversed(range(R)):
            f.write('r = ' + str(r) + '\n')
            for c in range(C):
                # print('overall accuracy in r = ', r, ' -> ', accuracy[r])
                f.write('c = ' + str(c + 1) + ' -> ' + str(producerAccuracyList[r][c]) + '\n')
        f.write('user accuracies\n\r')
        for r in reversed(range(R)):
            f.write('r = ' + str(r) + '\n')
            for c in range(C):
                # print('overall accuracy in r = ', r, ' -> ', accuracy[r])
                f.write('c = ' + str(c + 1) + ' -> ' + str(userAccuracyList[r][c]) + '\n')
      
        f.write('\n')
        # close file
        f.close()

    if averageAccuracy is not None:
        # write average accuracies in file
        f = open(output_path, "a+")
        f.write('average producer accuracies\n\r')
        for r in reversed(range(R)):
            print('average producer accuracy in r = ', r, ' -> ', averageAccuracy[r])
            f.write('r = ' + str(r) + ' -> ' + str(averageAccuracy[r]) + '\n')
        f.write('average user accuracies\n\r')
        for r in reversed(range(R)):
            print('average user accuracy in r = ', r, ' -> ', averageUserAccuracy[r])
            f.write('r = ' + str(r) + ' -> ' + str(averageUserAccuracy[r]) + '\n')
        f.write('\n')
        # close file
        f.close()

    if kappaCoeff is not None:
        # write cohen kappa score in file
        f = open(output_path, "a+")
        f.write('cohen kappa scores\n\r')
        for r in reversed(range(R)):
            # print('overall accuracy in r = ', r, ' -> ', accuracy[r])
            f.write('r = ' + str(r) + ' -> ' + str(cohenKappaScoreList[r]) + '\n')
        f.write('\n')
        # close file
        f.close()

    if confusionMat is not None:
        # write confusion matrix in file
        f = open(output_path, "a+")
        f.write('confusion matrices\n\r')
        for r in reversed(range(R)):
            mat = np.matrix(confusionMatrixList[r])
            # with open('outfile.txt','wb') as f:
            for line in mat:
                np.savetxt(f, line, fmt='%i', delimiter='    ')
            f.write('\n')
        # close file
        f.close()

    # add a blank line at the bottom
    f = open(output_path, "a+")
    f.write('\n')
    f.close()
