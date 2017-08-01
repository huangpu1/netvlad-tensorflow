import os
import math
import numpy as np
import h5py

import val_utils


def cut_suffix(fileName):
    return fileName.split('.')[0]

def get_idList(data_dir):
    fileList = [x for x in os.listdir(data_dir) if os.path.splitext(x)[1] == '.csv']
    idList = list(map(cut_suffix, fileList))

    return idList

def compute_dist(data_dir, h5_file):
    fH5 = h5py.File(h5_file, "r+")
    
    fileList = [x for x in os.listdir(data_dir) if os.path.splitext(x)[1] == '.csv']
    imageList = [x for x in os.listdir(data_dir) if os.path.splitext(x)[1] == '.jpg']
    idList = list(map(cut_suffix, fileList))

    numImage = len(fileList)
    if not "distance_matrix" in fH5:
        fH5.create_dataset('distance_matrix', shape = (numImage, numImage), dtype = 'f')

    distMat = fH5['distance_matrix']

    for i in range(numImage):
        with open(os.path.join(data_dir, fileList[i]), 'r') as f_1:
            line_1 = f_1.read().split(',')
            x_1 = float(line_1[-2])
            y_1 = float(line_1[-1])

            for j in range(i, numImage):
                if i == j:
                    distMat[i, j] = -1
                    continue
                with open(os.path.join(data_dir, fileList[j]), 'r') as f_2:
                    line_2 = f_2.read().split(',')
                    x_2 = float(line_2[-2])
                    y_2 = float(line_2[-1])
                    distMat[i, j] = math.sqrt(pow(x_1 - x_2, 2) + pow(y_1 - y_2, 2))
                    distMat[j, i] = distMat[i, j]

    fH5.close()

    return idList


def h5_initial():
    if not os.path.exists("index_dir"):
        os.mkdir("index_dir")
    if not os.path.exists("index_dir/datafile.hdf5"):
        f = h5py.File("index_dir/datafile.hdf5", 'w')
        f.close()

    return

def load_image(data_dir, h5File, idList):
    print("Loading image data...\n")
    fH5 = h5py.File(h5File, 'r+')
    for i, ID in enumerate(idList):
        if not "imageData" in fH5[ID]:
            fH5.create_dataset("%s/imageData" % ID, (224, 224, 3), dtype = 'f')
        fH5["%s/imageData" % ID][:] = utils.load_image(("%s/%s.jpg" % (data_dir, idList[i])))
    fH5.close()
    print("Done!\n")

    return

def index_initial(h5File, idList):
    fH5 = h5py.File(h5File, 'r+')
    distMat = fH5['distance_matrix']
    for i, ID in enumerate(idList):
        if not ID in fH5:
            fH5.create_group(ID)
        if not "positives" in fH5[ID]:
            fH5.create_dataset("%s/positives" % ID, (10, ), dtype = 'i')
        if not "negatives" in fH5[ID]:
            fH5.create_dataset("%s/negatives" % ID, (20, ), dtype = 'i')
        if not "potential_negatives" in fH5[ID]:
            fH5.create_dataset("%s/potential_negatives" % ID, (300, ), dtype = 'i')
        """if not 'descriptor' in fH5[ID]:
            fH5.create_dataset("%s/descriptor" % ID, (32768, ), 'f')"""
        
        pos = fH5["%s/positives" % ID]
        neg = fH5["%s/negatives" % ID]
        pneg = fH5["%s/potential_negatives" % ID]

        posDic = {}
        negDic = {}

        for j, dist in enumerate(distMat[i, :]):
            if dist >= 0 and dist <= 10:
                posDic['%s' % j] = dist
            elif dist > 25:
                negDic['%s' % j] = dist

        posSorted = sorted(posDic.items(), key = lambda e:e[1])
        negSorted = sorted(negDic.items(), key = lambda e:e[1])

        if len(posDic) >= 10:
            for k in range(10):
                pos[k] = int(posSorted[k][0])
        else:
            for k in range(len(posSorted)):
                pos[k] = int(posSorted[k][0])
            for k in range(len(posSorted), 10):
                pos[k] = pos[k - 1]
        
        for k in range(300):
            pneg[k] = int(negSorted[k][0])

        for k in range(10):
            neg[k] = int(negSorted[k][0])
        for k in range(10, 20):
            neg[k] = neg[k - 10]

    fH5.close()

    return