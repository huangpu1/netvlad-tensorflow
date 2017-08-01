import os
import math
import numpy as np
import scipy.io as sio
import h5py
import thread
from multiprocessing import Pool

import train_utils


def cut_extension(fileName):
    name, ext = os.path.splitext(fileName)
    return name

def get_List(mat_path):
    boxes = sio.loadmat(mat_path)["dbStruct"]
    qList = [str(x[0][0]) for x in boxes["qImageFns"][0, 0]]
    dbList = [str(x[0][0]) for x in boxes["dbImageFns"][0, 0]]

    return qList, dbList

def compute_dist(mat_path, h5_file):
    boxes = sio.loadmat(mat_path)["dbStruct"]

    qList = [str(x[0][0]) for x in boxes["qImageFns"][0, 0]]
    dbList = [str(x[0][0]) for x in boxes["dbImageFns"][0, 0]]

    qLoc = boxes["utmQ"][0, 0].transpose()
    dbLoc = boxes["utmDb"][0, 0].transpose()

    fH5 = h5py.File(h5_file, "r+")
    numQ = len(qList)
    numDB = len(dbList)
    if not "distance_matrix" in fH5:
        fH5.create_dataset('distance_matrix', shape = (numQ, numDB), dtype = 'f')
    distMat = fH5['distance_matrix']
    for i in range(numQ):
        distMat[i, :] = np.linalg.norm(qLoc[i, :] - dbLoc, axis = 1)
    fH5.close()

    return qList, dbList


def h5_initial(train_h5File):
    if not os.path.exists("index"):
        os.mkdir("index")
    if not os.path.exists(train_h5File):
        f = h5py.File(train_h5File, 'w')
        f.close()

    return

def index_initial(h5File, qList, dbList):
    fH5 = h5py.File(h5File, 'r+')
    distMat = fH5['distance_matrix']

    def single_compute(idxS, idxE):
        for i in range(idxS, idxE):
            if i % 10 == 0:
                print("computing idx of image %s" % i)
            ID = qList[i]
            if not ID in fH5:
                fH5.create_group(ID)
            if not "positives" in fH5[ID]:
                fH5.create_dataset("%s/positives" % ID, (10, ), dtype = 'i')
            if not "negatives" in fH5[ID]:
                fH5.create_dataset("%s/negatives" % ID, (20, ), dtype = 'i')
            if not "potential_negatives" in fH5[ID]:
                fH5.create_dataset("%s/potential_negatives" % ID, (300, ), dtype = 'i')

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
        return

    numProc = 8
    qBlock = len(qList) / numProc

    p = Pool(numProc)
    for i in range(numProc):
        idxS = qBlock * i
        idxE = (i + 1) * qBlock
        # thread.start_new_thread(single_compute, (idxS, idxE, ))
        p.apply_async(single_compute, (idxS, idxE, ))
    p.close()
    p.join()

    single_compute(numProc * qBlock, len(qList))

    for i, ID in enumerate(dbList):
        if not ID in fH5:
            fH5.create_group(ID)

    fH5.close()

    return

"""def load_image(data_dir, h5File, qList, dbList):
    print("Loading query image data...\n")
    fH5 = h5py.File(h5File, 'r+')
    for i, ID in enumerate(qList):
        if i % 10 == 0:
            print("progress %.4f\n" % (float(i) / len(qList) * 100))
        if not "imageData" in fH5[ID]:
            fH5.create_dataset("%s/imageData" % ID, (224, 224, 3), dtype = 'f')
        fH5["%s/imageData" % ID][:] = train_utils.load_image(("%s/%s" % (data_dir, qList[i])))

    print("Loading database image data...\n")
    for i, ID in enumerate(dbList):
        if i % 10 == 0:
            print("progress %.4f%%\n" % (float(i) / len(dbList) * 100))
        if not "imageData" in fH5[ID]:
            fH5.create_dataset("%s/imageData" % ID, (224, 224, 3), dtype = 'f')
        fH5["%s/imageData" % ID][:] = train_utils.load_image(("%s/%s" % (data_dir, dbList[i])))
    fH5.close()
    print("Done!\n")

    return"""

    
def multipro_load_image(data_dir, h5File, qList, dbList):
    print("loading query image data...\n")
    fH5 = h5py.File(h5File, 'r+')

    numProc = 8
    qBlock = len(qList) / (numProc)
    dbBlock = len(dbList) / (numProc)

    def single_load(idList, idxS, idxE):
        for i in range(idxS, idxE):
            if i % 100 == 0:
                print("image %s loaded" % i)
            ID = idList[i]
            if not "imageData" in fH5[ID]:
                fH5.create_dataset("%s/imageData" % ID, (224, 224, 3), dtype = 'f')
            fH5["%s/imageData" % ID][:] = train_utils.load_image(("%s/%s" % (data_dir, idList[i])))
        return

    p = Pool(numProc)
    for i in range(numProc):
        idxS = i * qBlock
        idxE = (i + 1) * qBlock
        # thread.start_new_thread(single_load, (qList, idxS, idxE, ))
        p.apply_async(single_load, (qList, idxS, idxE))
    p.close()
    p.join()
    single_load(qList, numProc * qBlock, len(qList))

    p = Pool(numProc)
    for i in range(numProc):
        idxS = i * dbBlock
        idxE = (i + 1) * dbBlock
        # thread.start_new_thread(single_load, (dbList, idxS, idxE, ))
        p.apply_async(single_load, (dbList, idxS, idxE))
    p.close()
    p.join()
    single_load(dbList, numProc * dbBlock, len(dbList))
    
    fH5.close()
    print("Done!\n")
    return