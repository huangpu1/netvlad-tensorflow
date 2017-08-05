import os
import math
import random
import numpy as np
import scipy.io as sio
import h5py
import thread
from multiprocessing import Pool

import train_utils

def get_List(mat_path):
    boxes = sio.loadmat(mat_path)["dbStruct"]
    qList = [str(x[0][0]) for x in boxes["qImageFns"][0, 0]]
    dbList = [str(x[0][0]) for x in boxes["dbImageFns"][0, 0] if not (x in boxes["qImageFns"][0, 0])]

    return qList, dbList

def compute_dist(mat_path, h5_file):
    boxes = sio.loadmat(mat_path)["dbStruct"]

    qList = [str(x[0][0]) for x in boxes["qImageFns"][0, 0]]
    dbList = [str(x[0][0]) for x in boxes["dbImageFns"][0, 0] if (not x in boxes["qImageFns"][0, 0])]

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

    return


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

    for i in range(len(qList)):
        if i % 50 == 0:
            print("computing idx of image %s" % i)
        ID = qList[i]
        if not ID in fH5:
            fH5.create_group(ID)
        if not "positives" in fH5[ID]:
            fH5.create_dataset("%s/positives" % ID, (40, ), dtype = 'i')
        if not "negatives" in fH5[ID]:
            fH5.create_dataset("%s/negatives" % ID, (20, ), dtype = 'i')
        if not "potential_negatives" in fH5[ID]:
            fH5.create_dataset("%s/potential_negatives" % ID, (1000, ), dtype = 'i')

        pos = fH5["%s/positives" % ID]
        neg = fH5["%s/negatives" % ID]
        pneg = fH5["%s/potential_negatives" % ID]

        pos[:] = np.argsort(distMat[i, :])[:40]

        indices = np.where(distMat[i, :] >= 25)[0]
        pneg[:] = random.sample(indices, 1000)
        neg[:] = random.sample(pneg, 20)

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

    def single_load(idList, idxS, idxE):
        for i in range(idxS, idxE):
            if i % 100 == 0:
                print("image %s loaded" % i)
            ID = idList[i]
            if not "imageData" in fH5[ID]:
                fH5.create_dataset("%s/imageData" % ID, (224, 224, 3), dtype = 'f')
            fH5["%s/imageData" % ID][:] = train_utils.load_image(("%s/%s" % (data_dir, idList[i])))
        return

    single_load(qList, 0, len(qList))

    single_load(dbList, 0, len(dbList))
    
    fH5.close()
    print("Done!\n")
    return