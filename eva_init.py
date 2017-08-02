import os
import math
import random
import numpy as np
import scipy.io as sio
import h5py

import eva_utils

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

    return


def h5_initial(eva_h5File):
    if not os.path.exists("index"):
        os.mkdir("index")
    if not os.path.exists(eva_h5File):
        f = h5py.File(eva_h5File, 'w')
        f.close()

    return

    
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
            fH5["%s/imageData" % ID][:] = eva_utils.load_image(("%s/%s" % (data_dir, idList[i])))
        return

    single_load(qList, 0, len(qList))

    single_load(dbList, 0, len(dbList))
    
    fH5.close()
    print("Done!\n")
    return