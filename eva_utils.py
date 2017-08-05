import skimage
import skimage.io
import skimage.transform
import numpy as np

import tensorflow as tf

import math
import h5py
import os


# returns image of shape [224, 224, 3]
# [height, width, depth]
def load_image(path):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224))
    return resized_img


def load_image2(path, height=None, width=None):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    if height is not None and width is not None:
        ny = height
        nx = width
    elif height is not None:
        ny = height
        nx = img.shape[1] * ny / img.shape[0]
    elif width is not None:
        nx = width
        ny = img.shape[0] * nx / img.shape[1]
    else:
        ny = img.shape[0]
        nx = img.shape[1]
    return skimage.transform.resize(img, (ny, nx))


def evaluate(sess, model, batch_size, h5File, qList, dbList, numRecall):
    fH5 = h5py.File(h5File, 'r+')
    distMat = fH5['distance_matrix']
    numQ = len(qList)
    numDB = len(dbList)
    descriptorQ = np.zeros((numQ, 32768))
    descriptorDB = np.zeros((numDB, 32768))

    batch = np.zeros((batch_size, 224, 224, 3))
    single = np.zeros((1, 224, 224, 3))

    numBatchQ = int(math.floor(len(qList) / batch_size))
    for i in range(numBatchQ):
        if i % 10 == 0:
            print("query image forward progress: %s\n" % (float(i) / numBatchQ))
        for j in range(batch_size):
            ID = qList[i * batch_size + j]
            batch[j, :] = fH5["%s/imageData" % ID]
        descriptorQ[(i * batch_size) : (i * batch_size + batch_size), :] = sess.run(model.vlad_output, feed_dict = {'query_image:0': batch, 'train_mode:0' : False})
    for i in range(batch_size * numBatchQ, len(qList)):
        single[0, :] = fH5["%s/imageData" % qList[i]]
        descriptorQ[i, :] = sess.run(model.vlad_output, feed_dict = {'query_image:0': single, 'train_mode:0' : False})

    numBatchDB = int(math.floor(len(dbList) / batch_size))
    for i in range(numBatchDB):
        if i % 10 == 0:
            print("database image forward progress: %s\n" % (float(i) / numBatchDB))
        for j in range(batch_size):
            ID = dbList[i * batch_size + j]
            batch[j, :] = fH5["%s/imageData" % ID]
        descriptorDB[(i * batch_size) : (i * batch_size + batch_size), :] = sess.run(model.vlad_output, feed_dict = {'query_image:0': batch, 'train_mode:0' : False})
    for i in range(batch_size * numBatchDB, len(dbList)):
        single[0, :] = fH5["%s/imageData" % dbList[i]]
        descriptorDB[i, :] = sess.run(model.vlad_output, feed_dict = {'query_image:0': single, 'train_mode:0' : False})

    # compute mutual L2 distance between query and database
    A = np.dot(descriptorQ, descriptorDB.transpose())
    B = np.linalg.norm(descriptorQ, axis = 1, keepdims = True) ** 2
    C = np.linalg.norm(descriptorDB, axis = 1) ** 2
    L2_distance = np.sqrt(B + C - 2 * A)

    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    accuracy1 = 0
    accuracy2 = 0
    accuracy3 = 0
    accuracy4 = 0
    for i in range(numQ):
        if i % 20 == 0:
            print("current accuracy: %.4f   %.4f    %.4f    %.4f   evaluation progress: %.4f" % (accuracy1, accuracy2, accuracy3, accuracy4, (float(i) / numQ)))
        indices1 = np.argsort(L2_distance[i, :])[:1]
        indices2 = np.argsort(L2_distance[i, :])[:5]
        print(indices2)
        indices3 = np.argsort(L2_distance[i, :])[:10]
        indices4 = np.argsort(L2_distance[i, :])[:20]
        for j in indices1:
            if distMat[i, j] < 25:
                count1 += 1
                break
        for j in indices2:
            if distMat[i, j] < 25:
                count2 += 1
                break
        for j in indices3:
            if distMat[i, j] < 25:
                count3 += 1
                break
        for j in indices4:
            if distMat[i, j] < 25:
                count4 += 1
                break
        accuracy1 = float(count1) / (i + 1)
        accuracy2 = float(count2) / (i + 1)
        accuracy3 = float(count3) / (i + 1)
        accuracy4 = float(count4) / (i + 1)
    fH5.close()
    print("Done!\n")
    return
