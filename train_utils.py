import skimage
import skimage.io
import skimage.transform
import numpy as np

import tensorflow as tf

import math
import h5py
import os
import random
import time


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

def index_update(sess, model, batch_size, h5File, qList, dbList):
    print("Updating positives and negatives...\n")
    fH5 = h5py.File(h5File, 'r+')
    descriptorQ = np.zeros((len(qList), 32768))
    descriptorDB = np.zeros((len(dbList)), 32768)
    L2_distance = np.zeros((len(qList), len(dbList)))

    batch = np.zeros((batch_size, 224, 224, 3))
    single = np.zeros((1, 224, 224, 3))

    numBatchQ = int(math.floor(len(qList) / batch_size))
    for i in range(numBatchQ):
        for j in range(batch_size):
            ID = qList[i * batch_size + j]
            batch[j, :] = fH5["%s/imageData" % ID]
        descriptorQ[(i * batch_size) : (i * batch_size + batch_size), :] = sess.run(model.vlad_output, feed_dict = {'query_image:0': batch, 'train_mode:0' : False})
    for i in range(batch_size * numBatchQ, len(qList)):
        single[0, :] = fH5["%s/imageData" % qList[i]]
        descriptorQ[i, :] = sess.run(model.vlad_output, feed_dict = {'query_image:0': single, 'train_mode:0' : False})

    numBatchDB = int(math.floor(len(dbList) / batch_size))
    for i in range(numBatchDB):
        for j in range(batch_size):
            ID = dbList[i * batch_size + j]
            batch[j, :] = fH5["%s/imageData" % ID]
        descriptorDB[(i * batch_size) : (i * batch_size + batch_size), :] = sess.run(model.vlad_output, feed_dict = {'query_image:0': batch, 'train_mode:0' : False})
    for i in range(batch_size * numBatchDB, len(dbList)):
        single[0, :] = fH5["%s/imageData" % dbList[i]]
        descriptorDB[i, :] = sess.run(model.vlad_output, feed_dict = {'query_image:0': single, 'train_mode:0' : False})

        
    for i, ID in enumerate(qList):
        neg = fH5["%s/negatives" % ID]
        L2_dist = {}
        pneg = fH5["%s/potential_negatives" % ID]
        for j in pneg:
            L2_dist[str(j)] = np.linalg.norm(descriptorQ[i, :] - descriptorDB[j, :])
        L2Sorted = sorted(L2_dist.items(), key = lambda e:e[1])

        for k in range(10):
            neg[10 + k] = neg[k]
            neg[k] = int(L2Sorted[k][0])
    fH5.close()
    print("Done!\n")
    return

def next_batch(sess, model, batch_size, h5File, qList, dbList):
    numQ = len(qList)
    numBatch = math.floor(numQ / batch_size)
    fH5 = h5py.File(h5File, 'r+')
    idx = random.randint(0, numQ - 1)
    for i in range(int(numBatch + 1)):
        z = i / numBatch
        x = np.zeros((batch_size, 224, 224, 3))
        labels = np.zeros((batch_size, 32768, 30))
        batch = np.zeros((batch_size * 30, 224, 224, 3))
        for t in range(batch_size):
            idx = idx % numQ
            
            x[t, :] = fH5["%s/imageData" % qList[idx]]
            pos = fH5["%s/positives" % qList[idx]]
            neg = fH5["%s/negatives" % qList[idx]]

            for j in range(10):
                batch[(batch_size * j + t), :] = fH5["%s/imageData" % dbList[pos[j]]]
            for k in range(20):
                batch[(batch_size * k + 10 * batch_size + t), :] = fH5["%s/imageData" % dbList[neg[k]]]
            idx += 1

        output = sess.run(model.vlad_output, feed_dict = {'query_image:0': batch, 'train_mode:0' : False})
        for j in range(30):
            labels[:, :, j] = output[(batch_size * j) : (batch_size * j + batch_size), :]

        yield x, labels, z
    fH5.close()
    return