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

def index_update(sess, model, data_dir, h5File, idList):
    fH5 = h5py.File(h5File, 'r+')
    descriptor = np.zeros((len(idList), 32768))
    L2_distance = np.zeros((len(idList), len(idList)))

    for i, ID in enumerate(idList):
        img = load_image("%s/%s.jpg" % (data_dir, ID))
        batch = img.reshape((1, 224, 224, 3))
        descriptor[i,:] = tf.reshape(sess.run(model.vlad_output, feed_dict = {'query_image:0': batch, 'train_mode:0' : False}), [32768])

    """for i in range(len(idList)):
        for j in range(len(idList)):
            L2_distance[i, j] = tf.norm(tf.subtract(descriptor[i, :], descriptor[j,:]))"""
        
    for i, ID in enumerate(idList):
        neg = fH5["%s/negatives" % ID]
        L2_dist = {}
        pneg = fH5["%s/potential_negatives" % ID]
        for j in pneg:
            L2_dist[str(i)] = tf.norm(tf.subtract(descriptor[i, :], descriptor[j, :]))
        L2Sorted = sorted(L2_dist.items(), key = lambda e:e[1])

        for k in range(10):
            neg[10 + k] = neg[k]
            neg[k] = int(L2Sorted[k][0])
    fH5.close()
    return

def next_batch(sess, model, data_dir, h5File, idList):
    length = len(idList)
    numBatch = math.floor(length / 4)
    fH5 = h5py.File(h5File, 'r+')
    idx1 = random.randint(0, length - 1)
    idx2 = idx1 + 1
    idx3 = idx1 + 2
    idx4 = idx1 + 3
    for i in range(int(numBatch + 1)):
        z = i / numBatch

        idx1 = idx1 % length
        idx2 = idx2 % length
        idx3 = idx3 % length
        idx4 = idx4 % length

        x = np.zeros((4, 224, 224, 3))
        x[0, :] = load_image("%s/%s.jpg" % (data_dir, idList[idx1]))
        x[1, :] = load_image("%s/%s.jpg" % (data_dir, idList[idx2]))
        x[2, :] = load_image("%s/%s.jpg" % (data_dir, idList[idx3]))
        x[3, :] = load_image("%s/%s.jpg" % (data_dir, idList[idx4]))
        print("x shape is :\n")
        print(x.shape)

        pos1 = fH5["%s/positives" % idList[idx1]]
        pos2 = fH5["%s/positives" % idList[idx2]]
        pos3 = fH5["%s/positives" % idList[idx3]]
        pos4 = fH5["%s/positives" % idList[idx4]]
        neg1 = fH5["%s/negatives" % idList[idx1]]
        neg2 = fH5["%s/negatives" % idList[idx2]]
        neg3 = fH5["%s/negatives" % idList[idx3]]
        neg4 = fH5["%s/negatives" % idList[idx4]]

        labels = np.zeros((4, 32768, 30))
        batch = np.zeros((120, 224, 224, 3))
        for j in range(10):
            batch[(4 * j), :] = load_image("%s/%s.jpg" % (data_dir, idList[pos1[j]]))
            batch[(4 * j + 1), :] = load_image("%s/%s.jpg" % (data_dir, idList[pos2[j]]))
            batch[(4 * j + 2), :] = load_image("%s/%s.jpg" % (data_dir, idList[pos3[j]]))
            batch[(4 * j + 3), :] = load_image("%s/%s.jpg" % (data_dir, idList[pos4[j]]))
            print("j: %s\n" % j)
        for k in range(20):
            batch[(4 * k + 40), :] = load_image("%s/%s.jpg" % (data_dir, idList[neg1[k]]))
            batch[(4 * k + 41), :] = load_image("%s/%s.jpg" % (data_dir, idList[neg1[k]]))
            batch[(4 * k + 42), :] = load_image("%s/%s.jpg" % (data_dir, idList[neg1[k]]))
            batch[(4 * k + 43), :] = load_image("%s/%s.jpg" % (data_dir, idList[neg1[k]]))
            print("k: %s\n" % k)
        begin = time.clock()
        output = sess.run(model.vlad_output, feed_dict = {'query_image:0': batch, 'train_mode:0' : False})
        end = time.clock()
        print("forward time is %.4f\n" % (end - begin))
        for j in range(30):
            labels[:, :, j] = output[(30 * j) : (30 * j + 4), :]

        yield x, labels, z

        idx1 += 4
        idx2 += 4
        idx3 += 4
        idx4 += 4 
    fH5.close()
    return