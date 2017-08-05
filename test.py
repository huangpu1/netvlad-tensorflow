import tensorflow as tf

import netvlad
import eva_utils
import eva_init
import os
import numpy as np

tf.app.flags.DEFINE_string('model_path', 'checkpoint/netvlad_epoch_29_loss_1.243645.npy', 'path of trained models')
FLAGS = tf.app.flags.FLAGS



def main(_):
    with tf.device('/gpu:0'):
        sess = tf.Session()

        query_image = tf.placeholder(tf.float32,[None, 224, 224, 3], name = 'query_image')
        train_mode = tf.placeholder(tf.bool, name = 'train_mode')

        model = netvlad.Netvlad(FLAGS.model_path)
        model.build(query_image, train_mode)

        print("number of total parameters in the model is %d\n" % model.get_var_count())

        sess.run(tf.global_variables_initializer())
        print("evaluation begins!\n")
        
        batch = np.zeros((11, 224, 224, 3))
        for i in range(1, 12):
            batch[i - 1, :, :, :] = eva_utils.load_image(("test/%s.JPG" % (i)))
        descriptor = sess.run(model.vlad_output, feed_dict = {'query_image:0': batch, 'train_mode:0' : False})
        A = np.dot(descriptor, descriptor.transpose())
        B = np.sum(descriptor ** 2, axis = 1, keepdims = True)
        C = np.sum(descriptor ** 2, axis = 1)
        print(B + C - 2 * A)
        D = B + C - 2 * A
        D[D < 0] = 0
        L2_distance = np.sqrt(D)
        print(L2_distance)
if __name__ == '__main__':
    tf.app.run()