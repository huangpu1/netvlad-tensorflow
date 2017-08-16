import tensorflow as tf

import netvlad
import eva_utils
import eva_init
import os

tf.app.flags.DEFINE_string('model_path', 'checkpoint/', 'path of trained models')
tf.app.flags.DEFINE_string('data_dir', 'tokyoTM/images', 'directory of image data')
tf.app.flags.DEFINE_string('eva_h5File', 'index/evadata.hdf5', 'evaluation dataset hdf5 file')
tf.app.flags.DEFINE_string('mat_path', 'tokyoTM/tokyoTM_val.mat', 'evaluation image dataset .mat')

tf.app.flags.DEFINE_integer('batch_size', 120, 'num of triplets in a batch')
tf.app.flags.DEFINE_integer('print_every', 5, 'print every ... batch')
tf.app.flags.DEFINE_integer('numRecall', 10, 'number of recall candidates')

tf.app.flags.DEFINE_boolean('initH5', True, 'init hdf5 file or not')
tf.app.flags.DEFINE_boolean('computeDist', True, 'compute distances of images or not')
tf.app.flags.DEFINE_boolean('loadImage', False, 'load dataset images or not')

FLAGS = tf.app.flags.FLAGS


def main(_):
    qList, dbList = eva_init.get_List(FLAGS.mat_path)
    update_index_every = 600 / FLAGS.batch_size

    if FLAGS.initH5:
        eva_init.h5_initial(FLAGS.eva_h5File)
    if FLAGS.computeDist:
        eva_init.compute_dist(FLAGS.mat_path, FLAGS.eva_h5File)
    if FLAGS.loadImage:
        eva_init.multipro_load_image(FLAGS.data_dir, FLAGS.eva_h5File, qList, dbList)

    with tf.device('/gpu:0'):
        sess = tf.Session()

        query_image = tf.placeholder(tf.float32,[None, 224, 224, 3], name = 'query_image')
        train_mode = tf.placeholder(tf.bool, name = 'train_mode')

        model = netvlad.Netvlad(FLAGS.model_path)
        model.build(query_image, train_mode)

        print("number of total parameters in the model is %d\n" % model.get_var_count())

        sess.run(tf.global_variables_initializer())
        print("evaluation begins!\n")
        eva_utils.evaluate(sess, model, FLAGS.batch_size, FLAGS.eva_h5File, qList, dbList, FLAGS.numRecall)

if __name__ == '__main__':
    tf.app.run()