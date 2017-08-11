import tensorflow as tf

import netvlad
import train_utils
import train_init
import os

tf.app.flags.DEFINE_string('checkpoint_dir', 'checkpoint', 'directory to save trained models')
tf.app.flags.DEFINE_string('data_dir', 'tokyoTM/images', 'directory of image data')
tf.app.flags.DEFINE_string('train_h5File', 'index/traindata.hdf5', 'training dataset hdf5 file')
tf.app.flags.DEFINE_string('mat_path', 'tokyoTM/tokyoTM_train.mat', 'image dataset .mat')

tf.app.flags.DEFINE_integer('batch_size', 4, 'num of triplets in a batch')
tf.app.flags.DEFINE_integer('numEpoch', 30, 'num of epochs to train')
tf.app.flags.DEFINE_integer('lr', 0.0001, 'initial learning rate')
tf.app.flags.DEFINE_integer('print_every', 5, 'print every ... batch')
tf.app.flags.DEFINE_integer('save_every', 2, 'save model every ... epochs')

tf.app.flags.DEFINE_boolean('randomStartIdx', True, 'start fetching batch from idx...')

tf.app.flags.DEFINE_boolean('initH5', True, 'init hdf5 file or not')
tf.app.flags.DEFINE_boolean('computeDist', False, 'compute distances of images or not')
tf.app.flags.DEFINE_boolean('initIndex', False, 'init index of positives and negatives or not')
tf.app.flags.DEFINE_boolean('loadImage', False, 'load dataset images or not')
tf.app.flags.DEFINE_boolean('useRelu', False, 'use relu in loss function or not')

FLAGS = tf.app.flags.FLAGS


def triplet_loss(q, labels, m):
    # L2_distance = tf.norm(tf.subtract(tf.expand_dims(q, axis = -1), labels), axis = 1)
    L2_distance = tf.reduce_sum((tf.expand_dims(q, axis = -1) - labels) ** 2, axis = 1)
    positives, negatives = tf.split(L2_distance, [40, 20], axis = 1)
    if FLAGS.useRelu:
        loss = tf.reduce_sum(tf.nn.relu(tf.subtract(tf.add(tf.reduce_min(positives, axis = -1, keep_dims = True), m), negatives)))
    else:
        loss = tf.reduce_sum(tf.reduce_min(positives, axis = -1, keep_dims = True) + m - negatives)
    # loss = tf.reduce_sum(tf.nn.relu(tf.reduce_min(positives, axis = -1, keep_dims = True) + m - negatives))
    return loss

def main(_):
    qList, dbList = train_init.get_List(FLAGS.mat_path)
    update_index_every = 1000 / FLAGS.batch_size

    if FLAGS.initH5:
        train_init.h5_initial(FLAGS.train_h5File)
    if FLAGS.computeDist:
        train_init.compute_dist(FLAGS.mat_path, FLAGS.train_h5File)
    if FLAGS.initIndex:
        train_init.index_initial(FLAGS.train_h5File, qList, dbList)
    if FLAGS.loadImage:
        train_init.multipro_load_image(FLAGS.data_dir, FLAGS.train_h5File, qList, dbList)

    with tf.device('/gpu:0'):
        sess = tf.Session()

        query_image = tf.placeholder(tf.float32,[None, 224, 224, 3], name = 'query_image')
        labels = tf.placeholder(tf.float32, [None, 32768, 60])
        train_mode = tf.placeholder(tf.bool, name = 'train_mode')

        model = netvlad.Netvlad('vgg16.npy')
        model.build(query_image, train_mode)

        print("number of total parameters in the model is %d\n" % model.get_var_count())

        

        loss = triplet_loss(model.vlad_output, labels, 0.1)
        train = tf.train.RMSPropOptimizer(FLAGS.lr).minimize(loss)
        sess.run(tf.global_variables_initializer())
        train_loss = 0
    
        count = 0
        print("training begins!\n")
        for i in range(FLAGS.numEpoch):
        
            for x, y, z in train_utils.next_batch(sess, model, FLAGS.batch_size, FLAGS.train_h5File, FLAGS.randomStartIdx, qList, dbList):
                count = count + 1
                if count >= update_index_every:
                    count = 0
                    train_utils.index_update(sess, model, FLAGS.batch_size * 30, FLAGS.train_h5File, qList, dbList)
                
                _, train_loss = sess.run([train, loss], feed_dict = {query_image: x, labels: y, train_mode: True})
                if count % FLAGS.print_every == 0:
                    print("Epoch: %d    progress: %.4f%%  training_loss = %.6f\n" % (i, z, train_loss))
            if (i + 1) % FLAGS.save_every == 0:
                model.save_npy(sess, "%s/netvlad_epoch_%d_loss_%.6f" % (FLAGS.checkpoint_dir, i, train_loss))
                update_index_every *= 2

if __name__ == '__main__':
    tf.app.run()