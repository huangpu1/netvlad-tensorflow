import tensorflow as tf

import netvlad
import utils
import initial

batch_size = 4
numEpoch = 30
data_dir = "247query_subset_v2"
h5File = "index_dir/datafile.hdf5"

"""
initial.h5_initial()
idList = initial.compute_dist(data_dir, h5File)
initial.index_initial(h5File, idList)
"""

idList = initial.get_idList(data_dir)
# initial.load_image(data_dir, h5File, idList)

def triplet_loss(q, labels, m):
    L2_distance = tf.norm(tf.subtract(tf.expand_dims(q, axis = -1), labels), axis = 1)
    positives, negatives = tf.split(L2_distance, [10, 20], axis = 1)
    loss = tf.reduce_sum(tf.nn.relu(tf.reduce_min(positives, axis = -1, keep_dims = True) + m - negatives))
    return loss

with tf.device('/gpu:0'):
    sess = tf.Session()

    query_image = tf.placeholder(tf.float32,[None, 224, 224, 3], name = 'query_image')
    labels = tf.placeholder(tf.float32, [None, 32768, 30])
    train_mode = tf.placeholder(tf.bool, name = 'train_mode')

    model = netvlad.Netvlad('./vgg16.npy')
    model.build(query_image, train_mode)

    print("number of total parameters in the model is %d\n" % model.get_var_count())

    sess.run(tf.global_variables_initializer())

    loss = triplet_loss(model.vlad_output, labels, 0.1)
    train = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
    train_loss = 0
    

    count = 0
    print("training begins!\n")
    for i in range(numEpoch):
        for x, y, z in utils.next_batch(sess, model, data_dir, h5File, idList):
            if count >= 50:
                count = 0
                utils.index_update(sess, model, data_dir, h5File, idList)
            count = count + 1
            _, train_loss = sess.run([train, loss], feed_dict = {query_image: x, labels: y, train_mode: True})
            if count % 1 == 0:
                print("Epoch: %d    progress: %.4f  training_loss = %.6f\n" % (i, z, train_loss))
        if (i + 1) % 5 == 0:
            model.save_npy(sess, "./netvlad_training_epoch_%d_loss_%.6f" % (i, train_loss))

