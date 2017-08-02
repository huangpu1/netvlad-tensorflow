import tensorflow as tf

import netvlad
import train_utils
import train_init
import os


batch_size = 4
update_index_every = 600 / batch_size
numEpoch = 30
checkpoint_dir = "checkpoint"
data_dir = "tokyoTM/images"
train_h5File = "index/traindata.hdf5"
mat_path = "tokyoTM/tokyoTM_train.mat"

qList, dbList = train_init.get_List(mat_path)
train_init.h5_initial(train_h5File)
#qList, dbList = train_init.compute_dist(mat_path, train_h5File)
#train_init.index_initial(train_h5File, qList, dbList)
train_init.multipro_load_image(data_dir, train_h5File, qList, dbList)

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
    lr = 0.001
    train = tf.train.GradientDescentOptimizer(lr).minimize(loss)
    train_loss = 0
    
    

    count = 0
    print("training begins!\n")
    for i in range(numEpoch):
        
        for x, y, z in train_utils.next_batch(sess, model, batch_size, train_h5File, qList, dbList):
            if count >= update_index_every:
                count = 0
                train_utils.index_update(sess, model, batch_size * 30, train_h5File, qList, dbList)
            count = count + 1
            _, train_loss = sess.run([train, loss], feed_dict = {query_image: x, labels: y, train_mode: True})
            if count % 5 == 0:
                print("Epoch: %d    progress: %.4f%%  training_loss = %.6f\n" % (i, z, train_loss))
        if (i + 1) % 5 == 0:
            model.save_npy(sess, "%s/netvlad_epoch_%d_loss_%.6f" % (checkpoint_dir, i, train_loss))
            lr = lr / 2
            update_index_every *= 2

