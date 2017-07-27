import tensorflow as tf

def VLAD_pooling(inputs,
              k_centers,
              scope,
              use_xavier = True,
              stddev = 1e-3):
    """ VLAD orderless pooling - based on netVLAD paper:
  title={NetVLAD: CNN architecture for weakly supervised place recognition},
  author={Arandjelovic, Relja and Gronat, Petr and Torii, Akihiko and Pajdla, Tomas and Sivic, Josef},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={5297--5307},
  year={2016}

    Args:
      inputs: 4-D tensor B x H x W x D
      k_centers: scalar number of cluster centers

    Returns:
      Variable tensor
    """

    num_batches = inputs.get_shape()[0].value
    dim_features = inputs.get_shape()[3].value

    #Initialize the variables for learning w,b,c - Random initialization
    if use_xavier:
        initializer = tf.contrib.layers.xavier_initializer()
    else:
        initializer = tf.truncated_normal_initializer(stddev = stddev)

    with tf.variable_scope(scope) as sc:
        w = tf.get_variable('weights',
                            shape = [1, dim_features, 1, k_centers],
                            initializer = initializer)  # w is 1 x D x 1 x K
        c = tf.get_variable('centers',
                            shape = [num_batches, dim_features, k_centers],
                            initializer = initializer) # c is B x D x K


        #Pooling
        inputs_reshape = tf.reshape(tensor = inputs, shape = [num_batches, -1, dim_features], name = 'reshape')  # inputs_reshape is B x N x D
        descriptor = tf.expand_dims(input = inputs_reshape, axis = -1, name = 'expanddim')  # descriptor is B x N x D x 1
        conv1 = tf.nn.convolution(input = descriptor, filter = w, padding = VALID, name = 'conv1')  # conv1 is B x N x 1 x K
        a_k = tf.nn.softmax(logits = tf.squeeze(conv1), dim = -1, name = 'softmax1')  # a_k is B x N x K

        V_1 = tf.matmul(a = inputs_reshape, b = a_k, transpose_a = True)  # V_1 is B x D x K
        V_2 = tf.multiply(x = tf.tile(input = tf.reduce_sum(input_tensor = a_k, axis = 1, keep_dims = True), multiples = [1, dim_features, 1]), y = c)  # V_2 is B x D x K
        V = V_1 - V_2  # V is B x D x K

        V_norm_1 = tf.nn.reshape(tensor = tf.nn.l2_normalize(x = V, dim = 1), shape = [num_batches, -1])  # V_norm_1 is B x (D x K)
        output = tf.nn.l2_normalize(x = V_norm_1, dim = 1)

        return output