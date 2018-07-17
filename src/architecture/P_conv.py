import tensorflow as tf
import numpy as np


def conv_emb(model):
    x, reuse_op = model.X_prot, None

    model.P_visualize_conv = []
    seq_size = float(model.max_seq_length)
    for i in range(model.P_nb_conv):
        height = model.nb_aa if i == 0 else model.P_nb_filters[i - 1]
        # No pooling !
        # seq_size = np.ceil(model.max_seq_length / (model.conv_strides * model.pooling_strides[i]))
        seq_size = float(np.ceil(seq_size / float(model.P_conv_strides)))
        print('conv', i, ' -> seq size :', seq_size)
        with tf.variable_scope('conv_layer_' + str(i + 1)):
            x = tf.layers.conv2d(x, model.P_nb_filters[i],
                                 kernel_size=(model.P_filter_size, height),
                                 strides=(model.P_conv_strides, height),
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(model.Pregph),
                                 bias_initializer=tf.constant_initializer(0.1),
                                 bias_regularizer=tf.contrib.layers.l2_regularizer(model.Pregph),
                                 padding='same',
                                 reuse=reuse_op)
            if model.summary_bool:
                tf.contrib.layers.summarize_tensor(x)
                tf.contrib.layers.summarize_activation(x)
            model.P_visualize_conv.append(x)
        # No pooling !
        # with tf.variable_scope('pool_layer_' + str(i + 1)):
        #     x = tf.layers.max_pooling2d(inputs=x,
        #                                 pool_size=(model.pooling_size[i], 1),
        #                                 strides=(model.pooling_strides[i], 1),
        #                                 padding='same')
        #     tf.contrib.layers.summarize_tensor(x)
        #     model.visualize_conv.append(x)
            x = tf.transpose(x, perm=[0, 1, 3, 2])

    with tf.variable_scope('prot_embedding'):
        emb_size = seq_size * model.P_nb_filters[-1]

    return x, emb_size
