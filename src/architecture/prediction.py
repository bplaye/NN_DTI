import tensorflow as tf


def pred(model, itask=None):
    emb, emb_size, nb_outputs, batch_size_placeholder = \
        tf.concat(model.P_emb, model.M_emb), model.P_emb_size + model.M_emb_size, \
        model.nb_outputs, model.batch_size_placeholder
    X = emb

    # logits of size [batch size, n_classes]
    X = tf.reshape(X, tf.stack([batch_size_placeholder, emb_size]))
    for i in range(len(model.nb_fully_con_units)):
        with tf.variable_scope('fully_con_layer_' + str(i + 1)):
            X = tf.layers.dense(X, units=model.nb_fully_con_units[i], activation=tf.nn.relu,
                                kernel_initializer=None,
                                bias_initializer=tf.zeros_initializer())
            X = tf.layers.dropout(X, rate=model.keep_prob_ph)
            if model.summary_bool:
                tf.contrib.layers.summarize_tensor(X)

    with tf.variable_scope('y_conv'):
        y = tf.layers.dense(X, units=nb_outputs)
        if model.summary_bool:
            tf.contrib.layers.summarize_tensor(y)
        logits = y

    model.visualize_pred = [logits, emb]

    return logits
