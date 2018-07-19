import tensorflow as tf


def pred(model, task=None):
    if task is None or task == 'DTI':
        emb_size = tf.cast(model.P_emb_size + model.M_emb_size, dtype=tf.int32)
        P_emb_size = tf.cast(model.P_emb_size, dtype=tf.int32)
        M_emb_size = tf.cast(model.M_emb_size, dtype=tf.int32)

        if task is None:
            nb_outputs, batch_size_placeholder = \
                model.nb_outputs[task], model.batch_size_placeholder
        else:
            nb_outputs, batch_size_placeholder = \
                model.nb_outputs[task], model.batch_size_placeholder[task]
        # nb_outputs = 1 if nb_outputs == 2 else nb_outputs
        P_emb = tf.reshape(model.P_emb, tf.stack([batch_size_placeholder, P_emb_size]))
        M_emb = tf.reshape(model.M_emb, tf.stack([batch_size_placeholder, M_emb_size]))
        emb = tf.reshape(tf.concat([P_emb, M_emb], axis=1),
                         tf.stack([batch_size_placeholder, emb_size]))
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
            y = tf.layers.dense(X, units=nb_outputs, activation=None)
            if model.summary_bool:
                tf.contrib.layers.summarize_tensor(y)
            logits = y

    elif task == 'ATC':
        with tf.variable_scope('fully_con_layer_' + str(i + 1)):
            M_emb_size = tf.cast(model.M_emb_size, dtype=tf.int32)

            nb_outputs, batch_size_placeholder = \
                model.nb_outputs[task], model.batch_size_placeholder[task]
            emb = tf.reshape(model.M_emb, tf.stack([batch_size_placeholder, M_emb_size]))
            X = emb

            # logits of size [batch size, n_classes]
            for i in range(len(model.nb_fully_con_units)):
                with tf.variable_scope('fully_con_layer_' + str(i + 1)):
                    X = tf.layers.dense(
                        X, units=model.nb_fully_con_units[i], activation=tf.nn.relu,
                        kernel_initializer=None, bias_initializer=tf.zeros_initializer())
                    X = tf.layers.dropout(X, rate=model.keep_prob_ph)
                    if model.summary_bool:
                        tf.contrib.layers.summarize_tensor(X)

            with tf.variable_scope('y_conv'):
                y = tf.layers.dense(X, units=nb_outputs, activation=None)
                if model.summary_bool:
                    tf.contrib.layers.summarize_tensor(y)
                logits = y

    elif task == 'bioassays':
        pass

    return logits, emb
