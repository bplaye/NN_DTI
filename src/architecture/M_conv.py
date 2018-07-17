import tensorflow as tf


def update_graph_emb(X, lay, model, hidden_units_emb, attribute_size):
    # graph_emb += [sigmoid(sum_{atom=a}(W^i_a*features_a)+b^i)]_{i=1:size of graph_emb}
    with tf.variable_scope('Emb_Conv' + str(lay), reuse=False):
        X = tf.layers.conv2d(X, hidden_units_emb,
                             kernel_size=(1, attribute_size),
                             strides=(1, attribute_size),
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(model.Mregph),
                             bias_initializer=tf.constant_initializer(0.1),
                             bias_regularizer=tf.contrib.layers.l2_regularizer(model.Mregph),
                             padding='same')
        if model.summary_bool:
            tf.contrib.layers.summarize_tensor(X)
            tf.contrib.layers.summarize_activation(X)
        X = tf.transpose(X, perm=[0, 1, 3, 2])
    return X


def update_atom_att(X, A, X_b, lay, atom_size, hidden_units_up, model):
  # Update atom embeddings
    with tf.variable_scope('Up_Conv' + str(lay), reuse=False):
      if lay == 1:
        atom_features_size = atom_size
      else:
        atom_features_size = hidden_units_up

      if model.use_bond_att is True:
        nodes_tensor = tf.transpose(X, perm=[0, 3, 1, 2])
        nodes_tensor = tf.transpose(tf.matmul(A, nodes_tensor), perm=[0, 2, 3, 1])
        X = tf.concat(axis=2, values=[nodes_tensor, X_b])
        filter_size = atom_features_size + self.edge_size
      else:
        filter_size = atom_features_size

      X = tf.layers.conv2d(X, hidden_units_up,
                           kernel_size=(1, filter_size),
                           strides=(1, filter_size),
                           activation=tf.nn.relu,
                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                           kernel_regularizer=tf.contrib.layers.l2_regularizer(model.Mregph),
                           bias_initializer=tf.constant_initializer(0.1),
                           bias_regularizer=tf.contrib.layers.l2_regularizer(model.Mregph),
                           padding='same')
      if model.summary_bool:
          tf.contrib.layers.summarize_tensor(X)
          tf.contrib.layers.summarize_activation(X)
      X = tf.transpose(X, perm=[0, 1, 3, 2])
    return X


def conv_emb(model):
    n_atoms, nb_layers = model.nb_MaxAtoms, model.M_nb_emb_layers
    atom_size, edge_size = model.nb_AtomFeatures, model.nb_BondFeatures
    hidden_units_emb = atom_size if model.M_hidden_units_emb is None else \
        model.M_hidden_units_emb
    hidden_units_up = atom_size if model.M_hidden_units_up is None else \
        model.M_hidden_units_up

    graph_emb_list = []

    A = tf.transpose(model.Aaug, perm=[0, 3, 1, 2])
    # make [batch_size, n_atoms, bond_attributes_size, n_channel] tensor with
    # for each sample and each channel, contains for each atom the sum of bond attributes
    X_b = tf.reduce_sum(model.X_b, 2)
    X_a = model.X_a # [batch_size, N_atoms_max, p_fea_atoms, n_channel=1]

    # Increment graph embedding
    emb = update_graph_emb(X_a, 0, model, hidden_units_emb, atom_size)
    graph_emb_list.append(emb)

    for lay in range(1, nb_layers + 1):
      X_a = update_atom_att(X_a, A, X_b, lay, atom_size, hidden_units_up, model)

      emb = update_graph_emb(X_a, lay, model, hidden_units_emb, hidden_units_up)
      graph_emb_list.append(emb)

    with tf.variable_scope('mol_embedding'):
        emb_size = hidden_units_emb
        emb = tf.reduce_sum(tf.add_n(graph_emb_list), axis=[1, 3]),
    return emb, emb_size
