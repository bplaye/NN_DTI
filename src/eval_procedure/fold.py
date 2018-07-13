import tools.util as tls
from model.model_tf import Model_tf_ST


# evaluate and print performance on a single fold (early stopping overfitted on it)
def fold(dict_param, n, nb_fold=5):
    print('\33[91m' + 'Start one fold on ' + str(n) + '\33[0m')
    dataset, stream_type = dict_param['dataset'], dict_param['stream_type']

    model = Model_tf_ST(dict_param)
    test_files, y_te = tls.get_test_components_TFrecords(dataset, n, nfold=nb_fold,
                                                         stream_type=stream_type)
    train_files, n_tr_steps = tls.get_train_components_TFrecords(
        dataset, [i for i in range(nb_fold) if i != n], nfold=nb_fold, class_weight=False,
        stream_type=stream_type)

    print('number of samples in train set:', n_tr_steps)
    model.fit((train_files, n_tr_steps), evaluate_train=False, test_data=test_files)
    model.predict_eval(test_files, force=True)
