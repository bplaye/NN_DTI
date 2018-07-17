import numpy as np
from sklearn import metrics

LIST_TASKS = ['DB_S']

LIST_MULTICLF_TASKS = []
LIST_CLF_TASKS = ['DB_S']
LIST_REGR_TASKS = []


def convert_to_list(ob, ):
    if type(ob) == np.float32:
        return [ob]
    elif type(ob) == np.ndarray:
        return ob.tolist()
    else:
        return ob


def get_perf(Y_test, test_pred, dataset):

  if dataset in LIST_CLF_TASKS:
    # print('get_perf: clf task')
    auc = metrics.roc_auc_score(Y_test, test_pred, average=None)
    pr = metrics.average_precision_score(Y_test, test_pred, average=None)

    test_pred_ = np.round(test_pred)
    if len(test_pred_.shape) > 1:
        accuracy = np.asarray([metrics.accuracy_score(Y_test[:, ind], test_pred_[:, ind])
                               for ind in range(test_pred_.shape[1])])
    else:
        accuracy = metrics.accuracy_score(Y_test, test_pred_)
    return auc, pr, accuracy

  elif dataset in LIST_MULTICLF_TASKS:
    auc = np.asarray([metrics.roc_auc_score(Y_test[:, ind], test_pred_[:, ind])
                      for ind in range(test_pred_.shape[1])])
    pr = np.asarray([metrics.average_precision_score(Y_test[:, ind], test_pred_[:, ind])
                     for ind in range(test_pred_.shape[1])])

    test_pred_ = np.asarray(test_pred, dtype=int)
    accuracy = np.asarray([metrics.accuracy_score(Y_test[:, ind], test_pred_[:, ind])
                           for ind in range(test_pred_.shape[1])])
    return auc, pr, accuracy

  elif dataset in LIST_REGR_TASKS:
    mse = np.asarray([metrics.mean_squared_error(Y_test[:, ind], test_pred[:, ind])
                      for ind in range(Y_test.shape[1])])
    return mse


def get_path(dataset):
    return 'data/' + dataset + '/'


def get_test_components_TFrecords(dataset, n, nfold=5, stream_type='TFrecords'):
    path = get_path(dataset) + dataset + '_' + str(nfold) + 'fold_' + str(n)
    return [path + '.TFrecord'], np.load(path + '_y.npy')


def get_train_components_TFrecords(dataset, list_ind, nfold=5, class_weight=False,
                                   stream_type='TFrecords'):
    if class_weight is False:
        balanced = ''
    else:
        balanced = '_balanced'
    n_tr_steps = 0
    train_files = []
    for i in list_ind:
        path = get_path(dataset) + dataset + '_' + str(nfold) + 'fold_' + str(i)
        train_files.append(path + balanced + '.TFrecord')
        n_tr_steps += (np.load(path + balanced + '_y.npy')).shape[0]

    return train_files, n_tr_steps


def percent(x):
    return np.round(x * 100, 2)
