import numpy as np
import tools.util as tls
import random
import os
import glob
from model.model_tf import Model_tf_ST


MIN_EPOCH = 0


def reload_best_model(model, itask=None):
    model.best_test_perf = None
    model.best_test_pred = None
    model.emb_saver.restore(model.session, model.emb_save_file)
    model.pred_saver.restore(model.session, model.pred_save_file)


def short_nested_cv(dico, nb=1, nb_fold=5):
    print('\33[91m' + 'Start shortened nested cross-validation' + '\33[0m')

    dataset, class_weight, stream_type = \
        dico['dataset'], dico['balance_class'], dico['stream_type']
    if dataset in tls.LIST_CLF_TASKS + tls.LIST_MULTICLF_TASKS:
        perf_auc, perf_pr = [[], []], [[], []]
    elif dataset in tls.LIST_REGR_TASKS:
        perf_mse = [[], []]

    for i in range(nb_fold):
        nested_test_files, y_te = \
            tls.get_test_components_TFrecords(dataset, i, nb_fold, stream_type)

        print('#####' + '\33[34m' + '\nInner cross-validation\n' + '\33[0m' + '#####')
        rest = [n for n in list(range(nb_fold)) if n != i]
        valid = random.sample(rest, nb)

        i_v = 0
        while i_v < len(valid):
            v = valid[i_v]
            model = Model_tf_ST(dico)

            valid_files, y_valid = \
                tls.get_test_components_TFrecords(dataset, v, nb_fold, stream_type)
            train_files_inner, n_tr_steps_inner = \
                tls.get_train_components_TFrecords(
                    dataset, [k for k in range(nb_fold) if k != i and k != v],
                    nb_fold, class_weight, stream_type)
            print('i', i, 'v', v, 'valid', valid, 'train_files_inner', train_files_inner,
                  'test_files', valid_files, 'nested_test_files', nested_test_files)
            train, test, y_test = \
                (train_files_inner, n_tr_steps_inner), valid_files, y_valid

            # print('number of samples in inner train set:', n_tr_steps_inner)
            model.fit(train, evaluate_train=False, test_data=test)
            if model.e > MIN_EPOCH:  # check if enought learning
                print('\33[33m->perf on INNER test fold \33[0m')
                model.predict(test)
                model.evaluate(y_test)

                if dataset in tls.LIST_CLF_TASKS + tls.LIST_MULTICLF_TASKS:
                    perf_auc[0].append(model.auc)
                    perf_pr[0].append(model.pr)
                elif dataset in tls.LIST_REGR_TASKS:
                    perf_mse[0].append(model.mse)

                # get perf on the outer test fold
                print('\n\33[33m->perf on OUTER test fold \33[0m')
                reload_best_model(model)
                model.predict_eval(test_files=nested_test_files, force=True)
                if dataset in tls.LIST_CLF_TASKS + tls.LIST_MULTICLF_TASKS:
                    perf_auc[1].append(model.auc)
                    perf_pr[1].append(model.pr)
                elif dataset in tls.LIST_REGR_TASKS:
                    perf_mse[1].append(model.mse)

                filelist = glob.glob(os.path.join('model/' + str(model) + '/', "*"))
                for filename in filelist:
                    os.remove(filename)
                i_v += 1
            del model

    if dataset in tls.LIST_CLF_TASKS + tls.LIST_MULTICLF_TASKS:
        print('inner auc_ncv: {}, pr_ncv: {}'.format(np.mean(perf_auc[0]), np.mean(perf_pr[0])))
        print('outer auc_ncv: {}, pr_ncv: {}'.format(np.mean(perf_auc[1]), np.mean(perf_pr[1])))
        return perf_auc, perf_pr
    elif dataset in tls.LIST_REGR_TASKS:
        print('inner mse_ncv: {}'.format(np.mean(perf_mse[0])))
        print('outer mse_ncv: {}'.format(np.mean(perf_mse[1])))
        return perf_mse
