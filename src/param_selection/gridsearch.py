import pickle
import sys
import os
try:
    import eval_procedure.nested_cross_validation as evaln
except ImportError:
    sys.path.append(os.getcwd() + '/src/')
    import eval_procedure.nested_cross_validation as evaln


def str_model(param_dict):
    list_cle = ['dataset', 'P_architecture', 'M_architecture',
                 'batch_size', 'init_lr', 'decay_steps',
                'lr_decay_factor', 'early_stopping_counter', 'nb_epochs',
                'nb_fully_con_units', 'dropout_keep_prob', 'balance_class', 'stream_type']
    if param_dict['M_architecture'] == 'ConvModel':
        list_cle += ['M_nb_filters', 'M_filter_size', 'M_l2_reg_coef']
        if param_dict['M_conv_strides'] != 1:
            list_cle.append('M_conv_strides')

    if param_dict['P_architecture'] == 'ConvModel':
        list_cle += ['P_nb_filters', 'P_filter_size', 'P_l2_reg_coef']
        if param_dict['P_conv_strides'] != 1:
            list_cle.append('P_conv_strides')

    return "_".join([str(param_dict[cle]).replace("[", "(").replace("]", ")").replace(' ', '')
                     for cle in list_cle])


# last parameter  says the perf assessment scheme:
# 's' for 5 fold short nested cv (only one inner fold)
# 'c' for 10 fold cv (only two inner folds)
# 'n' for 10 fold nestedcv (only two inner folds)
def gridsearch_dict(dict_params, nb_fold=5):
    print('#########' + '\33[93m' + '\nStarting Gridsearch\n' + '\33[0m' + '#########')

    for dico in dict_params:
        print("result/" + str_model(dico) + ".data")
        if not os.path.isfile("result/" + str_model(dico) + ".data"):
            perf = evaln.short_nested_cv(dico, 1, nb_fold)
            print("result/" + str_model(dico) + ".data")
            pickle.dump(perf, open("result/" + str_model(dico) + ".data", 'wb'))
        else:
            print(pickle.load(open("result/" + str_model(dico) + ".data", 'rb')))
