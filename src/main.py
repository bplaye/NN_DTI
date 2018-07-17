import argparse
import eval_procedure.fold as evalf
# import eval_procedure.cross_validation as evalc
import eval_procedure.nested_cross_validation as evaln
import param_selection.gridsearch as selection
from model.model_tf import Model_tf_ST
import pickle


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    list_dataset = ['DB_S']
    list_architecture = ['ConvModel']
    list_prot_dataset = []
    list_mol_dataset = []
    dict_param = {'dataset': 'DB_S',
                  'stream_type': 'TFrecords',
                  ## learning procedure parameters
                  'batch_size': 5, 'init_lr': 0.0001, 'decay_steps': 1, 'lr_decay_factor': 0.99,
                  'early_stopping_counter': 5, 'nb_epochs': 20,

                  ## prot embedding network architecture parameters
                  'P_architecture': 'ConvModel',
                  # conv
                  'P_nb_filters': [5, 10], 'P_filter_size': 6, 'P_conv_strides': 3,
                  'P_pooling_size': 1, 'P_pooling_strides': 1,
                  # reg
                  'P_l2_reg_coef': 0.01,

                  ## mol embedding network architecture parameters
                  'M_architecture': 'ConvModel',
                  # conv
                  # 'M_nb_emb_layers': 3, 'M_hidden_units_emb': 100,
                  # 'M_hidden_units_up': 100,
                  'M_nb_emb_layers': 1, 'M_hidden_units_emb': 10,
                  'M_hidden_units_up': 10,
                  # reg
                  'M_l2_reg_coef': 0.01,

                  ## prediction network architecture parameters
                  # seq tasks
                  # 'nb_fully_con_units': [1000, 100], 'dropout_keep_prob': 0.8,
                  'nb_fully_con_units': [100], 'dropout_keep_prob': 0.8,

                  'balance_class': False,
                  'summary': False,
                  }
    # pickle.dump([dict_param], open('model/test.data', 'wb'), protocol=2)
    # exit(1)

    parser.add_argument('-f', '--fold', type=int, help='which fold to test on')
    # all below useless if -f (int, fold index) is specified
    parser.add_argument('-x', '--xvalid', help='cross-validation or not', action='store_true')
    # all below useless if -x (bool, do or not a overall cross validation) is specified
    parser.add_argument('-nf', '--nestedfold', type=int,
                        help='which fold for nested cross-validation')
    # all below useless if -nf (int, fold index of outer test fold) is specified
    parser.add_argument('-nx', '--nestedxvalid', help='whole nested cv', action='store_true')
    # all below useless if -nx (bool, do or not a overall nested cross validation) is specified
    parser.add_argument('-g', '--gridsearch', help='gridsearch on dict')
    # parser.add_argument('-p', '--paramgrid', help='gridsearch on param', action='store_true')

    args = parser.parse_args()
    dataset = dict_param['dataset']

    # start procedure depending on the command line parameter
    if args.fold is not None:
        # asses perf on one fold by 10 fold cv
        evalf.fold(dict_param, args.fold)

    # elif args.xvalid:
    #     if type(dict_param['dataset']) is str:
    #         model = Model_tf_ST(dict_param)
    #     elif type(dict_param['dataset']) is list:
    #         model = Model_tf_MT(dict_param)
    #     # asses perf by 10fold cv
    #     evalc.cv(model)

    # elif args.nestedfold is not None:
    #     if type(dict_param['dataset']) is str:
    #         model = Model_tf_ST(dict_param)
    #     elif type(dict_param['dataset']) is list:
    #         model = Model_tf_MT(dict_param)
    #     # asses perf on one fold (whose index is args.nestedfold) by nested cv
    #     evaln.one_nested_cv(model, args.nestedfold)

    # elif args.nestedxvalid:
    #     if type(dict_param['dataset']) is str:
    #         model = Model_tf_ST(dict_param)
    #     elif type(dict_param['dataset']) is list:
    #         model = Model_tf_MT(dict_param)
    #     evaln.nested_cv(model)

    elif args.gridsearch is not None:
        if args.gridsearch == 'local':
            dict_params = [dict_param]
        else:
            dict_params = pickle.load(open(args.gridsearch, 'rb'))
        print(dict_params)

        selection.gridsearch_dict(dict_params)

    else:
        print('Please select fold or (nested) cross validation')
