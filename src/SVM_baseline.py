import pickle
from sklearn.svm import SVC
import argparse
import numpy as np


def SVM_predict(K_tr, y_tr, K_te, K_nested_te, C):
    """
    Computes prediction.
    """
    pred = np.zeros((K_te.shape[0], 1))
    nested_pred = np.zeros((K_nested_te.shape[0], 1))
    clf = SVC(C=C, kernel='precomputed', probability=True, class_weight='balanced')
    clf.fit(K_tr, y_tr)
    pred[:, 0] = clf.predict_proba(K_te)[:, 1]
    nested_pred[:, 0] = clf.predict_proba(K_nested_te)[:, 1]

    return pred, nested_pred


class SVM_experiment():
  """

  """

  def __init__(self, dataset_name, C, ind_fold):
    self.dataset = dataset_name
    self.C = C
    self.ind_fold = ind_fold
    self.cv_pred = []
    self.cv_nested_pred = []
    self.nb_fold = 5

  def run_SVM(self, ):
    # load folds
    list_folds = pickle.load(open('data/' + self.dataset + '/' + self.dataset + '_' +
                                  str(self.nb_fold) + 'fold.data', 'rb'))
    nested_te = list_folds[self.ind_fold]
    list_tr = [list_folds[i] for i in range(self.nb_fold) if i != self.ind_fold]

    # load kernels
    Kcouple_filename = 'data/' + self.dataset + '/' + self.dataset + '_Kcouple.npy'
    y_filename = 'data/' + self.dataset + '/' + self.dataset + '_y.npy'
    Y = np.load(y_filename)
    Kcouple = np.load(Kcouple_filename)

    for i_fold in range(len(list_tr)):
      print('i_fold', i_fold)
      te = list_tr[i_fold]
      tr = [list_tr[i][j] for i in range(len(list_tr)) if i != i_fold
            for j in range(len(list_tr[i]))]
      K_tr, K_te, K_nested_te = Kcouple[tr, :], Kcouple[te, :], Kcouple[nested_te, :]
      y_tr = Y[tr]
      K_tr, K_te, K_nested_te = K_tr[:, tr], K_te[:, tr], K_nested_te[:, tr]
      del te
      del tr
      print('start pred')
      pred, nested_pred = SVM_predict(K_tr, y_tr, K_te, K_nested_te, self.C)
      self.cv_pred.append(pred)
      self.cv_nested_pred.append(nested_pred)
      del K_tr
      del y_tr
      del K_te


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', type=str, default=None,
                      help='Name of the considered dataset.')
  parser.add_argument('--C', type=float, default=None,
                      help='Reg. parameter of SVM.')
  parser.add_argument('--ind_fold', type=int, default=None,
                      help='index of the current fold.')
  FLAGS, unparsed = parser.parse_known_args()

  exp = SVM_experiment(dataset_name=FLAGS.dataset, C=FLAGS.C, ind_fold=FLAGS.ind_fold)
  exp.run_SVM()
  pickle.dump(exp, open('result/SVM_exp_' + FLAGS.dataset + '_C:' + str(FLAGS.C) + '_i:' +
                        str(FLAGS.ind_fold) + '.data', 'wb'))
