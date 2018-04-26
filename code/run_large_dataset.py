
from __future__ import division

from sklearn.datasets import load_svmlight_file
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
import pickle 
import os
import numpy as np

from run_methods import do_all

# Author: Raj Agrawal
# Date: 04/26/18

# The following runs RFM-GIGA, RFM-JL, and RFM on  

if __name__ == "__main__":
	sparse_mat = load_svmlight_file('../data/large_dataset')
	X = sparse_mat[0].todense()
	y = sparse_mat[1] 
	y = np.array(y, dtype=np.int32)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
	result = do_all([100, 250, 500, 750, 1000], X_train, y_train, X_test, y_test, 1, .25, J_up=5000, V=20000, CV=True)
	pickle.dump(result, open( "../results/large_dataset/large_dataset_5k_20k_iter_0.p".format(i), "wb" ) )
