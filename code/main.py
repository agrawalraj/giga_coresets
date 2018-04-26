
from __future__ import division

from sklearn.datasets import load_svmlight_file
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle 
import os
import numpy as np

from run_methods import do_all

# Author: Raj Agrawal
# Date: 04/26/18

# The following runs RFM-GIGA, RFM-JL, and RFM on adult, MNSIT, sensorless, and human activitiy datasets
# from UCI and LIBSVM 

if __name__ == "__main__":

	########## Adult ################### 
	sparse_mat = load_svmlight_file('../data/a9a')
	X = sparse_mat[0].todense()
	y = sparse_mat[1] 
	y = np.array(y, dtype=np.int32)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

	results = []
	for i in range(20):
		print('Adult iteration = ' + str(i))
		result = do_all([100, 250, 500, 750, 1000], X_train, y_train, X_test, y_test, 1, .25, J_up=5000, V=20000)
		pickle.dump(result, open( "../results/adult/adult_5k_20k_iter_{}.p".format(i), "wb" ) )
		results.append(result)

	######## Sensorless ################### 
	sparse_mat = load_svmlight_file('../data/Sensorless.scale') 
	X = sparse_mat[0].todense() 
	y = sparse_mat[1] 
	y = np.array(y, dtype=np.int32)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

	results = []
	for i in range(20):
		print('Sensorless iteration = ' + str(i))
		result = do_all([100, 250, 500, 750, 1000], X_train, y_train, X_test, y_test, 25, 5.75, J_up=5000, V=20000)
		pickle.dump(result, open( "../results/sensorless/sensorless_5k_20k_iter_{}.p".format(i), "wb" ) )
		results.append(result)

	######## Human Activiity ################ 
	X_train = np.loadtxt('../data/UCI HAR Dataset/train/X_train.txt')
	X_test = np.loadtxt('../data/UCI HAR Dataset/test/X_test.txt')
	y_train = np.loadtxt('../data/UCI HAR Dataset/train/y_train.txt')
	y_train = np.array(y_train, dtype=np.int32)
	y_test = np.loadtxt('../data/UCI HAR Dataset/test/y_test.txt')
	y_test = np.array(y_test, dtype=np.int32)

	results = []
	for i in range(20):
		print('Human iteration = ' + str(i))
		result = do_all([100, 250, 500, 750, 1000], X_train, y_train, X_test, y_test, 1, .1, J_up=5000, V=20000)
		pickle.dump(result, open( "../results/human_activity/human_activity_5k_20k_iter_{}.p".format(i), "wb" ) )
		results.append(result)

	############## MNIST ################ ]
	mnist = fetch_mldata('MNIST original')
	X = mnist.data / 255.
	y = mnist.target
	labelencoder_y = LabelEncoder()
	y = labelencoder_y.fit_transform(y)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

	results_mnist = []
	for i in range(20):
		print('iteration = ' + str(i))
		result = do_all([100, 250, 500, 750, 1000], X_train, y_train, X_test, y_test, 5, .05, J_up=5000, V=20000)
		pickle.dump(result, open( "../results/mnist/mnist_5k_20k_iter_{}.p".format(i), "wb" ) )
		results_mnist.append(result)
