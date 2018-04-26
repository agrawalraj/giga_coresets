
from __future__ import division

import numpy as np
import pandas as pd
import bayesiancoresets as bc

# Author: Raj Agrawal
# Date: 04/26/18

# The following constructs GIGA coresets based on the GIGA algorithm as implemented in 
# https://github.com/trevorcampbell/bayesian-coresets

def sample_mat(N, V):
	"""
	Input: 
	N: Total numnber of datapoints
	V: Number of datapoint pairs sampled

	Output:
	list of two numpy arrays, where first element is a list of the indices of the rows 
	and second a list of the indices of the columns
	"""
    comp1 = np.random.choice(np.arange(N), V)
    comp2 = np.random.choice(np.arange(N), V)
    return (comp1, comp2)

def GIGA_construct_w(X, sampler, J, V, M, um='fast'):
	"""
	Input:
	X: N x d numpy data matrix
	sampler: sklearn object to sample random features (e.g. RBFSampler)
	J: "Up" projection dimension
	V: Number of data pairs to randomly sample 
	M: "down" projection dimension i.e. number of random features
	Note: The number of features could be less than M

	Output:
	w_fw: weights of length J with at most M non-zero entries
	w_active: weights of only the non-zero components
	sampler: sampler used to construct the random features
	"""
    N = X.shape[0]
    rand_data_pairs = sample_mat(N, V)
    unique_pts = sorted(list(set(rand_data_pairs[0]).union(set(rand_data_pairs[1]))))
    idx_dict = dict(zip(unique_pts, range(len(unique_pts))))
    # sampler.fit(X[unique_pts, :]) 
    X_rand_feats = sampler.transform(X[unique_pts, :])
    H_mat = np.zeros((J, V))
    root_V = np.sqrt(V)
    for i in range(V):
        idx1 = idx_dict[rand_data_pairs[0][i]] # index in X_rand_feats array
        idx2 = idx_dict[rand_data_pairs[1][i]] 
        H_mat[:, i] = root_V * np.multiply(X_rand_feats[idx1, :], X_rand_feats[idx2, :])
    fw_giga = bc.GIGA(H_mat)
    fw_giga.run(M, update_method=um)
    w_fw = fw_giga.weights()
    w_mask = w_fw > 0
    feats_used = (sampler.random_weights_[:, w_mask], sampler.random_offset_[w_mask])
    w_active = w_fw[w_mask]
    return(w_fw, w_active, feats_used, sampler)
