
import numpy as np

# Author Raj Agrawal
# Date: 04/26/18

# Utility functions to approximate kernel matrix errors

def approx_kern_error_simple(X, feat_mat, kern_fn, nsamps=1000, ord='fro'):
	"""
	Input:
	X: N x d data matrix
	feat_mat: N x num_rand_feats matrix corresponding to embedding of X into the kern_fn feature space
	kern_fn: transforms a data matrix to a kernel matrix
	n_sams: Number of randomly picked datapoints in X to calculate the (sub) kernel matrix 
	ord: Metric of the matrix error 

	Output:
	Relative error
	"""
    N = X.shape[0]
    idx = np.arange(N)
    np.random.shuffle(idx)
    rand_idx = idx[:nsamps]
    X_sub_mat = X[rand_idx, :]
    K_X = kern_fn(X_sub_mat)
    feat_sub_mat = feat_mat[rand_idx, :]
    K_feat = feat_sub_mat.dot(feat_sub_mat.T)
    return np.linalg.norm(K_X - K_feat, ord=ord) / np.linalg.norm(K_X, ord=ord)

def approx_kern_error_avg(X, feat_mat, kern_fn, nsamps=1000, n_times=20, ord='fro'):
	"""
	See approx_kern_error_simple. Just averages error over n_times number of randomly selected
	columns
	"""
    errors = []
    for _ in range(n_times):
        errors.append(approx_kern_error_simple(X, feat_mat, kern_fn, nsamps, ord=ord))
    return np.mean(errors)