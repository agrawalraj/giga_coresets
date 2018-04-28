
import numpy as np
import os
import pickle
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

# TODO need to add kernel approx. Messy helper function to help with easy plotting... 
# i = 2 for class acc
# i = 3 for kern error 
def make_clean_dict(folder_path, i=2):
    RFM_results = dict()
    RFM_results['RFM'] = []
    RFM_results['RFM-JL'] = []
    RFM_results['RFM-GIGA'] = []
    RFM_sparsities = []

    paths = os.listdir(folder_path)
    for path in paths:
        result = pickle.load( open(folder_path + '/' + path, "rb" ) )
        RFM_sparsities += result[4]['RFM']
        RFM_results['RFM'] += result[i]['RFM']
        RFM_results['RFM-GIGA'] += result[i]['RFM_GIGA']
        RFM_results['RFM-JL'] += result[i]['RFM_JL']

    unique_values_RFM = sorted(list(set(RFM_sparsities)))
    RFM_final_dict = {}
    nyst_final_dict = {}
    RFM_final_dict['RFM'] = dict()
    RFM_final_dict['RFM-GIGA'] = dict()
    RFM_final_dict['RFM-JL'] = dict()
    for sparsity in unique_values_RFM:
        for method in ['RFM', 'RFM-GIGA', 'RFM-JL']:
            RFM_sparsities = np.array(RFM_sparsities)
            mask = RFM_sparsities == sparsity
            RFM_results[method] = np.array(RFM_results[method])
            RFM_final_dict[method][sparsity] = RFM_results[method][mask]
    return RFM_final_dict

def plot_results_rfm(result_obj, ax, i=2, title='plot'):
    result_obj = result_obj
    RFM = result_obj['RFM']
    RFM_JL = result_obj['RFM-JL']
    RFM_GIGA = result_obj['RFM-GIGA']
    sparsities = sorted(list(RFM_GIGA.keys()))
    RFM_means = []
    RFM_sds = []
    RFM_JL_means = []
    RFM_JL_sds = []
    RFM_GIGA_means = []
    RFM_GIGA_sds = []
    for sparsity in sparsities:
        RFM_means.append(np.mean(RFM[sparsity]))
        RFM_JL_means.append(np.mean(RFM_JL[sparsity]))
        RFM_GIGA_means.append(np.mean(RFM_GIGA[sparsity]))
        RFM_sds.append(np.std(RFM[sparsity]))
        RFM_JL_sds.append(np.std(RFM_JL[sparsity]))
        RFM_GIGA_sds.append(np.std(RFM_GIGA[sparsity]))
    # plt.plot(sparsities, RFM_means, yerr=RFM_sds, color='blue', label='RFM') 
    ax.errorbar(sparsities, RFM_means, RFM_sds, color='blue', label='RFM')
    ax.errorbar(sparsities, RFM_JL_means, RFM_JL_sds, color='red', label='RFM-JL') 
    ax.errorbar(sparsities, RFM_GIGA_means, RFM_JL_sds, color='green', label='RFM-GIGA') 
    ax.set_xlabel('# of Random Features') 
    if i == 2:
        ax.set_ylabel('Test Set Classification Accuracy') 
    else:
        ax.set_ylabel('Relative Error (Frobenius Norm)') 
    ax.set_title(title) 
    ax.legend()
