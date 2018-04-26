
from __future__ import division

import time
import numpy as np
from sklearn import random_projection
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics.pairwise import pairwise_kernels, rbf_kernel
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from .giga_RFM import sample_mat, GIGA_construct_w
from .utilities import approx_kern_error_simple, approx_kern_error_avg

# Author: Raj Agrawal
# Date: 04/26/18

# The following runs RFM-GIGA, RFM-JL, and RFM on a given dataset and measures
# test set classification performance, kernel approximation error, cpu time, and wall clock
# time

def do_all(J_grid, X_train, y_train, X_test, y_test, C, gamma, J_up=5000, V=20000, normalize=False, CV=False):
    wall_times = {'RFM': [], 'RFM_GIGA': [], 'RFM_JL': []}
    cpu_times = {'RFM': [], 'RFM_GIGA': [], 'RFM_JL': []}
    class_perform = {'RFM': [], 'RFM_GIGA': [], 'RFM_JL': []}
    kern_errors = {'RFM': [], 'RFM_GIGA': [], 'RFM_JL': []}
    sparsites = {'RFM': []}
    cv_params = []
    if normalize:
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
    if CV:
        classifier = SVC(kernel = 'rbf', random_state = 0)
        parameters = [
              {'C': [1, 5, 25, 125], 'kernel': ['rbf'],
               'gamma': [0.01, 0.05, 0.25, 1.25, 5.75]}]
        grid_search = GridSearchCV(estimator = classifier,
                                   param_grid = parameters,
                                   scoring = 'accuracy',
                                   cv = 5,)
        samps_idx = np.random.choice(X_train.shape[0],min(5000,X_train.shape[0])) 
        grid_search.fit(X_train[samps_idx, :], y_train[samps_idx])
        best_accuracy = grid_search.best_score_
        best_parameters = grid_search.best_params_
        C = best_parameters['C']
        gamma = best_parameters['gamma']
        cv_params = {'C': C, 'gamma': gamma}
        print(cv_params)
    kern_fn = lambda x: rbf_kernel(x, gamma=gamma)
    sampler_up = RBFSampler(gamma=gamma, n_components=J_up)
    sampler_up.fit(X_train)
    X_full_tr = sampler_up.transform(X_train)
    X_full_tst = sampler_up.transform(X_test)
    for J in J_grid:
        # Do for RFM-GIGA
        t_cpu = time.process_time()
        t_clock = time.perf_counter()
        w, w_active, _, _ = GIGA_construct_w(X_train, sampler_up, J_up, V, J)
        X_tr = X_full_tr[:, np.abs(w) > 0] * np.sqrt(w_active)
        X_tst = X_full_tst[:, np.abs(w) > 0] * np.sqrt(w_active)
        clf = LinearSVC(loss='squared_hinge', penalty="l2", C=C, dual=False)
        clf.fit(X_tr, y_train)
        acc = clf.score(X_tst, y_test)
        elapsed_time_cpu = time.process_time() - t_cpu
        elapsed_time_clock = time.perf_counter() - t_clock
        num_feats = len(w_active)
        kern_acc = approx_kern_error_avg(X_train, X_tr, kern_fn, nsamps=min(5000, X_train.shape[0]))
        kern_errors['RFM_GIGA'].append(kern_acc)
        print('RFM_GIGA for J = {4} - had clock time = {0}, cpu time = {3}, kernel error = {1}, class acc = {2}'.format(elapsed_time_clock, kern_acc, acc, elapsed_time_cpu, num_feats))
        print('RFM GIGA Sparsity = {0}'.format(len(w_active)))
        sparsites['RFM'].append(num_feats)
        wall_times['RFM_GIGA'].append(elapsed_time_clock)
        cpu_times['RFM_GIGA'].append(elapsed_time_cpu)
        class_perform['RFM_GIGA'].append(acc)
        # Do for RFM 
        t_cpu = time.process_time()
        t_clock = time.perf_counter()
        sampler = RBFSampler(gamma=gamma, n_components=num_feats)
        sampler.fit(X_train) 
        X_del = sampler.transform(X_train) # JUST FOR TIMING DONT USE / WANT SAME Rand FEATS
        X_del = sampler.transform(X_test) # JUST FOR TIMING DONT USE / WANT SAME Rand FEATS
        X_del = 0 
        clf = LinearSVC(loss='squared_hinge', penalty="l2", C=C, dual=False)
        clf.fit(X_full_tr[:, :num_feats], y_train)
        acc = clf.score(X_full_tst[:, :num_feats], y_test)
        elapsed_time_cpu = time.process_time() - t_cpu
        elapsed_time_clock = time.perf_counter() - t_clock
        kern_acc = approx_kern_error_avg(X_train, X_full_tr[:, :num_feats], kern_fn, nsamps=min(500, X_train.shape[0]))
        kern_errors['RFM'].append(kern_acc)
        print('RFM J = {4} - had clock time = {0}, cpu time = {3}, kernel error = {1}, class acc = {2}'.format(elapsed_time_clock, kern_acc, acc, elapsed_time_cpu, num_feats))
        wall_times['RFM'].append(elapsed_time_clock)
        cpu_times['RFM'].append(elapsed_time_cpu)
        class_perform['RFM'].append(acc)
        # Do for RFM-JL 
        t_cpu = time.process_time()
        t_clock = time.perf_counter()
        transformer = random_projection.GaussianRandomProjection(n_components=num_feats)
        X_tr = transformer.fit_transform(X_full_tr)
        X_tst = transformer.transform(X_full_tst)
        clf = LinearSVC(loss='squared_hinge', penalty="l2", C=C, dual=False)
        clf.fit(X_tr, y_train)
        acc = clf.score(X_tst, y_test)
        elapsed_time_cpu = time.process_time() - t_cpu
        elapsed_time_clock = time.perf_counter() - t_clock
        kern_acc = approx_kern_error_avg(X_train, X_tr, kern_fn, nsamps=min(500, X_train.shape[0]))
        kern_errors['RFM_JL'].append(kern_acc)
        print('RFM_JL for J = {4} - had clock time = {0}, cpu time = {3}, kernel error = {1}, class acc = {2}'.format(elapsed_time_clock, kern_acc, acc, elapsed_time_cpu, num_feats))
        wall_times['RFM_JL'].append(elapsed_time_clock)
        cpu_times['RFM_JL'].append(elapsed_time_cpu)
        class_perform['RFM_JL'].append(acc)
    return wall_times, cpu_times, class_perform, kern_errors, sparsites, cv_params

