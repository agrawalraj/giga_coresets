
from __future__ import division

import numpy as np
from sklearn.kernel_approximation import RBFSampler
from run_methods import GIGA_transform

# data matrix
X = np.random.rand(500, 20)
rbf = RBFSampler(n_components=1000)
rbf.fit(X)
X_RFM_ground_truth = rbf.transform(X)
X_RFM_ours = GIGA_transform(X, rbf.random_weights_, rbf.random_offset_, np.ones(1000))
print(np.linalg.norm(X_RFM_ours - X_RFM_ground_truth))

w_giga = np.zeros(1000)
w_giga[10] = 3
w_giga[200] = 4
w_active = w_giga[w_giga > 0]
X_giga_ground_truth = X_RFM_ground_truth[:, w_giga > 0] * np.sqrt(w_active)
X_giga_ours = GIGA_transform(X, rbf.random_weights_, rbf.random_offset_, w_giga)
print(np.linalg.norm(X_giga_ours - X_giga_ground_truth))
