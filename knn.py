import numpy as np

class KnnBlockMerger:
    def __init__(self, knn_object, fold_size=1000):
        self.knn_object = knn_object
        self.fold_size = fold_size

    def get_fold(self, fold_idx):
        return np.s_[fold_idx * self.fold_size:(fold_idx + 1) * self.fold_size]

    def kneighbors(self, Y):
        k = self.knn_object.n_neighbors
        ans = np.empty((Y.shape[0], 2 * k), dtype='float,int')
        for yfold_idx in xrange(-(Y.shape[0] / -self.fold_size)):
            for xfold_idx in xrange(-(self.X.shape[0] / -self.fold_size)):
                xfold = self.X[self.get_fold(xfold_idx)]
                yfold = Y[self.get_fold(yfold_idx)]
                self.knn_object.fit(xfold)
                dists, indices = self.knn_object.kneighbors(yfold)
                indices += xfold_idx * self.fold_size
                dists_merged = np.empty(dists.shape, dtype='float,int')
                dists_merged['f0'] = dists
                dists_merged['f1'] = indices
                if xfold_idx > 0:
                    ans[self.get_fold(yfold_idx), k:2*k] = dists_merged
                    ans[self.get_fold(yfold_idx)] = \
                        np.partition(ans[self.get_fold(yfold_idx)], k, order='f0', axis=1)
                else:
                    ans[self.get_fold(yfold_idx), :k] = dists_merged

        np.sort(ans, order='f0', axis=1)
        return ans[:, :k]['f0'],  ans[:, :k]['f1']

    def fit(self, X):
        self.X = X
        
    def transform_predict_proba(self, X, X_ans, Y):
        self.fit(X)
        dists, indices = self.kneighbors(Y)
        count_first_class = np.sum(X_ans[indices], axis=1)
        return count_first_class > indices.shape[1] / 2, 1.0 * count_first_class / indices.shape[1]


class KNN:
    def __init__(self, metric, knn_args=dict(), n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.knn_args = knn_args

    def kneighbors(self, Y):
        dists = self.metric(Y, self.X, **self.knn_args)
        dists_k = np.sort(dists, axis=1)[:, :self.n_neighbors]
        indices_k = np.argsort(dists, axis=1)[:, :self.n_neighbors]

        return dists_k, indices_k

    def fit(self, X):
        self.X = X