import numpy as np

class WeightedQDA:

    def __init__(self, reg_param=1e-2):
        self.reg_param = reg_param

    def fit(self, X, y, sample_weight):

        X = np.asarray(X)
        y = np.asarray(y)
        w = np.asarray(sample_weight)

        self.classes_ = np.unique(y)

        d = X.shape[1]

        self.means_ = {}
        self.covs_ = {}
        self.priors_ = {}

        for c in self.classes_:

            mask = (y == c)

            Xc = X[mask]
            wc = w[mask]

            wc_sum = wc.sum()

            if wc_sum == 0:
                wc_sum = 1e-8

            mu = np.average(Xc, axis=0, weights=wc)

            X_centered = Xc - mu

            cov = (X_centered.T * wc) @ X_centered / wc_sum

            cov += self.reg_param * np.eye(d)

            self.means_[c] = mu
            self.covs_[c] = cov
            self.priors_[c] = wc_sum / w.sum()

        return self

    def _log_gaussian(self, X, mean, cov):

        Xc = X - mean

        inv_cov = np.linalg.inv(cov)

        log_det = np.linalg.slogdet(cov)[1]

        return -0.5 * (
            np.sum(Xc @ inv_cov * Xc, axis=1) + log_det
        )

    def predict_proba(self, X):

        X = np.asarray(X)

        log_probs = []

        for c in self.classes_:

            log_prior = np.log(self.priors_[c] + 1e-12)

            log_lik = self._log_gaussian(
                X,
                self.means_[c],
                self.covs_[c]
            )

            log_probs.append(log_prior + log_lik)

        log_probs = np.vstack(log_probs).T

        log_probs -= log_probs.max(axis=1, keepdims=True)

        probs = np.exp(log_probs)

        probs /= probs.sum(axis=1, keepdims=True)

        return probs

    def predict(self, X):

        return self.classes_[
            np.argmax(self.predict_proba(X), axis=1)
        ]

