import numpy as np
from numpy.core.numeric import Inf
from sklearn.cluster import KMeans


def likelihood(X: np.ndarray, M: np.ndarray):
    """Simplified multivariate lognormal likelihood

    Simplified for Σ=I and equal priors

    Args:
        X (np.ndarray): Samples
        M (np.ndarray): Target points

    Returns:
        (np.ndarray) : Likelihood
    """    

    return X @ M.T - 0.5 * (M**2).sum(axis=1)


def maximization(X, M):
    """Maximization of the likelihood

    Args:
        X (np.ndarray): Samples
        M (np.ndarray): Comparison targets

    Returns:
        int: Indices of the maximum likely comparison targets
    """

    return likelihood(X, M).argmax(axis=1)


class SuperKlust:
    """The Super-klust Algorithm
    """

    RATE_FORMAT = "12.10f"

    def __init__(self, k=2, dtype=np.float32, verbose=False, 
                 corr_max_iter=100, corr_err_bound=2.0, 
                 kmns_n_init=1, kmns_max_iter=100, kmns_tol=1e-2):
        """SuperKlust initialization

        Args:
            k (int, optional): Number of clusters per class. Defaults to 2.
            dtype (optional): Data type. Defaults to np.float32.
        """

        self.k = k
        self.dtype = dtype
        self.corr_max_iter = corr_max_iter
        self.corr_err_bound = corr_err_bound
        self.kmns_n_init = kmns_n_init
        self.kmns_max_iter = kmns_max_iter
        self.kmns_tol = kmns_tol
        self.verbose = verbose

    ##########################################
    # For scikit-learn estimator compatibility
    def get_params(self, deep=True):
        return {"k" : self.k,
        "dtype" : self.dtype,
        "corr_max_iter" : self.corr_max_iter,
        "corr_err_bound" : self.corr_err_bound,
        "kmns_n_init" : self.kmns_n_init,
        "kmns_max_iter" : self.kmns_max_iter,
        "kmns_tol" : self.kmns_tol,
        "verbose" : self.verbose}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    ##########################################

    def load_data(self, X, y):
        self.samples = X.astype(self.dtype)
        self.classes, self.labels = np.unique(y, return_inverse=True)
        self.n_classes = self.classes.shape[0]

    @property
    def n_samples(self): return self.samples.shape[0]

    @property
    def n_dim(self): return self.samples.shape[1]

    @property
    def n_genpts(self): return self.genpts.shape[0]


    def cluster(self):

        class_genpts = []

        kmns = KMeans(n_clusters=self.k, n_init=self.kmns_n_init,
                      max_iter=self.kmns_max_iter, tol=self.kmns_tol, random_state=0)
        for cls_inx in range(self.n_classes):
            cls_mask = self.labels == cls_inx
            kmns.fit(self.samples[cls_mask])
            class_genpts.append(kmns.cluster_centers_.astype(self.dtype))

        self.class_genpts = class_genpts


    def merge_and_label(self):

        genpts = np.vstack(self.class_genpts)

        interclass_assignments = maximization(self.samples, genpts)
        genpts_labels = np.empty(len(genpts), dtype=int)

        valid_genpt_indices = np.unique(interclass_assignments)
        for inx in valid_genpt_indices:
            labels = self.labels[interclass_assignments == inx]
            classes, counts = np.unique(labels, return_counts=True)
            genpts_labels[inx] = classes[counts.argmax()]

        self.genpts = genpts[valid_genpt_indices]
        self.genpts_labels = genpts_labels[valid_genpt_indices]


    def correct(self):

        best_params = (self.genpts.copy(), self.genpts_labels.copy())
        lowest_error = Inf

        for _ in range(self.corr_max_iter):
            assignments = maximization(self.samples, self.genpts)
            classification = self.genpts_labels[assignments]

            fp_mask = classification != self.labels
            fp_samples = self.samples[fp_mask]
            fp_assignments = assignments[fp_mask]

            error = np.sum(fp_mask) / self.n_samples

            if error < lowest_error:
                best_params = (self.genpts.copy(), self.genpts_labels.copy())
                lowest_error = error

            if error > self.corr_err_bound * lowest_error:
                break

            for genpt_inx in np.unique(fp_assignments):
                n_all = (assignments == genpt_inx).sum()
                n_fp = (fp_assignments == genpt_inx).sum()
                if n_all > n_fp:
                    self.genpts[genpt_inx] = (self.genpts[genpt_inx] * n_all - fp_samples[fp_assignments == genpt_inx].sum(axis=0)) / (n_all - n_fp)
                
        self.genpts, self.genpts_labels = best_params


    def train(self):

        #### Cluster ####
        self.cluster()
        
        #### Merge & Label ####
        self.merge_and_label()

        #### Correct ####
        self.correct()


    def fit(self, X, y):
        self.load_data(X, y)
        self.train()
        if self.verbose:
            print("Number of generator points: {}".format(self.n_genpts))
            print("Rate: {:{rfmt}}".format(self.rate(), rfmt=self.RATE_FORMAT))
        return self


    def predict(self, X):
        genpt_indices = maximization(X.astype(self.dtype), self.genpts)
        class_indices = self.genpts_labels[genpt_indices]
        return self.classes[class_indices]


    def score(self, X, y):
        n_correct = (y == self.predict(X)).sum()
        return n_correct / y.shape[0]


    def rate(self):
        assignments = maximization(self.samples, self.genpts)
        classification = self.genpts_labels[assignments]
        return (classification == self.labels).sum() / self.n_samples
