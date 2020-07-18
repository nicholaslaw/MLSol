import numpy as np

class MLSol:
    def __init__(self):
        self.minority_class = {
            "SF": 0,
            "BD": 1,
            "RR": 2,
            "OT": 3,
            "MJ": 4
        }
        self.k = None
        self.num_samples = None
        self.num_features = None
        self.num_classes = None
        self.knn_ = None
        self.C = None
        self.w = None
        self.T = None

    def oversample(self, X: np.ndarray, Y: np.ndarray, P: float, k: int):
        """
        PARAMS
        ==========
        X: 2D numpy array
            shape (n, d)
            n: number of data samples
            d: number of features
        Y: 2D numpy array
            shape (n, q)
            n: number of data samples
            q: number of labels
            contains zeros and ones only since they are indicator entries
        P: float
            range (0, 1)
        k: integer
            number of nearest neighbours

        RETURNS
        ==========
        T: 2D numpy array
            shape (n, q)
            matrix indicating types of instances
        """
        GenNum = int(np.ceil(len(X) * P))
        X_prime, Y_prime = X.copy(), Y.copy()
        self.k = k
        self.num_samples = len(X)
        self.num_features = self.X.shape[-1]
        self.num_classes = self.Y.shape[-1]
        self.knn_ = self._knn_majority(X, Y)
        self.C = self.get_C(all_knn, Y)
        self.w = self.get_w(C, Y)
        self.T = self.InitTypes(Y)

        for _ in range(GenNum):
            seed_int, ref_int = self.seed_ref_instance()
            X_c, Y_c = self.GenerateInstance(seed_int, ref_int, X, Y, T)
            X_prime = np.vstack((X_prime, X_c))
            Y_prime = np.vstack((Y_prime, Y_c))

        return X_prime, Y_prime

    def get_C(self, knn_dic: dict, Y: np.ndarray):
        """
        PARAMS
        ==========
        knn_dic: dict
            output of _knn() method
        Y: 2D numpy array
            shape (n, q)
            n: number of data samples
            q: number of labels

        RETURNS
        ==========
        C: 2D numpy array
            shape (n, q)
            n: number of data samples
            q: number of labels
        """
        C = np.zeros_like(Y)

        for idx, ind in knn_dic.items(): # idx refers to the index of data sample
            Y_arr = Y[idx, :]
            knn_Y = Y[ind["knn_idx"], :]
            ind_ones = np.argwhere(Y_arr)[:, 0] # return indices of entries which are 1
            temp_mat = np.zeros((self.k, self.num_classes))
            temp_mat[:, ind_ones] = 1
            C[idx, :] = (knn_Y != temp_mat).sum(axis=0)
        
        return C

    def get_w(self, C: np.ndarray, Y: np.ndarray):
        """
        PARAMS
        ==========
        C: 2D numpy array
            shape (n, q)
            n: number of data samples
            q: number of labels
        Y: 2D numpy array
            shape (n, q)
            n: number of data samples
            q: number of labels

        RETURNS
        ==========
        w: 2D numpy array
            shape (n, )
        """
        C_less_1 = C < 1
        numerators = C * Y * C_less_1
        denominators = numerators.sum(axis=0)
        divide = np.ones_like(numerators)
        divide[:, :] = denominators
        return (numerators / divide).sum(axis=1)

    def _knn_majority(self, X: np.ndarray, Y: np.ndarray):
        """
        PARAMS
        ==========
        X: 2D numpy array
            shape (n, d)
            n: number of data samples
            d: number of features

        RETURNS
        ==========
        result: dict
            keys are indices of training data, values are arrays of indices of corresonding k neighbors
        """
        X_copy = X.copy()
        result = dict()
        for idx, row in enumerate(X_copy):
            # obtain knn
            row -= X
            row = np.linalg.norm(row, ord=2, axis=1)
            ind_ = np.argsort(row)
            ind_ = ind_[ind_!=idx][:self.k]

            # calculate majority of classes
            classes_counts = Y[ind_, :].sum(axis=0)
            majority = np.argsort(classes_counts)[::-1][0]

            result[idx] = {
                "knn_idx": ind_,
                "MJ": majority
            }
        
        return result

    def InitTypes(self, Y: np.ndarray):
        """
        PARAMS
        ==========
        Y: 2D numpy array
            shape (n, q)
            n: number of data samples
            q: number of labels

        RETURNS
        ==========
        T: 2D numpy array
            shape (n, q)
            matrix indicating types of instances
        """
        T = self.C.copy()
        T[T < 0.3] = self.minority_class["SF"]
        T[(T>=0.3) & (T<0.7)] = self.minority_class["BD"]
        T[(T>=0.7) & (T<1)] = self.minority_class["RR"]
        T[T==1] = self.minority_class["OT"]

        for row, dic in self.knn_.items():
            maj_class = dic["MJ"]

            if Y[row, maj_class] == 1:
                T[row, maj_class] = 1

        while True:
            change_check = False
            for i in range(self.num_samples):
                for j in range(self.num_classes):
                    if T[i, j] == self.minority_class["RR"]:
                        for nearest in self.knn_[i]["knn_idx"]:
                            if T[nearest, j] in [self.minority_class["SF"], self.minority_class["BD"]]:
                                T[i, j] = self.minority_class["BD"]
                                change_check = True
                                break
            
            if not change_check:
                return T

    def seed_ref_instance(self):
        seed_int = np.random.choice(list(range(self.num_samples)), 1, p=self.w)[0] 
        ref_int = self.knn_[seed_integer]["knn_idx"][np.random.randint(self.k, size=1)[0]]
        return seed_int, ref_int

    def GenerateInstance(self, seed_int: int, ref_int: int, X: np.ndarray, Y: np.ndarray, T: np.ndarray):
        """
        PARAMS
        ==========
        seed_int: integer
            seed index
        ref_int: integer
            reference index
        X: 2D numpy array
            shape (n, d)
            n: number of data samples
            d: number of features
        Y: 2D numpy array
            shape (n, q)
            n: number of data samples
            q: number of labels
            contains zeros and ones only since they are indicator entries
        T: 2D numpy array
            shape (n, q)
            matrix indicating types of instances

        RETURNS
        ==========
        X_t: 1D numpy array
            synthetic instance
        Y_t: 1D numpy array
            synthetic instance
        """
        X_s, Y_s = X[seed_int, :], Y[seed_int, :]
        X_r, Y_r = X[ref_int, :], Y[ref_int, :]
        X_c = np.random(low=0, high=1, size=self.num_features) * (X_r - X_s)
        Y_c = np.zeros_like(X_c)
        d_s = np.linalg.norm(X_c - X_s, ord=2)
        d_r = np.linalg.norm(X_c - X_r, ord=2)
        cd = d_s / (d_s + d_r)

        for j in range(self.num_classes):
            if Y_s[j] == Y_r[j]:
                Y_c[j] = Y_s[j]
            else:
                if T[seed_int, j] == self.minority_class["MJ"]:
                    seed_int, ref_int = ref_int, seed_int
                    cd = 1 - cd
                
                theta = 0.5
                if T[seed_int, j] == self.minority_class["BD"]:
                    theta = 0.75
                elif T[seed_int, j] == self.minority_class["RR"]:
                    theta = 1.00001
                elif T[seed_int, j] == self.minority_class["RR"]:
                    theta = -0.00001

                Y_c[j] = Y_s[j] if cd<=theta else Y_r[j]

        return X_c, Y_c