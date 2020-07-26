from sklearn.datasets import fetch_olivetti_faces
import numpy as np
import cv2
from collections import OrderedDict as OD


class faceClassifier():
    def __init__(self):
        self.data = None            # data vectors
        self.target = None          # actual labels
        self.train_set = OD()       # maps sample index to data and label
        # how many eigenfaces to keep - use arbitrary value for now
        self.K = 200
        # MxK matrix - each row stores the coords of each image in the eigenface space
        self.W = None
        # mean needed for reconstruction
        self._mean = np.zeros((1, 64*64), dtype=np.float32)
        self._load_olivetti_data(self)
        self._TRAIN_SAMPLE = 1
        self._PRED_SAMPLE = 0


    def _load_olivetti_data(self, do_centre = True):
        data, target = fetch_olivetti_faces(return_X_y = True)
        # data as floating vectors of length 64^2, ranging from 0 to 1
        self.data = np.array(data)
        # subject labels (sequential from to 0 to ...)
        self.target = target


    def _record_mean(self):
        self._mean += np.mean(self.data, axis=0) # along columns


    def _subtract_mean(self):
        """
        Make the mean of every column of self.data zero
        """
        self._record_mean()
        M = self.data.shape[0]
        C = np.eye(M) - 1/M*np.ones((M,1)) # centring matrix
        self.data = np.matmul(C, self.data)


    def add_img_data(self, fpaths = [], from_webcam = False):
        """add_img_data.

        Parameters
        ----------
        fpaths : list
            fpaths list of image files. If empty, they're ignored
        from_webcam : bool
            from_webcam if True, opens webcam and lets the user capture face data
        """
        assert len(self.target) != 0, "No labels have been generated!"
        # create new label for new subject
        target_new = self.target[-1] + 1
        self.target = np.append(self.target, [target_new]*len(fpaths))
        # convert image to 64*64 data vector (ranging from 0 to 1)
        for i, f in enumerate(fpaths):
            im = cv2.imread(f)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im = np.asarray(cv2.resize(im, dsize = (64,64))).ravel()
            # normalise from 0 to 1 - the range of original Olivetti data
            im = im/255
            self.data = np.array([*self.data, np.array(im, dtype=np.float32)])
        if from_webcam:
            #TODO: capture webcam, populate self.data, self.target etc.
            raise NotImplementedError(
                    "Capturing webcam data not supported yet")

        # subtract mean image from dataset
        self.data = np.array(self.data)
        #self._subtract_mean()


    def train(self):
        self._divide_dataset()
        # the matrix X to use for training
        X = np.array([v[0] for v in self.train_data.values()])
        # compute eig of MxN^2 matrix first instead of the N^2xN^2, N^2 >> M
        XXT = np.matmul(X, X.T)
        eval_XXT, evec_XXT = np.linalg.eig(XXT)
        # sort eig data by decreasing eigvalue values
        idx_eval_XXT = eval_XXT.argsort()[::-1]
        eval_XXT = eval_XXT[idx_eval_XXT]
        evec_XXT = evec_XXT[idx_eval_XXT]
        # now compute eigs of covariance matrix (N^2xN^2)
        self.evec_XTX = np.matmul(X.T, evec_XXT)
        # coordinates of each face in "eigenface" subspace
        self.W = np.matmul(X, self.evec_XTX)
        self.W = self.W[:, :self.K]


    def _divide_dataset(self, ratio = 0.85):
        """Divides dataset in training and test (prediction) data"""
        if not 0 < ratio < 1:
            raise RuntimeError("Provide a ratio between 0 and 1.")
        self.training_or_test = [self._random_binary(ratio) for _ in self.data]
        self._subtract_mean()

        #train_inds = sorted([i for i in self.train_data.keys()])
        train_inds = [i for i,t in enumerate(self.training_or_test) if t == self._TRAIN_SAMPLE]
        # {index: (data_vector, data_label)}, index starts from 0 
        self.train_data = OD( # ordered dictionary
                dict(zip(train_inds, # keys
            zip(self.data[train_inds,:], self.target[train_inds]))) # vals
                )


    def _random_binary(self, prob_of_1 = .5) -> int:
        """_random_binary. Randomly returns 0 or 1. Accepts probability 
        to return 1 as input."""
        return np.round(np.random.uniform(.5, 1.5) - 1 + prob_of_1).astype(np.uint8)


    def get_random_image(self) -> tuple:
        """get_random_image.
        Fetches random prediction image and its label from existing data

        Returns
        -------
        tuple
            containing image label and its index in self.data
        """
        i = np.random.randint(0, len(self.data))
        while self.training_or_test[i] == self._TRAIN_SAMPLE:
            i = np.random.randint(0, len(self.data))
        return self.target[i], i


    def classify(self, x_new:np.array) -> tuple:
        train_inds = sorted([i for i in self.train_data.keys()])
        # find eigenface space coordinates
        w_new = np.matmul(self.evec_XTX.T, x_new.T)
        w_new = w_new[:self.K]
        M = len(self.train_data)
        # if not match w/ itself and match w/ one of training data else inf
        dists = [np.linalg.norm(w_new - self.W[i,:])
                if ((np.linalg.norm(w_new - self.W[i,:]) > 0.0) and
                (self.training_or_test[train_inds[i]] == self._TRAIN_SAMPLE)) else
                np.infty
                for i in range(M)]
        return (self.train_data[train_inds[np.argmin(dists)]][1], # label
                self.train_data[train_inds[np.argmin(dists)]][0]) # data


    def vec2img(self, x:list):
        x = np.array(x) + self._mean
        x = np.reshape(255*x, (64,64))
        return np.asarray(x, np.uint8)


    def img2vec(self, im) -> np.ndarray:
        if not len(im.shape) == 2:
            raise RuntimeError("Provide a greyscale image as input.")
        x = np.asarray(cv2.resize(im, dsize=(64,64)), np.float32).ravel()
        x /= 255
        x = np.reshape(x, self._mean.shape)
        x -= self._mean
        # needed for eigenface coords dot product, do NOT delete this
        x = np.reshape(x, self.data[-1,:].shape)
        return x



fd = faceClassifier()
fd.add_img_data(['leo_4.jpg', 'leo_2.jpg', 'leo_1.jpg', 'leo_3.jpg', 'leo_5.jpg', 'leo_0.jpg', 'leo_7.jpg'])
fd.train()
for _ in range(8):
    lbl_actual, i_actual = fd.get_random_image()
    x_actual = fd.data[i_actual]
    lbl_pred, x_pred = fd.classify(x_actual)
    print("act: %d, pred: %d" %(lbl_actual, lbl_pred))
    cv2.imshow("", fd.vec2img(fd.data[i_actual]))
    cv2.waitKey(500)
    cv2.destroyAllWindows()
    cv2.imshow("", fd.vec2img(x_pred))
    cv2.waitKey(500)
    cv2.destroyAllWindows()
    import time
    time.sleep(1)
