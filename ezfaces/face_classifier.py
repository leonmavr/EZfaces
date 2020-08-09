from sklearn.datasets import fetch_olivetti_faces
import numpy as np
import cv2
from collections import OrderedDict as OD
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
import pickle as pkl
import os
import time
import glob
import itertools as it
from typing import List, Tuple


class FaceClassifier():
    def __init__(self, ratio = 0.85, K = 200, data_pkl = None, target_pkl = None):
        """__init__. Class constructor.

        Parameters
        ----------
        ratio :
            How much of the total data to use for training (0 to 1)
        K :
            How many eigenface space base vectors to keep in order to express each image
        data_pkl :
            Pickle serialised file that contains data (see export method)
        target_pkl :
            Pickle serialised file that contains label (see export method)
        """
        if data_pkl is not None:
            with open(data_pkl, 'rb') as f:
                self.data = pkl.load(f)
        else:
            self.data = None        # data vectors
        if target_pkl is not None:
            with open(target_pkl, 'rb') as f:
                self.labels = pkl.load(f)
        else:
            self.labels = None      # label (ground truth) vectors
        self.train_data = OD()      # maps sample index to data and label
        self.test_data = OD()       # maps sample index to data and label
        # how many eigenfaces to keep
        self.K = K
        # how much training data to use as part of total data
        self.ratio = ratio
        # MxK matrix - each row stores the coords of each image in the eigenface space
        self.W = None
        self.classification_report = None # obtained from benchmarking
        # mean needed for reconstruction
        self._mean = np.zeros((1, 64*64), dtype=np.float32)
        if self.data is None and self.labels is None: # no pre-loaded data
            self._load_olivetti_data()
        self._TRAIN_SAMPLE = 1
        self._PRED_SAMPLE = 0


    def __str__(self):
        M = len(self.data)
        return "Loaded %d samples in total.\n"\
            "%d for training and %d for testing."\
            % (M, self.ratio*M, (1-self.ratio)*M)


    def _load_olivetti_data(self):
        """Load the Olivetti face data and save them in the class."""
        data, target = fetch_olivetti_faces(return_X_y = True)
        # data as floating vectors of length 64^2, ranging from 0 to 1
        self.data = np.array(data)
        # subject labels (sequential from to 0 to ...)
        self.labels = target


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


    def _read_from_webcam(self, new_label, stream = 0):
        """Takes face snapshots from webcam. Pass the new label of the subject
        being photographed."""
        print("Position your face in the green box.\n"
                "Press p to capture your face profile from slightly different angles,\n"
                "or q to quit.")
        time.sleep(3)
        # if stream == 0, try to open default webcam, else video from path
        cap = cv2.VideoCapture(stream)
        while True:
            # Capture frame by frame
            _, frame = cap.read()
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            min_shape = min(grey.shape)
            cv2.rectangle( frame, (0,0), (int(3*min_shape/4),
                int(3*min_shape/4)), (0,255,0), thickness = 4)
            cv2.namedWindow('frame', flags=cv2.WINDOW_GUI_NORMAL)
            cv2.imshow('frame',frame)
            k = cv2.waitKey(10) & 0xff
            if k == ord('q'):
                break
            elif k == ord('p'):
                im_cropped = grey[:int(3*min_shape/4), :int(3*min_shape/4)]
                cv2.destroyAllWindows()
                cv2.namedWindow('new data', flags=cv2.WINDOW_GUI_NORMAL)
                cv2.imshow("new data", im_cropped)
                cv2.waitKey(1500)
                cv2.destroyAllWindows()
                x = self.img2vec(im_cropped)
                self.data = np.array([*self.data, np.array(x, dtype=np.float32)])
                self.labels = np.append(self.labels, new_label)
        cap.release()
        cv2.destroyAllWindows()


    def add_img_data(self, dir_img: str = "", from_webcam: bool = False) -> int:
        """add_img_data. Adds data and their labels to existing database.

        Parameters
        ----------
        dir_img : str
            directory where image(s) of a subject are saved
        from_webcam : bool
            if True, opens webcam and lets the user capture face data

        Returns
        -------
        int
            The label of the newly added subject.
        """
        assert len(self.labels) != 0, "No labels have been generated!"
        # find all images in given folder
        fpaths = glob.glob(os.path.join(dir_img, '*.png'))
        fpaths += glob.glob(os.path.join(dir_img, '*.jpg'))
        fpaths += glob.glob(os.path.join(dir_img, '*.bmp'))
        # create new label for new subject
        target_new = self.labels[-1] + 1
        self.labels = np.append(self.labels, [target_new]*len(fpaths))
        # convert image to 64*64 data vector (ranging from 0 to 1)
        for i, f in enumerate(fpaths):
            im = cv2.imread(f)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im = np.asarray(cv2.resize(im, dsize = (64,64))).ravel()
            # normalise from 0 to 1 - the range of original Olivetti data
            im = im/255
            self.data = np.array([*self.data, np.array(im, dtype=np.float32)])
        if from_webcam:
            self._read_from_webcam(new_label = target_new)
        self.data = np.array(self.data)
        return target_new


    def train(self):
        """ Find the coordinates of each training image in the eigenface space """
        self._divide_dataset(ratio = self.ratio)
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
        training_or_test = [self._random_binary(ratio) for _ in self.data]
        self._subtract_mean()

        train_inds = [i for i,t in enumerate(training_or_test) if t == self._TRAIN_SAMPLE]
        test_inds = [i for i,t in enumerate(training_or_test) if t == self._PRED_SAMPLE]
        # {index: (data_vector, data_label)}, index starts from 0 
        self.train_data = OD(                                       # ordered dict
                dict(zip(train_inds,                                # keys
            zip(self.data[train_inds,:], self.labels[train_inds]))) # vals
                )
        self.test_data = OD(                                        # ordered dict
                dict(zip(test_inds,                                 # keys
            zip(self.data[test_inds,:], self.labels[test_inds])))   # vals
                )


    def _random_binary(self, prob_of_1 = .5) -> int:
        """_random_binary. Randomly returns 0 or 1. Accepts probability 
        to return 1 as input."""
        return np.round(np.random.uniform(.5, 1.5) - 1 + prob_of_1).astype(np.uint8)


    def get_test_sample(self) -> tuple:
        """ Get random training sample and its label. Returns (data vector, label) """
        Ntest = len(self.test_data)
        n = np.random.randint(0, Ntest)
        test_ind = [k for k in self.test_data.keys()][n]
        return self.test_data[test_ind] # data, label


    def classify(self, x_new:np.array) -> tuple:
        """classify. Classify an input data vector.

        Parameters
        ----------
        x_new : np.array
            Data vector

        Returns
        -------
        tuple
            containing the predicted data vector and its label (data, label)
        """
        train_inds = sorted([i for i in self.train_data.keys()])
        M = len(train_inds)
        # find eigenface space coordinates
        w_new = np.matmul(self.evec_XTX.T, x_new.T)
        w_new = w_new[:self.K]
        # if not match w/ itself else inf
        dists = [np.linalg.norm(w_new - self.W[i,:])
                if (np.linalg.norm(w_new - self.W[i,:]) > 0.0) else
                np.infty
                for i in range(M)]
        return (self.train_data[train_inds[np.argmin(dists)]][0], # data
                self.train_data[train_inds[np.argmin(dists)]][1]) # label


    def vec2img(self, x:list):
        """Converts an 1D data vector stored in the class to image."""
        x = np.array(x) + self._mean
        x = np.reshape(255*x, (64,64))
        return np.asarray(x, np.uint8)


    def img2vec(self, im) -> np.ndarray:
        """Converts an input greyscale image to an 1D data vector."""
        if not len(im.shape) == 2:
            raise RuntimeError("Provide a greyscale image as input.")
        x = np.asarray(cv2.resize(im, dsize=(64,64)), np.float32).ravel()
        x /= 255
        x = np.reshape(x, self._mean.shape)
        x -= self._mean
        # needed for eigenface coords dot product, do NOT delete this
        x = np.reshape(x, self.data[-1,:].shape)
        return x


    def webcam2vec(self):
        """ Opens webcam. The user can take a picture. Returns picture
        as data vector"""
        cap = cv2.VideoCapture(0)
        print("Position your face in the green box.\n"
                "Press p to capture a face picture.")
        while True:
            # Capture frame by frame
            _, frame = cap.read()
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            min_shape = min(grey.shape)
            cv2.rectangle( frame, (0,0), (int(3*min_shape/4),
                int(3*min_shape/4)), (0,255,0), thickness = 4)
            cv2.imshow('frame',frame)
            k = cv2.waitKey(10) & 0xff
            if k == ord('q'):
                break
            elif k == ord('p'):
                im_cropped = grey[:int(3*min_shape/4), :int(3*min_shape/4)]
                cv2.destroyAllWindows()
                cv2.namedWindow("new data", flags=cv2.WINDOW_GUI_NORMAL)
                cv2.imshow("new data", im_cropped)
                cv2.waitKey(1500)
                cv2.destroyWindow("new data")
                x = self.img2vec(im_cropped)
                break
        cap.release()
        cv2.destroyAllWindows()
        return x


    def benchmark(self, imshow = False, wait_time = 0.5, which_labels = []):
        """benchmark. Iterates over each test sample and classifies it.
        Genrates a classification report with all the classification metrics.

        Parameters
        ----------
        imshow : bool
            If True, show the actual vs predicted image, each for some times.
        wait_time : float
            How many seconds to show each actual vs predicted image for.
        which_labels : list
            Which labels to show. Useful when a new label was just added.
        """
        self.train()
        lbl_actual = []
        lbl_test = []
        for ind_test, test_data_lbl in self.test_data.items():
            # if we want to show only certain labels
            if len(which_labels) != 0:
                if test_data_lbl[1] not in which_labels:
                    continue
            x_actual = test_data_lbl[0]
            lbl_actual.append(test_data_lbl[1])
            x_test, lbl = self.classify(x_actual)
            lbl_test.append(lbl)
            if imshow:
                fig = plt.figure(figsize=(64, 64))
                cols, rows = 2, 1
                ax1 = fig.add_subplot(rows, cols, 1)
                ax1.title.set_text('actual: %d' % test_data_lbl[1])
                plt.imshow(self.vec2img(x_actual))
                ax2 = fig.add_subplot(rows, cols, 2)
                ax2.title.set_text('predicted: %d' % lbl)
                plt.imshow(self.vec2img(x_test))
                plt.show(block=False)
                plt.pause(wait_time)
                plt.close()
        if len(lbl_actual) != 0 and len(lbl_test) != 0:
            self.classification_report = classification_report(y_true = lbl_actual,
                    y_pred = lbl_test)


    def export(self, dest_folder = '/tmp') -> Tuple[str, str]:
        """export.

        Parameters
        ----------
        dest_folder :
            dest_folder

        Returns
        -------
        Tuple[str, str]
            Tuple containing the path to exported data and label file respectively.
            Empty string tuple if failure.
        """
        try:
            fpath_data = os.path.join(dest_folder, 'data.pkl')
            fpath_lbl = os.path.join(dest_folder, 'labels.pkl')
            with open(fpath_data, 'wb') as f:
                pkl.dump(self.data, f)
            with open(fpath_lbl, 'wb') as f:
                pkl.dump(self.labels, f)
            print("Wrote data and target as .pkl at:\n%s"
                    % os.path.abspath(dest_folder))
            return os.path.abspath(fpath_data), os.path.abspath(fpath_lbl)
        except Exception as e:
            return "", ""


    def grouper(self, inputs, n, fillvalue=None) -> list:
        """Credits https://realpython.com/python-itertools/
        >>> fc = faceClassifier()
        >>> nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> print(list(fc.grouper(nums, 4)))
        [(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, None, None)]
        """
        iters = [iter(inputs)] * n
        return list(it.zip_longest(*iters, fillvalue=fillvalue))


    def show_album(self, wait_time = 2.0):
        """show_album. Shows all gathered subjects in a several
        pages of 8x8 grids (photo "album").

        Parameters
        ----------
        wait_time :
            how long to wait between successive grid pages in sec
        """
        data_every_64 = self.grouper(self.data, 64)
        lbl_every_64 = self.grouper(self.labels, 64)
        cols, rows = 8, 8
        blank = np.zeros((64, 64), np.uint8)

        for data, lbls in zip(data_every_64, lbl_every_64):
            fig = plt.figure(figsize=(64, 64))
            for i in range(1, cols*rows +1):
                ax = fig.add_subplot(rows, cols, i)
                try:
                    plt.imshow(self.vec2img(data[i-1]))
                    ax.title.set_text("%d" % lbls[i-1])
                except:
                    plt.imshow(blank)
                    ax.title.set_text("blank")
            plt.subplots_adjust(wspace = .6)
            plt.show(block = False)
            plt.pause(wait_time)
            plt.close()
