import unittest
import os
import sys
this_script_path = os.path.abspath(__file__)
this_script_folder = os.path.dirname(this_script_path)
sys.path.insert(1, os.path.join(this_script_folder, '..', 'src'))
from face_classifier import faceClassifier
import glob
import numpy as np


class TestUM(unittest.TestCase):

    def test_train_with_olivetti(self):
        fc = faceClassifier()
        self.assertEqual(len(fc.data.shape), 2)
        # data stored as 64*64 row vectors
        self.assertEqual(fc.data.shape[1], 64*64)
        # Olivetti data contain 40 subjects
        self.assertEqual(len(np.unique(fc.target)), 40)
        fc.train()
        # their coordinates in eigenface space as a matrix (.W)
        self.assertEqual(len(fc.W.shape), 2)


    def test_train_with_subject(self):
        img_dir = os.path.join(this_script_folder, 'images_yale')
        fc = faceClassifier()
        fc.add_img_data(img_dir)
        # data stored as 64*64 row vectors
        self.assertEqual(fc.data.shape[1], 64*64)
        # 40 + 1 subjects
        self.assertEqual(len(np.unique(fc.target)), 41)
        fc.train()
        # their coordinates in eigenface space as a matrix (.W)
        self.assertEqual(len(fc.W.shape), 2)


    def test_benchmark(self):
        img_dir = os.path.join(this_script_folder, 'images_yale')
        fc = faceClassifier(ratio = .725)
        fc.add_img_data(img_dir)
        fc.benchmark()
        self.assertNotEqual(fc.classification_report, None)
        print(fc.classification_report)
        fc.benchmark(imshow=True, wait_time=0.8, which_labels=[0, 5, 13, 28, 40])


    def test_export_import(self):
        img_dir = os.path.join(this_script_folder, 'images_yale')
        fc = faceClassifier()
        fc.add_img_data(img_dir)
        # write as pickle files
        fc.export()

        fc2 = faceClassifier(data_pkl = '/tmp/data.pkl', target_pkl = '/tmp/target.pkl')
        self.assertEqual(len(np.unique(fc2.target)), 41)


if __name__ == '__main__':
    unittest.main()
