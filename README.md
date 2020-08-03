<h1 align="center">EZfaces</h1>
<p align="center"><i>Easily create your own face recognition system in Python using Eigenfaces</i></p>
<hr><p align="center">

# Description
![Python package](https://github.com/0xLeo/EZfaces/workflows/Python%20package/badge.svg)  
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)  

A tool for face recognition in Python. it implements [Turk and Pentland's paper](https://sites.cs.ucsb.edu/~mturk/Papers/mturk-CVPR91.pdf) and is based on the [Olivetti faces dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_olivetti_faces.html). Some of its features are:
* Load Olivetti faces to initialise dataset.
* Load new subjects from file.
* Read new subjects directly from webcam.
* Predict a novel face from file.
* Predict novel face from webcam.
* Export currently loaded dataset and load it later.
* Built-in benchmarking (classificaiton report) method.  

**Note**: When you add a new subject, it is recommended to take several (5 or more) pictures of its face profile from slighly different small angles. 


# Requirements
Newer verisons than the ones below will possibly work too but the project has been tested in CI (see [workflows](https://github.com/0xLeo/EZfaces/tree/master/.github/workflows)) in Python 3.7 and 3.8 with the following dependencies installed.
```
opencv-python 4.1.2.30
numpy 1.17.4
matplotlib 3.1.2
scipy 1.4.1
scikit-image 0.16.2
scikit-learn 0.22
```
You can install them as follows:
```
cd <project_root>
pip install -r requirements.txt
```

# Usage example
1. Load new subject and predict from webcam
```
from face_classifier import faceClassifier
import cv2


fc = faceClassifier()
lbl_new = fc.add_img_data(from_webcam=True)
fc.train()
# take a snapshot from webcam
x_novel = fc.webcam2vec()
x_pred, lbl_pred = fc.classify(x_novel)
print("The ID of the newly added subject is %d. The prediction from the webcam is %d" %(lbl_new, lbl_pred))
cv2.imshow("Prediction", fc.vec2img(x_pred))
cv2.waitKey(3000)
cv2.destroyAllWindows()
```
