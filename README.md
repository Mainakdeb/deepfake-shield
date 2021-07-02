![](/assets/deepfake-shield-banner-wide.png)
[![](https://img.shields.io/badge/heroku-deployed-green)](https://deepfake-shield.herokuapp.com/)

## 1. What is a Deepfake?
The term “DeepFake” is referred to a deep learning based technique that is able able to swap the face of a person by the face of another person in images.

## 2. The Problem 
The free access to large-scale public databases, together with the fast progress of deep learning techniques has made it easy for anyone to create deepfakes. Some of the harmful usages of such fake content include fake news, hoaxes, blackmailing and financial fraud.

## 3. Our Solution
Deepfake Shield uses deep-learning techniques to classify deepfakes in images. The diagram below summarises our project. Feel free to try out the web-app - https://deepfake-shield.herokuapp.com/

![](/assets/summary.png)

## 4. Using MLRun
MLRun has proven to be super useful in the following use cases:
### 4.1 Data Preprocessing
![](/assets/mlrun_util_preprocessing.png)
### 4.2 Automated Hyperparameter Search
Finding suitable training hyperparameters manually can be quite tedious. We have automated this process using MLRun's Grid Search functionality, thus making the process of finding hyperparameters a lot less painful.
![](/assets/mlrun_util_grid_search.png)
### 4.3 Training and Evaluation pipeline using MLRun
![](/assets/mlrun_util_train.png)

## Instructions - Data Preprocessing and Training
We have published the notebooks that we used to train our model, refer to the links below.
* Data Preprocessing notebook - [NBViewer](https://nbviewer.jupyter.org/github/Mainakdeb/deepfake-shield/blob/main/notebooks/preprocess_and_explore_data.ipynb), [Github](https://github.com/Mainakdeb/deepfake-shield/blob/main/notebooks/preprocess_and_explore_data.ipynb)
* Training notebook - [NBViewer](https://nbviewer.jupyter.org/github/Mainakdeb/deepfake-shield/blob/main/notebooks/train_deep_shield_model.ipynb), [Github](https://github.com/Mainakdeb/deepfake-shield/blob/main/notebooks/train_deep_shield_model.ipynb)
* We've used a [modified version](https://www.kaggle.com/unkownhihi/deepfake) of the [deepfake-detection-challenge](https://www.kaggle.com/c/deepfake-detection-challenge) dataset from kaggle
