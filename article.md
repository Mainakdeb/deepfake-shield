![](/assets/deepfake-shield-banner-wide.png)


## 1. What is a Deepfake?
The term “[Deepfake](https://en.wikipedia.org/wiki/Deepfake)” is referred to a deep learning based technique that swaps the face of a person with another face in an image.

## 2. :mag: The Problem 
It is easier than ever to create deepfakes of anyone using the tools available online. Deepfakes can be used by people with evil intentions like generating fake news, hoaxes, blackmailing and financial fraud.

## 3. 🧑‍🔬 Our Solution
Deepfake Shield is a tool that uses deep-learning to detect deepfakes in an image. The diagram below summarises our project. Feel free to try out the web-app - https://deepfake-shield.herokuapp.com/

![](/assets/summary.png)

## 4. 🧙‍♂️ Making the most out of MLRun 

Our favourite way to use `mlrun` has been the `# mlrun: start-code` and  `# mlrun: end-code`. The ease of use in terms of tracking experiments helped us progress rapidly from experimentation to training and deployment, all without the hassle of trying to keep logs manually.


### 4.1 :mag: Data Preprocessing

When preprocessing the data, `mlrun.artifacts.PlotArtifacts` helped us visualise a bias in the dataset. We found that the number of real images is much lower than that of the number of fake images. This was fixed by inflating the number of real faces using the [FFHQ dataset](https://github.com/NVlabs/ffhq-dataset)

![](/assets/mlrun_util_preprocessing.png)

### 4.2 🧑‍🔬 Automated Hyperparameter Search
Finding suitable training hyperparameters manually can be quite tedious. We automated this process using `mlrun.new_task().with_hyper_params(grid_params, selector="min.loss")`, thus making the process of finding hyperparameters a lot less painful.

![](/assets/mlrun_util_grid_search.png)


### 4.3 💡 Training + Evaluation 

The pipeline that we built comprises of 2 different models:
* We use a pretrained [BlazeFace](https://github.com/hollance/BlazeFace-PyTorch) model (which can be retrained if needed) for extracting faces from images.
* We trained a customized implementation of EfficientNet for classsifying the extracted faces accordingly. 

The model was trained using the ideal hyperparameters found using grid-search with `mlrun`. The training and evaluatoion logs were also tracked using `mlrun`

![](/assets/pred_pipeline.png)
![](/assets/mlrun_util_train.png)


```
  

## 📗 Resources

* **Data exploration**: - [NBViewer](https://nbviewer.jupyter.org/github/Mainakdeb/deepfake-shield/blob/main/notebooks/preprocess_and_explore_data.ipynb), [Github](https://github.com/Mainakdeb/deepfake-shield/blob/main/notebooks/preprocess_and_explore_data.ipynb)
* **Preprocessing + Hyperparameter search**: - [NBViewer](https://nbviewer.jupyter.org/github/Mainakdeb/deepfake-shield/blob/main/notebooks/train_deep_shield_model.ipynb), [Github](https://github.com/Mainakdeb/deepfake-shield/blob/main/notebooks/train_deep_shield_model.ipynb)
* **Training + Evaluation**: [NBViewer]() [Github]()
* **Dataset**: We've used a [modified version](https://www.kaggle.com/unkownhihi/deepfake) of the [deepfake-detection-challenge](https://www.kaggle.com/c/deepfake-detection-challenge) dataset.
