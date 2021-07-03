![](/assets/deepfake-shield-banner-wide.png)
[![](https://img.shields.io/badge/heroku-deployed-green)](https://deepfake-shield.herokuapp.com/)

## 1. :performing_arts: What is a Deepfake?
The term “[Deepfake](https://en.wikipedia.org/wiki/Deepfake)” is referred to a deep learning based technique that swaps the face of a person with another face in an image.

## 2. :detective: The Problem 
It is easier than ever to create deepfakes of anyone using the tools available online. Deepfakes can be used by people with evil intentions like generating fake news, hoaxes, blackmailing and financial fraud.

## 3. :dart: Our Solution
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

The model was trained using the ideal hyperparameters found using grid search. The training and evaluatoion logs were tracked using `mlrun`

![](/assets/mlrun_util_train.png)

## 📦 Running locally

The webapp can be run locally with the following steps:
1. Clone the repo and naviigate into the folder 

  ```
  git clone https://github.com/Mainakdeb/deepfake-shield.git`
  ```

```
cd deepfake-shield
```

2. Install requirements (`venv` recommended)

```
pip install -r requirements.txt
```

3. Run webapp on localhost

```
python3 app.py
```

## 📗 Resources
We have published the notebooks that we used to train our model, refer to the links below.

* **Data exploration**: - [NBViewer](https://nbviewer.jupyter.org/github/Mainakdeb/deepfake-shield/blob/main/notebooks/preprocess_and_explore_data.ipynb), [Github](https://github.com/Mainakdeb/deepfake-shield/blob/main/notebooks/preprocess_and_explore_data.ipynb)
* **Preprocessing + Training + Evaluation**: - [NBViewer](https://nbviewer.jupyter.org/github/Mainakdeb/deepfake-shield/blob/main/notebooks/train_deep_shield_model.ipynb), [Github](https://github.com/Mainakdeb/deepfake-shield/blob/main/notebooks/train_deep_shield_model.ipynb)
* **Dataset**: We've used a [modified version](https://www.kaggle.com/unkownhihi/deepfake) of the [deepfake-detection-challenge](https://www.kaggle.com/c/deepfake-detection-challenge) dataset.
