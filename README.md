[![Build Status](https://travis-ci.com/italoPontes/fraud_detection.svg?token=xCXQ5y8dztyVs3aHPJLA&branch=master)](https://travis-ci.com/italoPontes/fraud_detection)

# SpaceYNet
> Simple project description.

A Approach to Pose and Depth-Scene Regression Simultaneously Using Neural Networks.


## Stakeholder

| Role                 | Responsibility         | Full name                | e-mail       |
| -----                | ----------------       | -----------              | ---------    |
| Data Scientist       | Author                 | Dunfrey P. Aragão | dunfrey@gmail.com   |


## Usage
Clone the repository:
```
git clone https://github.com/<@github_username>/spaceynet.git
cd spaceynet
```


#### Python

The project was written in Python 3, and work with later as well.
Also, please read up the subsequent libraries that are used: [Tensorflow](https://www.tensorflow.org/), [Matplotlib](https://matplotlib.org/), [OpenCV](https://opencv.org/), [Scikit](https://scikit-learn.org/stable/) and [Numpy](https://numpy.org/).


### Running

The project does not need user interaction of information in time execution.
To run the whole workflow of the project, it is possible by the following command:

```
$ python main.py \
    train \
    --path_data_train ../dataset/laser/ \
    --output_path ../output/
```

* **train**: SpaceYNet train method.
* **path_data_train**: training dataset to work on the training step.
* **output_path**: outcome from the classification model using validation data.

