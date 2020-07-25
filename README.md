# SpaceYNet
> 2020 International Conference on Systems, Signals and Image Processing (IWSSIP)

SpaceYNet: A Novel Approach to Pose and Depth-Scene Regression Simultaneously


## Stakeholder

| Role                 | Responsibility         | Full name                | e-mail       |
| -----                | ----------------       | -----------              | ---------    |
| Data Scientist       | Author                 | Dunfrey P. Aragão | dunfrey@gmail.com   |
| Advisor       | Advisor                 | Tiago Nascimento | tiagopn@ci.ufpb.br   |


#### Project Language

- Python 3
- [Tensorflow](https://www.tensorflow.org/)
- [Matplotlib](https://matplotlib.org/)
- [OpenCV](https://opencv.org/)
- [Scikit](https://scikit-learn.org/stable/)
- [Numpy](https://numpy.org/).


## Documentation

* [IEEExplore](https://ieeexplore.ieee.org/document/9145427): Paper IEEExplore link.

To run train project workflow, it is possible by the following command:

```
$ python main.py \
    train \
    --path_data_train ../dataset/laser/ \
    --output_path ../output/
```

#### Folder structure
>Explain you folder strucure
* **train**: SpaceYNet train method.
* **path_data_train**: training dataset to work on the training step.
* **output_path**: outcome from the classification model using validation data.
