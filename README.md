# AI_final_project
A repository of code for Kaggle competition: https://www.kaggle.com/competitions/happy-whale-and-dolphin

## Overview
In this project, we're focusing on a object classification task.
We are going to classify images into 30 possible classes.
Additionally, there can exist objects not including in the 30 classes.
We'll also proposed a method to deal with such difficulty.

## Environment Setting
The `requirements.txt` file lists all Python libraries and versions that your local
environment should be equipped with, and they will be installed using:

```
pip install -r requirements.txt
```

## Usage
First, we should download the resized dataset from Kaggle from the [website](https://www.kaggle.com/datasets/bryanb/happy-whale-and-dolphin-resized) or using:

```
kaggle datasets download -d bryanb/happy-whale-and-dolphin-resized --unzip
```

Then, we're going to build the dataset for our tf.data pipeline using:

```
python3 build_dataset.py
```

Finally, we can train our model with some parameters:

```
python3 train.py --model {vgg/res} --trainModel {top/full}
```
The model parameter indicates the base model to choose.

The trainModel parameter controlls whether to freeze the base model in order to implement transfer learning.

## Hyperparameter Setting
You can set the hyperparameter in [config.py](https://github.com/Lucas-Kuo/AI_final_project/blob/main/config.py).
Initially,
-  the image size is 100x100 on RGB channel
-  batch size is set to be 32
-  validation split is 10% from the entire training dataset
-  initial learning rate is $10^{-3}$
-  number of epochs to train can be set as you want, initially set to 30

## Experiment Results
