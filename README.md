# AI_final_project
A repository of code for Kaggle competition: https://www.kaggle.com/competitions/happy-whale-and-dolphin

## Overview
In this project, we're focusing on a object classification task.
We are going to classify images into 30 possible classes.
Additionally, there can exist objects not including in the 30 classes.
We'll also proposed a method to deal with such difficulty.

## Environment Setting
The `requirements.txt` file lists all Python libraries and versions that your local
environment should be equipped with and they will be installed using:

```
$ pip3 install -r requirements.txt
```

## Usage
First, we should download the resized dataset from Kaggle from the [website](https://www.kaggle.com/datasets/bryanb/happy-whale-and-dolphin-resized) or using:

```
$ kaggle datasets download -d bryanb/happy-whale-and-dolphin-resized --unzip
```

Also, download the training labels from Kaggle competition with:

```
$ kaggle competitions download happy-whale-and-dolphin -f train.csv

$ unzip train.csv.zip
```
Be sure that you have set your Kaggle account and personal API key (kaggle.json) before you use any Kaggle API.
For more information, please refer to the [official documentation](https://github.com/Kaggle/kaggle-api).
However, it is also possible for you to download train.csv directly from the [original competition datasest](https://www.kaggle.com/competitions/happy-whale-and-dolphin/data?select=train.csv).

Then, we're going to build the dataset for our tf.data pipeline using:

```
$ python3 build_dataset.py
```
Make sure you have the following data structure (including but not limited to):

```
.
├── dataset
│   ├── training [30 entries exceeds filelimit, not opening dir]
│   └── validation [30 entries exceeds filelimit, not opening dir]
├── train_images_128 [51034 entries exceeds filelimit, not opening dir]
├── arcface.py
├── build_dataset.py
├── config.py
├── pipeline.py
├── README.md
├── train2.py
├── train.csv
├── train.csv.zip
└── train.py
```

Finally, we can train our model with some parameters:

```
$ python3 train.py --model {vgg/res} --trainModel {top/full}
```
The model parameter indicates the base model to choose.

The trainModel parameter controls whether to freeze the base model in order to implement transfer learning.

For further usage, it is able to build a KNN classifier using:

```
$ python3 build_knn.py --model {vgg/res}
```
A knn.pickled file will be generated. 

To get the classifier in python, you can use the code snippet below:

```python
import pickle

with open('knn.pickle', 'rb') as f:
    neigh = pickle.load(f)
```
Then, you can get the KNN result using:
```python
model = tf.keras.models.load_model("PREVIOUS_WEIGHT.h5")
model = tf.keras.models.Model(inputs=model.input, outputs=model.layers[-3].output)
features = model.predict(trainDS)
predictions = neigh.predict_proba(features)
```

## Hyperparameter Setting
You can set the hyperparameter in [config.py](https://github.com/Lucas-Kuo/AI_final_project/blob/main/config.py).
Initially,
-  the image size is 100x100 on RGB channel
-  batch size is set to be 32
-  validation split is 10% from the entire training dataset
-  initial learning rate is $10^{-3}$
-  number of epochs to train can be set as you want, initially set to 30

## Experiment Results

| Model    | Top-1 Accuracy | Top-5 Accuracy |
|----------|----------------|----------------|
| Baseline | 0.0011         | 0.7026         |
| ResNet50 | 0.0013         | 0.7358         |


