from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.models import load_model, Model
import numpy as np
import os
import config
import pipeline
import pickle
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="base model to choose (either vgg or res)")

args = vars(ap.parse_args())

# load training dataset
trainDS = pipeline.load_dataset(subset="training")

# load original trained model
previous_weights = args["model"]+"_weights.h5"
model = load_model(previous_weights)

# feature extraction model
model = Model(inputs=model.input, outputs=model.layers[-3].output)

# perform feature extraction
features = model.predict(trainDS)

# label encoding
labels = np.argmax(trainDS, axis=-1)

# initialize KNN
neigh = KNeighborsClassifier(n_neighbors=55)

# fit the features and labels
neigh.fit(features, labels)

# save the KNN as pickle file
with open('knn.pickle', 'wb') as f:
    pickle.dump(neigh, f, protocol=pickle.HIGHEST_PROTOCOL)
