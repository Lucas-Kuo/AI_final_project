from tensorflow.keras.applications.efficientnet import EfficientNetB7
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import TopKCategoricalAccuracy, Accuracy
from tensorflow import keras

import tensorflow as tf
from arcface import ArcFace
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import config
import pipeline
import argparse

# plot out the training result, including loss(cross entropy) and accuracy
def plot_graph(history, model):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    
    top_5_acc = history.history['top_5_categorical_accuracy']
    val_top_5_acc = history.history['val_top_5_categorical_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Top 1 Accuracy')
    plt.plot(val_acc, label='Validation Top 1 Accuracy')
    plt.plot(top_5_acc, label='Training Top 5 Accuracy')
    plt.plot(val_top_5_acc, label='Validation Top 5 Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, max(plt.ylim())*1.1])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig(f"{model}_training_curve.jpg")
    plt.show()

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--trainModel", required=True,
	help="train only top layer or full layer (either top or full)")
args = vars(ap.parse_args())

# load training dataset
trainDS = pipeline.load_dataset(subset="training")

# load validation dataset
valDS = pipeline.load_dataset(subset="validation")

model = EfficientNetB7(
    include_top=True,
    weights=None,
    input_tensor=Input(shape=config.IMG_SHAPE),
    classes=len(config.CLASSES),
)

previous_weights = 'effarc_weights.h5'
if os.path.exists(previous_weights):
    print("[INFO] automatically loading previous weights:", previous_weights)
    model = load_model(previous_weights)
else:
    print(f'[INFO] no previous weights detected for effarc model, will train from scratch...')


# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
if args["trainModel"]=="top":
    print("[INFO] freezing base model...")
    for layer in model.layers:
        layer.trainable = False
elif args["trainModel"]=="full":
    print("[INFO] training the whole model...")
else:
    raise ValueError(f'Please specify which part of the model to train ({args["trainModel"]} was given)')


print("[INFO] compiling model...")
opt = Adam(learning_rate=config.INIT_LR) if args["trainModel"]=="top" else Adam(learning_rate=config.INIT_LR)
loss = CategoricalCrossentropy(name='categorical_crossentropy')
acc = Accuracy(name='accuracy')
top5acc = TopKCategoricalAccuracy(k=5, name='top_5_categorical_accuracy')

model.compile(loss=loss, optimizer=opt, metrics=[acc, top5acc])
model.summary()

checkpoint_callback = ModelCheckpoint(
    previous_weights,
    monitor="val_accuracy",
    save_best_only=True,
)

# model training with checkpoint saving
print("[INFO] training model...")
history = model.fit(
    x=trainDS,
    validation_data=valDS,
    epochs=config.NUM_EPOCHS,
    callbacks=[checkpoint_callback],
    verbose=1
)

# dumping training history
print("[INFO] saving training history...")
history_filename = 'effarc_history.json'
json.dump(history.history, open(history_filename, 'w'))

# plotting training curves
print("[INFO] plotting training curves...")
plot_graph(history, "effarc")
