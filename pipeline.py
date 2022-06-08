from tensorflow.data import AUTOTUNE
from imutils import paths
import tensorflow as tf
import numpy as np
import os
import config


def load_images(imagePath):
    # read the image from disk, decode it, resize it, and scale the
    # pixels intensities to the range [0, 1]
    image = tf.io.read_file(imagePath)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, config.IMG_SIZE) / 255.0
    # grab the label and encode it
    label = tf.strings.split(imagePath, os.path.sep)[-2]
    oneHot = label == config.CLASSES # [0, 0, ..., 0, 1, 0, ..., 0]
    encodedLabel = tf.argmax(oneHot)
    # return the image and the integer encoded label
    return (image, encodedLabel)

def to_double_input(image, label):
    return ((image, label), label)

def load_images_test(imagePath):
    # read the image from disk, decode it, resize it, and scale the
	# pixels intensities to the range [0, 1]
    image = tf.io.read_file(imagePath)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, config.IMG_SIZE) / 255.0
    # return the image and the integer encoded label
    return (image, None)

@tf.function
def augment(image, label):
    # perform random horizontal and vertical flips
# 	image = tf.image.random_flip_up_down(image) # This doesn't make sense for this task
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.25)
    # return the image and the label
    return (image, label)

def load_dataset(subset="training"): # for training and validation
    print(f"[INFO] loading image paths for {subset} subset...")
    if subset=="training":
        imagePaths = list(paths.list_images(config.TRAIN_PATH))
    elif subset=="validation":
        imagePaths = list(paths.list_images(config.VAL_PATH))
    else:
        raise ValueError(f"Parameter \'subset\' accepts either \'training\' or \'validation\', but \'{subset}\' was given.")

    print("[INFO] creating a tf.data input pipeline..")
    dataset = tf.data.Dataset.from_tensor_slices(imagePaths)
    if subset=="training":
        dataset = (dataset
            .shuffle(len(imagePaths))
            .map(load_images, num_parallel_calls=AUTOTUNE)
            .map(augment, num_parallel_calls=AUTOTUNE)
            .cache()
            .batch(config.BATCH_SIZE)
            .prefetch(AUTOTUNE)
        )
    elif subset=="validation":
        dataset = (dataset
            .map(load_images, num_parallel_calls=AUTOTUNE)
            .cache()
            .batch(config.BATCH_SIZE)
            .prefetch(AUTOTUNE)
        )
    return dataset

def load_dataset_arc(subset="training"): # for arcface top
    print(f"[INFO] loading image paths for {subset} subset...")
    if subset=="training":
        imagePaths = list(paths.list_images(config.TRAIN_PATH))
    elif subset=="validation":
        imagePaths = list(paths.list_images(config.VAL_PATH))
    else:
        raise ValueError(f"Parameter \'subset\' accepts either \'training\' or \'validation\', but \'{subset}\' was given.")

    print("[INFO] creating a tf.data input pipeline..")
    dataset = tf.data.Dataset.from_tensor_slices(imagePaths)
    if subset=="training":
        dataset = (dataset
            .shuffle(len(imagePaths))
            .map(load_images, num_parallel_calls=AUTOTUNE)
            .map(augment, num_parallel_calls=AUTOTUNE)
            .map(to_double_input, num_parallel_calls=AUTOTUNE)
            .cache()
            .batch(config.BATCH_SIZE)
            .prefetch(AUTOTUNE)
        )
    elif subset=="validation":
        dataset = (dataset
            .map(load_images, num_parallel_calls=AUTOTUNE)
            .map(to_double_input, num_parallel_calls=AUTOTUNE)
            .cache()
            .batch(config.BATCH_SIZE)
            .prefetch(AUTOTUNE)
        )
    return dataset


def load_test_dataset():
    print(f"[INFO] loading testing data...")
    imagePaths = list(paths.list_images(config.TEST_PATH))
    print("[INFO] creating a tf.data input pipeline...")
    dataset = tf.data.Dataset.from_tensor_slices(imagePaths)
    dataset = (dataset
        .map(load_images_test, num_parallel_calls=AUTOTUNE)
        .cache()
        .batch(config.BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )
    return dataset
