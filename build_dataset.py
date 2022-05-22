# import necessary packages
import config
from imutils import paths
import shutil
import random
import os

# grab the paths to all input images
# in the original input directory and shuffle them
imagePaths = list(paths.list_images(config.ORIG_INPUT_DATASET))

random.seed(42)
random.shuffle(imagePaths)

# we'll be using part of the training data for validation
i = int(len(imagePaths) * config.VAL_SPLIT)
val_Paths = imagePaths[:i]
train_Paths = imagePaths[i:]

# define the datasets that we'll be building
datasets = [
    ("training", train_Paths, config.TRAIN_PATH),
    ("validation", val_Paths, config.VAL_PATH),
]

# loop over the datasets
for (dType, imagePaths, baseOutput) in datasets:
    # show which data split we are creating
    print("[INFO] building '{}' split".format(dType))

    # if the output base output directory does not exist, create it
    if not os.path.exists(baseOutput):
        print("[INFO] creating '{}' directory".format(baseOutput))
        os.makedirs(baseOutput)

    # loop over the input image paths
    for imagePath in imagePaths:

        # extract the filename of the input image
        filename = imagePath.split(os.path.sep)[-1]

        # check the label of this image
        label_name = config.FILENAME_CLASS_PAIR[filename]

        # build the path to the label directory
        label_directory = os.path.sep.join([baseOutput, label_name])

        # if the label output directory does not exist, create it
        if not os.path.exists(label_directory):
            print("[INFO] creating '{}' directory".format(label_directory))
            os.makedirs(label_directory)

        # construct the path to the destination image and then copy
        # the image itself
        p = os.path.sep.join([label_directory, filename])
        shutil.copy2(imagePath, p)
