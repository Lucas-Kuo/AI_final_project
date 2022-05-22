# inport necessary packages
import os
import pandas as pd

# initialize the path to the *original* input directory of images
ORIG_INPUT_DATASET = "train_images_512"

TRAIN_LABEL_PATH = "train.csv"

# initialize the base path to the *new* directory that will contain
# our images after computing the training and validation split
BASE_PATH = "dataset"

# derive the training, validation, and testing directories
TRAIN_PATH = os.path.sep.join([BASE_PATH, "training"])
VAL_PATH = os.path.sep.join([BASE_PATH, "validation"])
TEST_PATH = "test_images_512"

# initialize the list of class label names
dataframe = pd.read_csv(TRAIN_LABEL_PATH) # has form of (image_name, species, unique_id)
CLASSES = list(dataframe.groupby("species").groups.keys()) # has 30 classes

# build class directories for training and validation datasets
for split in (TRAIN_PATH, VAL_PATH):
    for label in CLASSES:
        label_directory = os.path.sep.join([split, label])
        if not os.path.exists(label_directory):
            print("[INFO] creating '{}' directory".format(label_directory))
            os.makedirs(label_directory)

# set the image size and shape
IMG_SIZE = (512, 512)
IMG_SHAPE = IMG_SIZE + (3, )

# set the batch size
BATCH_SIZE = 32

# initialize our number of epochs, initial learning rate
NUM_EPOCHS = 30
INIT_LR = 1e-3
