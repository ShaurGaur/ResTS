import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, SGD
import sys
import glob
import argparse
from keras import __version__
from keras.applications.xception import Xception, preprocess_input
from keras.models import *
from keras.layers import *
from keras.activations import *
from keras.optimizers import SGD
from keras import optimizers
from keras import callbacks
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras.regularizers import l2, l1
import pandas as pd
import shutil

input_shape = (224, 224, 3)
nbr_of_classes = 38

# Use the same directory for both train and validation
DATASET_DIR = "../PlantVillage-Dataset/raw/segmented"
BATCH_SIZE = 16
SEED = 123
test_SPLIT = 0.2

DATASPLIT_DIR = "../PlantVillage-Dataset/lmdb/segmented-80-20"
TRAIN_TXT = f"{DATASPLIT_DIR}/train.txt"
TEST_TXT = f"{DATASPLIT_DIR}/test.txt"
LABELS_TXT = f"{DATASPLIT_DIR}/labels.txt"

# # Load training and validation datasets
# train_ds = tf.keras.utils.image_dataset_from_directory(
#     DATASET_DIR,
#     validation_split=test_SPLIT,
#     subset="training",
#     seed=SEED,
#     image_size=(224, 224),
#     batch_size=BATCH_SIZE,
#     label_mode="categorical",
# )

# test_ds = tf.keras.utils.image_dataset_from_directory(
#     DATASET_DIR,
#     validation_split=test_SPLIT,
#     subset="validation",
#     seed=SEED,
#     image_size=(224, 224),
#     batch_size=BATCH_SIZE,
#     label_mode="categorical",
# )

def load_image(file_path):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)  # or decode_png if needed
    image = tf.image.resize(image, [224, 224])  # resize as needed
    return image

def load_dataset(file_txt=TEST_TXT, labels_txt=LABELS_TXT):
    df = pd.read_csv(file_txt, sep='\t', header=None, names=['img_path', 'label_num'])
    file_paths = df['img_path'].tolist()
    labels = df['label_num'].tolist()

    # with open(labels_txt) as file:
    #     class_names = [line.rstrip() for line in file]
    # labels = [class_names[i] for i in label_idxs]
    
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    dataset = dataset.map(lambda x, y: (load_image(x), tf.one_hot(y, depth=nbr_of_classes)))
    return dataset

# Apply preprocessing (e.g., rescaling to match preprocess_input)
def preprocess_and_duplicate_labels(x, y):
    x = preprocess_input(x)  # Apply model-specific preprocessing
    return x, {"out1": y, "out2": y}  # Duplicate labels for dual output heads

train_ds = load_dataset(TRAIN_TXT)
test_ds = load_dataset(TEST_TXT)

print("train: ", len(train_ds))
print("test: ", len(test_ds))

# TODO: attach seed to shuffle
train_ds = train_ds.shuffle(len(train_ds), seed=SEED, reshuffle_each_iteration=True).batch(BATCH_SIZE).map(preprocess_and_duplicate_labels)
test_ds = test_ds.shuffle(len(test_ds), seed=SEED, reshuffle_each_iteration=True).batch(BATCH_SIZE).map(preprocess_and_duplicate_labels)

# Optional: Add augmentation (only on training set)
data_augmentation = tf.keras.Sequential(
    [
        RandomRotation(factor=40 / 360, fill_mode="nearest"),
        RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode="nearest"),
    ]
)

train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

# Prefetch for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# Encoder Start
base_model1 = tf.keras.applications.Xception(
    include_top=False, weights="imagenet", input_shape=input_shape
)
x1_0 = base_model1.output
x1_0 = Flatten(name="Flatten1")(x1_0)
dense1 = Dense(256, name="fc1", activation="relu")(x1_0)
x = classif_out_encoder1 = Dense(38, name="out1", activation="softmax")(
    dense1
)  # Latent Representation / Bottleneck

# Get Xception's tensors for skip connection.
conv14 = base_model1.get_layer("block14_sepconv2_act").output
conv13 = base_model1.get_layer("block13_sepconv2_bn").output
conv12 = base_model1.get_layer("block12_sepconv3_bn").output
conv11 = base_model1.get_layer("block11_sepconv3_bn").output
conv10 = base_model1.get_layer("block10_sepconv3_bn").output
conv9 = base_model1.get_layer("block9_sepconv3_bn").output
conv8 = base_model1.get_layer("block8_sepconv3_bn").output
conv7 = base_model1.get_layer("block7_sepconv3_bn").output
conv6 = base_model1.get_layer("block6_sepconv3_bn").output
conv5 = base_model1.get_layer("block5_sepconv3_bn").output
conv4 = base_model1.get_layer("block4_sepconv2_bn").output
conv3 = base_model1.get_layer("block3_sepconv2_bn").output
conv2 = base_model1.get_layer("block2_sepconv2_bn").output
conv1 = base_model1.get_layer("block1_conv2_act").output

# Decoder Start
dense2 = Dense(256, activation="relu")(x)

x = Add(name="first_merge")([dense1, dense2])
x = Dense(7 * 7 * 2048)(x)
reshape1 = Reshape((7, 7, 2048))(x)

# BLOCK 1
x = SeparableConv2D(2048, (3, 3), padding="same", name="block14_start")(reshape1)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = concatenate([conv14, x], axis=3)
x = SeparableConv2D(1536, (3, 3), padding="same")(x)
x = BatchNormalization()(x)
x = c14 = Activation("relu")(x)

# BLOCK 2
x = UpSampling2D((2, 2))(x)
x = Activation("relu")(x)
x = SeparableConv2D(1024, (3, 3), padding="same", name="block13_start")(x)
x = BatchNormalization()(x)
x = concatenate([conv13, x], axis=3)
x = Activation("relu")(x)
x = SeparableConv2D(728, (3, 3), padding="same")(x)
x = BatchNormalization()(x)

c1314 = Conv2D(728, (1, 1))(UpSampling2D()(c14))
x = add1 = Add()([c1314, x])

# BLOCK 3
x = Activation("relu")(x)
x = SeparableConv2D(728, (3, 3), padding="same", name="blockmiddle_start")(x)
x = BatchNormalization()(x)
x = concatenate([conv12, x], axis=3)
x = Activation("relu")(x)
x = SeparableConv2D(728, (3, 3), padding="same")(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = SeparableConv2D(728, (3, 3), padding="same")(x)
x = BatchNormalization()(x)
x = add2 = Add()([add1, x])
# BLOCK 4
x = Activation("relu")(x)
x = SeparableConv2D(728, (3, 3), padding="same")(x)
x = BatchNormalization()(x)
x = concatenate([conv11, x], axis=3)
x = Activation("relu")(x)
x = SeparableConv2D(728, (3, 3), padding="same")(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = SeparableConv2D(728, (3, 3), padding="same")(x)
x = BatchNormalization()(x)
x = add3 = Add()([add2, x])
# BLOCK 5
x = Activation("relu")(x)
x = SeparableConv2D(728, (3, 3), padding="same")(x)
x = BatchNormalization()(x)
x = concatenate([conv10, x], axis=3)
x = Activation("relu")(x)
x = SeparableConv2D(728, (3, 3), padding="same")(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = SeparableConv2D(728, (3, 3), padding="same")(x)
x = BatchNormalization()(x)
x = add4 = Add()([add3, x])
# BLOCK 6
x = Activation("relu")(x)
x = SeparableConv2D(728, (3, 3), padding="same")(x)
x = BatchNormalization()(x)
x = concatenate([conv9, x], axis=3)
x = Activation("relu")(x)
x = SeparableConv2D(728, (3, 3), padding="same")(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = SeparableConv2D(728, (3, 3), padding="same")(x)
x = BatchNormalization()(x)
x = add5 = Add()([add4, x])
# BLOCK 7
x = Activation("relu")(x)
x = SeparableConv2D(728, (3, 3), padding="same")(x)
x = BatchNormalization()(x)
x = concatenate([conv8, x], axis=3)
x = Activation("relu")(x)
x = SeparableConv2D(728, (3, 3), padding="same")(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = SeparableConv2D(728, (3, 3), padding="same")(x)
x = BatchNormalization()(x)
x = add6 = Add()([add5, x])
# BLOCK 8
x = Activation("relu")(x)
x = SeparableConv2D(728, (3, 3), padding="same")(x)
x = BatchNormalization()(x)
x = concatenate([conv7, x], axis=3)
x = Activation("relu")(x)
x = SeparableConv2D(728, (3, 3), padding="same")(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = SeparableConv2D(728, (3, 3), padding="same")(x)
x = BatchNormalization()(x)
x = add7 = Add()([add6, x])
# BLOCK 9
x = Activation("relu")(x)
x = SeparableConv2D(728, (3, 3), padding="same")(x)
x = BatchNormalization()(x)
x = concatenate([conv6, x], axis=3)
x = Activation("relu")(x)
x = SeparableConv2D(728, (3, 3), padding="same")(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = SeparableConv2D(728, (3, 3), padding="same")(x)
x = BatchNormalization()(x)
x = add8 = Add()([add7, x])
# BLOCK 10
x = Activation("relu")(x)
x = SeparableConv2D(728, (3, 3), padding="same")(x)
x = BatchNormalization()(x)
x = concatenate([conv5, x], axis=3)
x = Activation("relu")(x)
x = SeparableConv2D(728, (3, 3), padding="same", name="blockmiddle_end")(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = SeparableConv2D(728, (3, 3), padding="same")(x)
x = BatchNormalization()(x)
x = add9 = Add()([add8, x])

# BLOCK 11
x = UpSampling2D((2, 2))(x)
x = Activation("relu")(x)
x = SeparableConv2D(728, (3, 3), padding="same", name="block4_start")(x)
x = BatchNormalization()(x)
x = concatenate([conv4, x], axis=3)
x = Activation("relu")(x)
x = SeparableConv2D(728, (3, 3), padding="same")(x)
x = BatchNormalization()(x)

c45 = Conv2D(728, (1, 1))(UpSampling2D()(add9))
x = add10 = Add()([c45, x])

# BLOCK 12
x = Conv2DTranspose(1, (3, 3), strides=(2, 2))(x)
x = Activation("relu")(x)
x = SeparableConv2D(256, (3, 3), padding="valid", name="block3_start")(x)
x = BatchNormalization()(x)
x = concatenate([conv3, x], axis=3)
x = Activation("relu")(x)
x = SeparableConv2D(256, (3, 3), padding="same")(x)
x = BatchNormalization()(x)

c34 = Conv2D(256, (3, 3), padding="valid")(
    Conv2DTranspose(1, (3, 3), strides=(2, 2))(add10)
)
x = add11 = Add()([c34, x])

# BLOCK 13
x = Conv2DTranspose(1, (3, 3), strides=(2, 2))(x)
x = Activation("relu")(x)
x = SeparableConv2D(128, (3, 3), padding="valid", name="block2_start")(x)
x = BatchNormalization()(x)
x = concatenate([conv2, x], axis=3)
x = Activation("relu")(x)
x = SeparableConv2D(128, (3, 3), padding="same")(x)
x = BatchNormalization()(x)

c23 = Conv2D(128, (3, 3), padding="valid")(
    Conv2DTranspose(1, (3, 3), strides=(2, 2))(add11)
)
x = add12 = Add()([c23, x])

# BLOCK 14
x = Conv2D(64, (3, 3), padding="same", name="block1_start")(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = concatenate([conv1, x], axis=3)
x = ZeroPadding2D()(x)
x = Conv2D(32, (3, 3), padding="same")(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = UpSampling2D()(x)
x = ZeroPadding2D()(x)

x = Conv2D(
    2,
    3,
    activation="relu",
    padding="same",
)(x)
mask = x = Conv2D(3, 1, activation="sigmoid", name="Mask")(x)

base_model2 = tf.keras.applications.Xception(
    include_top=False, weights="imagenet", input_shape=(224, 224, 3)
)
x2_0 = base_model2(mask)
x2_0 = Flatten(name="Flatten2")(x2_0)
x2_1 = Dense(256, name="fc2", activation="relu")(x2_0)
classif_out_encoder2 = Dense(nbr_of_classes, name="out2", activation="softmax")(x2_1)

# Create ResTS Model
model = Model(base_model1.input, [classif_out_encoder1, classif_out_encoder2])

# Compile the mode to use multi-task learning
losses = {"out1": "categorical_crossentropy", "out2": "categorical_crossentropy"}
alpha = 0.4
lossWeights = {"out1": alpha, "out2": (1.0 - alpha)}
model.compile(
    optimizer=optimizers.SGD(learning_rate=1e-4, momentum=0.9),
    loss=losses,
    loss_weights=lossWeights,
    metrics={"out1": "accuracy", "out2": "accuracy"},
)
model.summary()

nb_epoch = 15
history = model.fit(
    train_ds,
    steps_per_epoch=None,
    epochs=nb_epoch,
    validation_data=test_ds,
    validation_steps=None,
)

df = pd.DataFrame(history.history)
df.to_csv(f"ResTS{nb_epoch}epochs.csv")
try:
    model.save("ResTS.h5")
except:
    print("Check if the model has been saved!")
 