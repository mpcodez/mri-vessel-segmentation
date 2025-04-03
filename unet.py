from PIL import Image
import tensorflow as tf
from keras import backend as K, losses
from keras.models import Model
from keras.optimizers import Adam
import argparse
from glob import glob
import numpy as np
from keras.layers import (
    Input, Conv2D, Conv2DTranspose, MaxPooling2D, Dropout, BatchNormalization, 
    LeakyReLU, concatenate, Lambda, add, ReLU
)
from random import randint
from aug_utils import random_augmentation

BATCH_SIZE = 16
INPUT_SHAPE = (64, 64)
SMOOTH = 1.
CARDINALITY = 32

# Custom activation function
def custom_activation(x):
    return K.relu(x, alpha=0.0, max_value=1)

# Focal loss function
def focal_loss(gamma=2., alpha=0.25):
    def loss_fn(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return loss_fn

# Utility layers
def add_common_layers(x):
    x = BatchNormalization()(x)
    return LeakyReLU()(x)

# Grouped convolution
def grouped_convolution(x, filters, strides):
    if CARDINALITY == 1:
        return Conv2D(filters, (3, 3), strides=strides, padding='same')(x)
    
    split_filters = filters // CARDINALITY
    groups = [
        Conv2D(split_filters, (3, 3), strides=strides, padding='same')(
            Lambda(lambda z: z[:, :, :, i * split_filters:(i + 1) * split_filters])(x)
        ) for i in range(CARDINALITY)
    ]
    return concatenate(groups)

# Residual block
def residual_block(x, in_channels, out_channels, strides=(1, 1), project_shortcut=False):
    shortcut = x
    x = Conv2D(in_channels, (1, 1), padding='same')(x)
    x = add_common_layers(x)
    x = grouped_convolution(x, in_channels, strides)
    x = add_common_layers(x)
    x = Conv2D(out_channels, (1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    
    if project_shortcut or strides != (1, 1):
        shortcut = Conv2D(out_channels, (1, 1), strides=strides, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
    
    x = add([shortcut, x])
    return LeakyReLU()(x)

# U-Net model
def build_unet(dropout_rate=0, activation=ReLU):
    inputs = Input((None, None, 3))
    
    def conv_block(x, filters):
        x = Dropout(dropout_rate)(activation()(Conv2D(filters, (3, 3), padding='same')(x)))
        x = Dropout(dropout_rate)(activation()(Conv2D(filters, (3, 3), padding='same')(x)))
        return add_common_layers(x)

    def up_block(x, skip, filters):
        x = concatenate([Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(x), skip], axis=3)
        x = add_common_layers(x)
        x = Dropout(dropout_rate)(activation()(Conv2D(filters, (3, 3), padding='same')(x)))
        x = Dropout(dropout_rate)(activation()(Conv2D(filters, (3, 3), padding='same')(x)))
        return residual_block(x, filters, filters)
    
    c1 = conv_block(inputs, 32)
    p1 = residual_block(MaxPooling2D((2, 2))(c1), 32, 32)
    c2 = conv_block(p1, 64)
    p2 = residual_block(MaxPooling2D((2, 2))(c2), 64, 64)
    c3 = conv_block(p2, 128)
    p3 = residual_block(MaxPooling2D((2, 2))(c3), 128, 128)
    c4 = conv_block(p3, 256)
    p4 = residual_block(MaxPooling2D((2, 2))(c4), 256, 256)
    
    c5 = residual_block(conv_block(p4, 512), 512, 512)
    
    u6 = up_block(c5, c4, 256)
    u7 = up_block(u6, c3, 128)
    u8 = up_block(u7, c2, 64)
    u9 = up_block(u8, c1, 32)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(u9)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(1e-3), loss=losses.binary_crossentropy, metrics=['accuracy'])
    return model

# Image processing functions
def read_image(path):
    return np.array(Image.open(path)) / 255.

def read_mask(path):
    return np.array(Image.open(path)) / 255.0[..., np.newaxis]

def random_crop(img, mask, crop_size=INPUT_SHAPE[0]):
    h, w = img.shape[:2]
    i, j = randint(0, h - crop_size), randint(0, w - crop_size)
    return img[i:i+crop_size, j:j+crop_size], mask[i:i+crop_size, j:j+crop_size]

# Data generator
def data_generator(data, augment=False):
    while True:
        indices = np.random.choice(len(data), BATCH_SIZE // 4)
        images = [read_image(data[i][0]) for i in indices]
        masks = [read_mask(data[i][1]) for i in indices]
        
        if augment:
            images, masks = zip(*[random_augmentation(img, mask) for img, mask in zip(images, masks)])
        
        yield np.array(images), np.array(masks)
