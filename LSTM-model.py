import os
import cv2
import shutil

import tensorflow as tf
import numpy as np
import pandas as pd
import tflearn

# set up directories
DATA_DIR = 'data'
IMG_SEQ_DIR = os.path.join(DATA_DIR, 'images-seq')
IMG_SEQ_TRAIN_DIR = os.path.join(IMG_SEQ_DIR, 'train')
IMG_SEQ_VALID_DIR = os.path.join(IMG_SEQ_DIR, 'valid')
ABNORMAL_IMG_SEQ_TRAIN_DIR = os.path.join(IMG_SEQ_TRAIN_DIR, 'abnormal')
NORMAL_IMG_SEQ_TRAIN_DIR = os.path.join(IMG_SEQ_TRAIN_DIR, 'normal')
ABNORMAL_IMG_SEQ_VALID_DIR = os.path.join(IMG_SEQ_VALID_DIR, 'abnormal')
NORMAL_IMG_SEQ_VALID_DIR = os.path.join(IMG_SEQ_VALID_DIR, 'normal')

checkpointFolder = './test-ckpt'

# read in cnn results
cnn_result = pd.read_csv(os.path.join(checkpointFolder, 'output.csv'))
cnn_result.head()

# prepare training and testing sets
## train
X_train = []
Y_train = []
for name in np.unique(cnn_result['img_name'].loc[cnn_result['data_type'] == 'train']):
  sample = []
  for t in range(0,31,5):
    sample.append(cnn_result.loc[(cnn_result['img_name'] == name) & (cnn_result['t'] == t),['softmax_0','softmax_1']].values)
  X_train.append(sample)
  Y_train.append([1 - np.unique(cnn_result.loc[cnn_result['img_name'] == name,'actual']), np.unique(cnn_result.loc[cnn_result['img_name'] == name,'actual'])])
X_train = np.squeeze(np.array(X_train))
Y_train = np.squeeze(np.array(Y_train))
## valid
X_valid = []
Y_valid = []
for name in np.unique(cnn_result['img_name'].loc[cnn_result['data_type'] == 'valid']):
  sample = []
  for t in range(0,31,5):
    sample.append(cnn_result.loc[(cnn_result['img_name'] == name) & (cnn_result['t'] == t),['softmax_0','softmax_1']].values)
  X_valid.append(sample)
  # Y_valid.append(np.unique(cnn_result.loc[cnn_result['img_name'] == name, 'actual']))
  Y_valid.append([1 - np.unique(cnn_result.loc[cnn_result['img_name'] == name,'actual']), np.unique(cnn_result.loc[cnn_result['img_name'] == name,'actual'])])
X_valid = np.squeeze(np.array(X_valid))
Y_valid = np.squeeze(np.array(Y_valid))
## test
X_test = []
Y_test = []
for name in np.unique(cnn_result['img_name'].loc[cnn_result['data_type'] == 'test']):
  sample = []
  for t in range(0,31,5):
    sample.append(cnn_result.loc[(cnn_result['img_name'] == name) & (cnn_result['t'] == t),['softmax_0','softmax_1']].values)
  X_test.append(sample)
  # Y_valid.append(np.unique(cnn_result.loc[cnn_result['img_name'] == name, 'actual']))
  Y_test.append([1 - np.unique(cnn_result.loc[cnn_result['img_name'] == name,'actual']), np.unique(cnn_result.loc[cnn_result['img_name'] == name,'actual'])])
X_test = np.squeeze(np.array(X_test))
Y_test = np.squeeze(np.array(Y_test))


# build LSTM model
frames = 7
input_size = 2
num_classes = 2
batch_size = 64
net = tflearn.input_data(shape=[None, frames, input_size])
net = tflearn.lstm(net, 256, dropout=0.2)
net = tflearn.fully_connected(net, num_classes, activation='softmax')
net = tflearn.regression(net, optimizer='adam',
                          loss='categorical_crossentropy', name='output1')
model = tflearn.DNN(net, tensorboard_verbose=0, best_checkpoint_path = 'model.tfl.ckpt')                                              
model_info = model.fit(X_train, Y_train, validation_set=(X_valid, Y_valid),
          show_metric=True, batch_size=batch_size,snapshot_step=100,
          snapshot_epoch = True, n_epoch=10)
model.evaluate(X_test, Y_test, batch_size = batch_size)

