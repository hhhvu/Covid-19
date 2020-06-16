import os
import cv2
import shutil

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler, ModelCheckpoint
import tensorflow_hub as hub

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import json
import pandas as pd

# set up directories
CNN_DIR = 'CNN-model'
TRAIN_DIR = os.path.join(CNN_DIR,'train')
VALID_DIR = os.path.join(CNN_DIR,'valid')
TEST_DIR = os.path.join(CNN_DIR,'test')


# download inception_v3 model from tf
module_selection = ("inception_v3", 299) 
handle_base, pixels = module_selection
MODULE_HANDLE ="https://tfhub.dev/google/imagenet/{}/feature_vector/4".format(handle_base)
IMAGE_SIZE = (pixels, pixels)
print("Using {} with input size {}".format(MODULE_HANDLE, IMAGE_SIZE))
BATCH_SIZE = 64

# preparing training, validating and testing sets
datagen_kwargs = dict(rescale=1./255, validation_split=.20)
dataflow_kwargs = dict(target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
                   interpolation="bilinear")
## valid
valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    **datagen_kwargs)
valid_generator = valid_datagen.flow_from_directory(
    VALID_DIR, shuffle=False, **dataflow_kwargs)
## train
do_data_augmentation = False 
if do_data_augmentation:
  train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
      rotation_range=40,
      horizontal_flip=True,
      width_shift_range=0.2, height_shift_range=0.2,
      shear_range=0.2, zoom_range=0.2,
      **datagen_kwargs)
else:
  train_datagen = valid_datagen
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR, shuffle=True, **dataflow_kwargs)
do_fine_tuning = False 
## test 
test_datagen = valid_datagen
test_generator = test_datagen.flow_from_directory(
    TEST_DIR, shuffle = False, **dataflow_kwargs
)

# add checkpoint callback argument to save the best model
checkpointFolder = './test-ckpt'
if not os.path.exists(checkpointFolder):
    os.makedirs(checkpointFolder)
filepath=checkpointFolder+"/model-{epoch:02d}-{val_accuracy:.2f}.hdf5"

checkpoint_callback = ModelCheckpoint(
    filepath, monitor='val_accuracy', verbose=1,
    save_best_only=True, save_weights_only=False,
    save_frequency=1)

# build model
print("Building model with", MODULE_HANDLE)
model = tf.keras.Sequential([
    # Explicitly define the input shape so the model can be properly
    # loaded by the TFLiteConverter
    tf.keras.layers.InputLayer(input_shape=IMAGE_SIZE + (3,)),
    hub.KerasLayer(MODULE_HANDLE, trainable=do_fine_tuning),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(train_generator.num_classes,
                          kernel_regularizer=tf.keras.regularizers.l2(0.0001))
])
model.build((None,)+IMAGE_SIZE+(3,))

model.compile(
  optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9), 
  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
  metrics=['accuracy'])

# train model
steps_per_epoch = train_generator.samples // train_generator.batch_size
validation_steps = valid_generator.samples // valid_generator.batch_size
model_info = model.fit(
    train_generator,
    epochs=1, steps_per_epoch=steps_per_epoch,
    callbacks = [checkpoint_callback],
    validation_freq=1,
    validation_data=valid_generator,
    validation_steps=validation_steps)

# get training and validating logits
X_train_logits = model.predict(train_generator, verbose = 1)
X_valid_logits = model.predict(valid_generator, verbose = 1)
X_test_logits = model.predict(test_generator, verbose = 1)
## softmax logit
X_train_softmax = tf.nn.softmax(X_train_logits)
X_valid_softmax = tf.nn.softmax(X_valid_logits)
X_test_softmax = tf.nn.softmax(X_test_logits)   
# save results    
data = pd.DataFrame({'data_type': 'train',
                     'file_name': train_generator.filenames,
                     'img_name': [f.split('/')[1].split('_')[0] + '_' + f.split('/')[1].split('_')[1] for f in train_generator.filenames],
                     't': [f.split('/')[1].split('_')[2].split('.')[0] for f in train_generator.filenames],
                     'logits_0': X_train_logits[:,0],
                     'logits_1': X_train_logits[:,1],
                     'softmax_0': X_train_softmax[:,0],
                     'softmax_1': X_train_softmax[:,1],
                     'actual': train_generator.classes})
data = data.append(pd.DataFrame({'data_type': 'valid',
                                'file_name': valid_generator.filenames,
                                'img_name': [f.split('/')[1].split('_')[0] + '_' + f.split('/')[1].split('_')[1] for f in valid_generator.filenames],
                                't': [f.split('/')[1].split('_')[2].split('.')[0] for f in valid_generator.filenames],
                                'logits_0': X_valid_logits[:,0],
                                'logits_1': X_valid_logits[:,1],
                                'softmax_0': X_valid_softmax[:,0],
                                'softmax_1': X_valid_softmax[:,1],
                                'actual': valid_generator.classes}))
data = data.append(pd.DataFrame({'data_type': 'test',
                                'file_name': test_generator.filenames,
                                'img_name': [f.split('/')[1].split('_')[0] + '_' + f.split('/')[1].split('_')[1] for f in test_generator.filenames],
                                't': [f.split('/')[1].split('_')[2].split('.')[0] for f in test_generator.filenames],
                                'logits_0': X_test_logits[:,0],
                                'logits_1': X_test_logits[:,1],
                                'softmax_0': X_test_softmax[:,0],
                                'softmax_1': X_test_softmax[:,1],
                                'actual': test_generator.classes}))
data.to_csv(os.path.join(CNN_DIR), 'output.csv'), encoding='utf-8', index=False)
