import os
import cv2
from math import *
import logging
import datetime

logging.basicConfig(level=logging.INFO)

# set up directories
ORIG_IMG_DIR = '../../dataset/images'
# CNN_DIR = 'CNN-model'
# TRAIN_DIR = os.path.join(CNN_DIR,'train')
# VALID_DIR = os.path.join(CNN_DIR, 'valid')
# TEST_DIR = os.path.join(CNN_DIR, 'test')
# ABNORMAL_TRAIN_DIR = os.path.join(TRAIN_DIR, 'abnormal')
# NORMAL_TRAIN_DIR = os.path.join(TRAIN_DIR, 'normal')
# ABNORMAL_VALID_DIR = os.path.join(VALID_DIR, 'abnormal')
# NORMAL_VALID_DIR = os.path.join(VALID_DIR, 'normal')
# ABNORMAL_TEST_DIR = os.path.join(TEST_DIR, 'abnormal')
# NORMAL_TEST_DIR = os.path.join(TEST_DIR, 'normal')
DATA_DIR = os.path.join('..','data')
os.makedirs(DATA_DIR, exist_ok = True)
TRAIN_DIR = os.path.join(DATA_DIR,'train')
VALID_DIR = os.path.join(DATA_DIR,'valid')
TEST_DIR = os.path.join(DATA_DIR,'test')
ABNORMAL_TRAIN_DIR = os.path.join(TRAIN_DIR, 'abnormal')
NORMAL_TRAIN_DIR = os.path.join(TRAIN_DIR, 'normal')
ABNORMAL_VALID_DIR = os.path.join(VALID_DIR, 'abnormal')
NORMAL_VALID_DIR = os.path.join(VALID_DIR, 'normal')
ABNORMAL_TEST_DIR = os.path.join(TEST_DIR, 'abnormal')
NORMAL_TEST_DIR = os.path.join(TEST_DIR, 'normal')

TRAIN_TXT = '../../Multitask-Learning-CXR/dataset/binary_train_small.txt'
TEST_TXT = '../../Multitask-Learning-CXR/dataset/binary_test_small.txt'
VALID_TXT = '../../Multitask-Learning-CXR/dataset/binary_validate_small.txt'

# read train, valid, and test sets
## train
train_img = []
train_indicator = []
file = open(TRAIN_TXT, 'r')
for l in file.readlines():
    img, indicator = l.split(' ')
    train_img.append(img)
    train_indicator.append(indicator)
## valid
valid_img = []
valid_indicator = []
file = open(VALID_TXT, 'r')
for l in file.readlines():
    img, indicator = l.split(' ')
    valid_img.append(img)
    valid_indicator.append(indicator)
## test
test_img = []
test_indicator = []
file = open(TEST_TXT, 'r')
for l in file.readlines():
    img, indicator = l.split(' ')
    test_img.append(img)
    test_indicator.append(indicator)

cnt = 0
# enhanced training img
os.makedirs(ABNORMAL_TRAIN_DIR, exist_ok = True)
os.makedirs(NORMAL_TRAIN_DIR, exist_ok = True)
logging.info(str(datetime.datetime.now()) + ' Enhancing training set')
for i in range(len(train_img)):
    if (cnt % 1000) == 0:
        logging.info(str(datetime.datetime.now()) + ' ' + str(cnt))
    file_name, file_tail = train_img[i].split('.')
    img = cv2.imread(os.path.join(ORIG_IMG_DIR, train_img[i]))
    img = cv2.resize(img, (299,299))
    if train_indicator[i] == 1:
        cv2.imwrite(os.path.join(ABNORMAL_TRAIN_DIR,str(file_name) + '_0' + '.' + file_tail), img)
        for t in range(5,31,5):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(t,t))
            # method: enhanced contrast paper
            opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            top = img - opening
            closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
            bottom = closing - img
            enhanced = img + top - bottom
            cv2.imwrite(os.path.join(ABNORMAL_TRAIN_DIR,str(file_name) + '_' + str(t) + '.' + file_tail), enhanced)
    else:
        cv2.imwrite(os.path.join(NORMAL_TRAIN_DIR,str(file_name) + '_0' + '.' + file_tail), img)
        for t in range(5,31,5):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(t,t))
            # method: enhanced contrast paper
            opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            top = img - opening
            closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
            bottom = closing - img
            enhanced = img + top - bottom
            cv2.imwrite(os.path.join(NORMAL_TRAIN_DIR,str(file_name) + '_' + str(t) + '.' + file_tail), enhanced)
    cnt += 1

# enhanced validating img
os.makedirs(ABNORMAL_VALID_DIR, exist_ok = True)
os.makedirs(NORMAL_VALID_DIR, exist_ok = True)
for i in range(len(valid_img)):
    if (cnt % 1000) == 0:
        logging.info(str(datetime.datetime.now()) + ' ' + str(cnt))
    file_name, file_tail = valid_img[i].split('.')
    img = cv2.imread(os.path.join(ORIG_IMG_DIR, valid_img[i]))
    img = cv2.resize(img, (299,299))
    if valid_indicator[i] == 1:
        cv2.imwrite(os.path.join(ABNORMAL_VALID_DIR,str(file_name) + '_0' + '.' + file_tail), img)
        for t in range(5,31,5):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(t,t))
            # method: enhanced contrast paper
            opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            top = img - opening
            closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
            bottom = closing - img
            enhanced = img + top - bottom
            cv2.imwrite(os.path.join(ABNORMAL_VALID_DIR,str(file_name) + '_' + str(t) + '.' + file_tail), enhanced)
    else:
        cv2.imwrite(os.path.join(NORMAL_VALID_DIR,str(file_name) + '_0' + '.' + file_tail), img)
        for t in range(5,31,5):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(t,t))
            # method: enhanced contrast paper
            opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            top = img - opening
            closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
            bottom = closing - img
            enhanced = img + top - bottom
            cv2.imwrite(os.path.join(NORMAL_VALID_DIR,str(file_name) + '_' + str(t) + '.' + file_tail), enhanced)
    cnt += 1
# enhancing test set
os.makedirs(ABNORMAL_TEST_DIR, exist_ok = True)
os.makedirs(NORMAL_TEST_DIR, exist_ok = True)
for i in range(len(test_img)):
    if (cnt % 1000) == 0:
        logging.info(str(datetime.datetime.now()) + ' ' + str(cnt))
    file_name, file_tail = test_img[i].split('.')
    img = cv2.imread(os.path.join(ORIG_IMG_DIR, test_img[i]))
    img = cv2.resize(img, (299,299))
    if test_indicator[i] == 1:
        cv2.imwrite(os.path.join(ABNORMAL_TEST_DIR,str(file_name) + '_0' + '.' + file_tail), img)
        for t in range(5,31,5):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(t,t))
            # method: enhanced contrast paper
            opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            top = img - opening
            closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
            bottom = closing - img
            enhanced = img + top - bottom
            cv2.imwrite(os.path.join(ABNORMAL_TEST_DIR,str(file_name) + '_' + str(t) + '.' + file_tail), enhanced)
    else:
        cv2.imwrite(os.path.join(NORMAL_TEST_DIR,str(file_name) + '_0' + '.' + file_tail), img)
        for t in range(5,31,5):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(t,t))
            # method: enhanced contrast paper
            opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            top = img - opening
            closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
            bottom = closing - img
            enhanced = img + top - bottom
            cv2.imwrite(os.path.join(NORMAL_TEST_DIR,str(file_name) + '_' + str(t) + '.' + file_tail), enhanced)
    cnt += 1
