import os
import cv2

# set up directories
ORIG_IMG_DIR = ''
HUONG_DIR = ''
TRAIN_DIR = os.path.join(HUONG_DIR,'train')
VALID_DIR = os.path.join(HUONG_DIR, 'valid')
TEST_DIR = os.path.join(HUONG_DIR, 'test')
ABNORMAL_TRAIN_DIR = os.path.join(TRAIN_DIR, 'abnormal')
NORMAL_TRAIN_DIR = os.path.join(TRAIN_DIR, 'normal')
ABNORMAL_VALID_DIR = os.path.join(VALID_DIR, 'abnormal')
NORMAL_VALID_DIR = os.path.join(VALID_DIR, 'normal')

TRAIN_TXT = 'binary_train.txt'
TEST_TXT = 'binary_test.txt'
VALID_TXT = 'binary_validate.txt'

# read train, valid, and test sets
file = open(TRAIN_TXT, 'r')
train_img, train_indicator = l.split(' ') for l in file.readlines()
file = open(VALID_TXT, 'r')
valid_img, valid_indicator = l.split(' ') for l in file.readlines()
file = open(TEST_TXT, 'r')
test_img, test_indicator = l.split(' ') for l in file.readlines()

# enhanced training img
os.makedirs(ABNORMAL_TRAIN_DIR, exist_ok = True)
os.makedirs(NORMAL_TRAIN_DIR, exist_ok = True)
for i in range(len(train_img)):
    file_name, file_tail = train_img[i].split('.')
    img = cv2.read(os.path.join(ORIG_IMG_DIR, train_img[i]))
    if train_indicator[i] == 1:
        cv2.imwrite(os.path.join(ABNORMAL_TRAIN_DIR,str(file_name) + '_0' + '.' + file_tail), img)
        for t in range(5,31,5):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(i,i))
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
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(i,i))
            # method: enhanced contrast paper
            opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            top = img - opening
            closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
            bottom = closing - img
            enhanced = img + top - bottom
            cv2.imwrite(os.path.join(NORMAL_TRAIN_DIR,str(file_name) + '_' + str(t) + '.' + file_tail), enhanced)

# enhanced validating img
os.makedirs(ABNORMAL_VALID_DIR, exist_ok = True)
os.makedirs(NORMAL_VALID_DIR, exist_ok = True)
for i in range(len(valid_img)):
    file_name, file_tail = valid_img[i].split('.')
    img = cv2.read(os.path.join(ORIG_IMG_DIR, valid_img[i]))
    if valid_indicator[i] == 1:
        cv2.imwrite(os.path.join(ABNORMAL_VALID_DIR,str(file_name) + '_0' + '.' + file_tail), img)
        for t in range(5,31,5):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(i,i))
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
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(i,i))
            # method: enhanced contrast paper
            opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            top = img - opening
            closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
            bottom = closing - img
            enhanced = img + top - bottom
            cv2.imwrite(os.path.join(NORMAL_VALID_DIR,str(file_name) + '_' + str(t) + '.' + file_tail), enhanced)

# enhancing test set
os.makedirs(TEST_TXT, exist_ok = True)
for f in test_img:
    file_name, file_tail = f[i].split('.')
    img = cv2.read(os.path.join(ORIG_IMG_DIR, f))
    cv2.imwrite(os.path.join(TEST_DIR,str(file_name) + '_0' + '.' + file_tail), img)
    for t in range(5,31,5):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(i,i))
        # method: enhanced contrast paper
        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        top = img - opening
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        bottom = closing - img
        enhanced = img + top - bottom
        cv2.imwrite(os.path.join(TEST_DIR,str(file_name) + '_' + str(t) + '.' + file_tail), enhanced)
