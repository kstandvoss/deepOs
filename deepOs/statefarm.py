import numpy as np

import os
import glob
import math
import pickle
import datetime
import pandas as pd
import theano
import theano.tensor as T

from sklearn.cross_validation import train_test_split
from keras.utils import np_utils
from keras.models import model_from_json
from scipy.misc import imread, imresize


class Statefarm:

    def __init__(self, 
                 use_cache=1, 
                 path_data="/net/store/ni/projects/deeplearning/statefarm", 
                 path_cache="/net/store/ni/projects/deeplearning/statefarm/cache"):
        self.use_cache = use_cache
        self.path_data = path_data
        self.path_cache = path_cache

    def get_im_skipy(self, path, img_rows, img_cols, color_type=1, interp='nearest'):
        # Load as grayscale
        if color_type == 1:
            img = imread(path, True)
        elif color_type == 3:
            img = imread(path)
        # Reduce size
        resized = imresize(img, (img_rows, img_cols), interp=interp)
        return resized

    def get_driver_data(self):
        dr = dict()
        path = os.path.join(self.path_data, 'driver_imgs_list.csv')
        print('Read drivers data')
        f = open(path, 'r')
        line = f.readline()
        while (1):
            line = f.readline()
            if line == '':
                break
            arr = line.strip().split(',')
            dr[arr[2]] = arr[0]
        f.close()
        return dr

    def load_train(self, img_rows, img_cols, color_type=1, interp='nearest'):
        X_train = []
        y_train = []
        driver_id = []

        driver_data = self.get_driver_data()

        print('Read train images')
        for j in range(10):
            print('Load folder c{}'.format(j))
            path = os.path.join(self.path_data, 'train', 'c' + str(j), '*.jpg')
            files = glob.glob(path)
            for fl in files:
                flbase = os.path.basename(fl)
                img = self.get_im_skipy(fl, img_rows, img_cols, color_type, interp=interp)
                X_train.append(img)
                y_train.append(j)
                driver_id.append(driver_data[flbase])

        unique_drivers = sorted(list(set(driver_id)))
        print('Unique drivers: {}'.format(len(unique_drivers)))
        print(unique_drivers)
        return X_train, y_train, driver_id, unique_drivers

    def load_test(self, img_rows, img_cols, color_type=1, interp='nearest'):
        print('Read test images')
        path = os.path.join(self.path_data, 'test', '*.jpg')
        files = glob.glob(path)
        X_test = []
        X_test_id = []
        total = 0
        thr = math.floor(len(files)/10)
        for fl in files:
            flbase = os.path.basename(fl)
            img = self.get_im_skipy(fl, img_rows, img_cols, color_type, interp=interp)
            X_test.append(img)
            X_test_id.append(flbase)
            total += 1
            if total%thr == 0:
                print('Read {} images from {}'.format(total, len(files)))

        return X_test, X_test_id

    def cache_data(self, data, path, filename):
        print(path)
        if os.path.isdir(os.path.dirname(path)):
            filepath = os.path.join(path, filename)
            file = open(filepath, 'wb')
            pickle.dump(data, file)
            file.close()
        else:
            print('Directory doesnt exists')

    def restore_data(self, path, filename):
        data = dict()
        filepath = os.path.join(path, filename)
        if os.path.isfile(filepath):
            file = open(filepath, 'rb')
            data = pickle.load(file)
        return data

    def save_model(self, model):
        json_string = model.to_json()
        cache_path = os.path.join(self.path_cache)
        if not os.path.isdir(cache_path):
            os.mkdir(cache_path)
        open(os.path.join(cache_path, 'architecture.json'), 'w').write(json_string)
        model.save_weights(os.path.join(cache_path, 'model_weights.h5'), overwrite=True)

    def read_model(self):
        cache_path = os.path.join(self.path_cache)
        model = model_from_json(open(os.path.join(cache_path, 'architecture.json')).read())
        model.load_weights(os.path.join(cache_path, 'model_weights.h5'))
        return model

    def split_validation_set(self, train, target, test_size):
        random_state = 51
        X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test

    def create_submission(self, predictions, test_id, info):
        result1 = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
        result1.loc[:, 'img'] = pd.Series(test_id, index=result1.index)
        now = datetime.datetime.now()
        if not os.path.isdir('subm'):
            os.mkdir('subm')
        suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
        sub_file = os.path.join('subm', 'submission_' + suffix + '.csv')
        result1.to_csv(sub_file, index=False)

    def read_and_normalize_train_data(self, img_rows, img_cols, color_type=1, interp='nearest'):
        cache_path = os.path.join(self.path_cache)
        filename = 'train_r_' + str(img_rows) + '_c_' + str(img_cols) + '_t_' + str(color_type) + '_interp_' + interp + '.dat'
        if not os.path.isfile(os.path.join(cache_path, filename)) or self.use_cache == 0:
            train_data, train_target, driver_id, unique_drivers = self.load_train(img_rows, img_cols, color_type, interp)
            self.cache_data((train_data, train_target, driver_id, unique_drivers), cache_path, filename)
        else:
            #print('Restore train from cache!')
            (train_data, train_target, driver_id, unique_drivers) = self.restore_data(cache_path, filename)

        train_data = np.array(train_data, dtype=np.uint8)
        train_target = np.array(train_target, dtype=np.uint8)
        train_data = train_data.reshape(train_data.shape[0], color_type, img_rows, img_cols)
        train_target = np_utils.to_categorical(train_target, 10)
        train_data = train_data.astype('float32')
        train_data /= 255
        #print('Train shape:', train_data.shape)
        #print(train_data.shape[0], 'train samples')
        return train_data, train_target, driver_id, unique_drivers

    def read_and_normalize_test_data(self, img_rows, img_cols, color_type=1, interp='nearest'):
        cache_path = os.path.join(self.path_cache)
        filename = 'test_r_' + str(img_rows) + '_c_' + str(img_cols) + '_t_' + str(color_type) + '_interp_' + interp + '.dat'
        if not os.path.isfile(os.path.join(cache_path, filename)) or self.use_cache == 0:
            test_data, test_id = self.load_test(img_rows, img_cols, color_type, interp)
            self.cache_data((test_data, test_id), cache_path, filename)
        else:
            #print('Restore test from cache!')
            (test_data, test_id) = self.restore_data(cache_path, filename)

        test_data = np.array(test_data, dtype=np.uint8)
        test_data = test_data.reshape(test_data.shape[0], color_type, img_rows, img_cols)
        test_data = test_data.astype('float32')
        test_data /= 255
        #print('Test shape:', test_data.shape)
        #print(test_data.shape[0], 'test samples')
        return test_data, test_id

    def dict_to_list(d):
        ret = []
        for i in d.items():
            ret.append(i[1])
        return ret

    def merge_several_folds_mean(data, nfolds):
        a = np.array(data[0])
        for i in range(1, nfolds):
            a += np.array(data[i])
        a /= nfolds
        return a.tolist()

    def merge_several_folds_geom(data, nfolds):
        a = np.array(data[0])
        for i in range(1, nfolds):
            a *= np.array(data[i])
        a = np.power(a, 1/nfolds)
        return a.tolist()

    def copy_selected_drivers(self, train_data, train_target, driver_id, driver_list):
        data = []
        target = []
        index = []
        for i in range(len(driver_id)):
            if driver_id[i] in driver_list:
                data.append(train_data[i])
                target.append(train_target[i])
                index.append(i)
        data = np.array(data, dtype=np.float32)
        target = np.array(target, dtype=np.float32)
        index = np.array(index, dtype=np.uint32)
        return data, target, index

    def log_loss_objective(y_true, y_pred):
        y_pred = T.clip(y_pred, 1.0e-15, 1.0 - 1.0e-15)
        y_pred /= y_pred.sum(axis=-1, keepdims=True)
        cce = T.nnet.categorical_crossentropy(y_pred, y_true)
        return cce
    
    def get_compiled_log_loss():
        # compile loss function:
        y_true= theano.tensor.matrix()
        y_pred = theano.tensor.matrix()
        log_loss_objective_compiled = theano.function([y_true,y_pred], Statefarm.log_loss_objective(y_true,y_pred))
        return log_loss_objective_compiled
