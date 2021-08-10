import numpy as np
import cv2
from PIL import Image
import time
import tensorflow as tf
from tensorflow.python.keras.utils.data_utils import Sequence
import config


class TFDataFeeder(Sequence):
    '''
    Use tf.data to load raw image data
    '''
    def __init__(self, tfdataset, batch_size, dataset_len, pair=False):
        # print('TFDataFeeder: batch size ', batch_size,
        #       ' dataset len', dataset_len, 'prefetch autotune')
        self.pair = pair
        self.batch_size = batch_size
        self.tfdataset = tfdataset.batch(self.batch_size)
        self.tfdataset = self.tfdataset.prefetch(tf.data.experimental.AUTOTUNE)
        self.iterator = iter(self.tfdataset)
        self.dataset_len = dataset_len

    def __len__(self):
        return int(np.floor(self.dataset_len / self.batch_size))

    def __getitem__(self, index):
        if (config.arc == 'SAGE'):
            batch_sampleID, batch_orientation, batch_eyelandmark,\
                batch_leye_im, batch_reye_im, batch_label = self.iterator.get_next()
            return [batch_orientation, batch_eyelandmark, batch_leye_im, batch_reye_im], batch_label
        else:
            batch_frameID, batch_orientation, batch_face_grid, batch_face_im,\
                batch_leye_im, batch_reye_im, batch_label = self.iterator.get_next()
            return [batch_orientation, batch_leye_im, batch_reye_im, batch_face_im, batch_face_grid], batch_label

    def reset(self):
        self.iterator = iter(self.tfdataset)
        return self

    def on_epoch_end(self):
        self.iterator = iter(self.tfdataset)


class TFDataFeeder_pipeline(Sequence):
    '''
    Used for whole pipeline model
    Output format: orientation, im
    '''
    def __init__(self, tfdataset, batch_size, dataset_len):
        # print('TFDataFeeder: batch size ', batch_size,
        #       ' dataset len', dataset_len, 'prefetch autotune')
        self.batch_size = batch_size
        self.tfdataset = tfdataset.batch(self.batch_size)
        self.tfdataset = self.tfdataset.prefetch(tf.data.experimental.AUTOTUNE)
        self.iterator = iter(self.tfdataset)
        self.dataset_len = dataset_len

    def __len__(self):
        return int(np.floor(self.dataset_len / self.batch_size))

    def __getitem__(self, index):
        batch_sampleID, batch_orientation, batch_im, batch_label = self.iterator.get_next()
        return [batch_orientation, batch_im], batch_label

    def reset(self):
        self.iterator = iter(self.tfdataset)
        return self

    def on_epoch_end(self):
        self.iterator = iter(self.tfdataset)


class TFDataFeeder_iTracker(Sequence):
    '''
    Used for iTracker architecture
    Output format: orientation, leye_grid, reye_grid, leye_im, reye_im
    '''
    def __init__(self, tfdataset, batch_size, dataset_len):
        # print('TFDataFeeder: batch size ', batch_size,
        #       ' dataset len', dataset_len, 'prefetch autotune')
        self.batch_size = batch_size
        self.tfdataset = tfdataset.batch(self.batch_size)
        self.tfdataset = self.tfdataset.prefetch(tf.data.experimental.AUTOTUNE)
        self.iterator = iter(self.tfdataset)
        self.dataset_len = dataset_len

    def __len__(self):
        return int(np.floor(self.dataset_len / self.batch_size))

    def __getitem__(self, index):
        batch_sampleID, batch_orientation, batch_grid_im, batch_face_im,\
            batch_leye_im, batch_reye_im, batch_label = self.iterator.get_next()
        return [batch_orientation, batch_leye_im, batch_reye_im, batch_face_im, batch_grid_im], batch_label

    def reset(self):
        self.iterator = iter(self.tfdataset)
        return self

    def on_epoch_end(self):
        self.iterator = iter(self.tfdataset)
