''' utility functions to load image and label data
    to feed into model'''
import config
import numpy as np
from math import exp, sqrt, pi
import tensorflow as tf
from tensorflow.image import convert_image_dtype, resize, pad_to_bounding_box


def get_frame_path(subjectID, frameID):
    if (config.enhanced & (config.arc == 'SAGE')):
        leye_path = config.enhanced_path + str(subjectID) + '_' + str(frameID) + '_leye.jpg'
        reye_path = config.enhanced_path + str(subjectID) + '_' + str(frameID) + '_reye.jpg'
        return [leye_path, reye_path]
    else:
        subject_path = config.path + 'gazecapture/' + '{:0>5d}'.format(int(subjectID)) + '/'
        frame_path = subject_path + 'frames/' + '{:0>5d}'.format(int(frameID)) + '.jpg'
        return frame_path


def gauss(x, stdv=0.5):
    return exp(-(1/(2*stdv))*(x**2))/(sqrt(2*pi*stdv))


def decode_img(img, region, label):
    precision_type = tf.float16
    region = tf.cast(region, tf.int32)

    face_im = tf.io.decode_and_crop_jpeg(
        img, [region[5], region[4], region[3], region[2]], channels=config.channel)

    if(config.regions == 'default'):  # only for the default GazeCapture facial regions
        '''Crop eye regions from face region'''
        leye_im = face_im[region[10]:(region[10]+region[7]), region[9]:(region[9]+region[8]), :]
        reye_im = face_im[region[15]:(region[15]+region[12]), region[14]:(region[14]+region[13]), :]
    else:
        '''Decode and crop eye regions directly from raw image'''
        leye_im = tf.io.decode_and_crop_jpeg(
            img, [region[10], region[9], region[8], region[7]], channels=config.channel)
        reye_im = tf.io.decode_and_crop_jpeg(
            img, [region[15], region[14], region[13], region[12]], channels=config.channel)

    '''Convert to float16/32 in the [0,1] range'''
    leye_im = convert_image_dtype(leye_im, precision_type)
    reye_im = convert_image_dtype(reye_im, precision_type)

    '''Resize'''
    leye_im = resize(leye_im, [config.eyeIm_size, config.eyeIm_size])
    reye_im = resize(reye_im, [config.eyeIm_size, config.eyeIm_size])

    '''Normalize'''
    # leye_im = tf.image.per_image_standardization(leye_im)
    # reye_im = tf.image.per_image_standardization(reye_im)

    orientation = tf.cast(tf.one_hot(region[24], depth=3), precision_type)

    if (config.arc == 'iTracker'):
        face_im = convert_image_dtype(face_im, precision_type)
        face_im = resize(face_im, [config.faceIm_size, config.faceIm_size])
        '''Create face grid'''
        face_grid_im = convert_image_dtype(tf.ones((region[19], region[19], 1)), precision_type)
        face_grid_im = pad_to_bounding_box(
            face_grid_im, region[18], region[17], config.faceGrid_size, config.faceGrid_size)

    elif (config.arc == 'SAGE'):
        eyelandmark = tf.cast(tf.concat([region[8:11], region[13:16]], 0), tf.float32)/640.0
        # SAGE mode
        leye_im = tf.image.flip_left_right(leye_im)
    '''Create heatmap label'''
    if (config.heatmap):
        hmFocus_size = 17  # if (config.mobile) else 9  # in pixel unit

        HM_FOCUS_IM = np.zeros((5, hmFocus_size, hmFocus_size, 1))

        stdv_list = [0.2, 0.25, 0.3, 0.35, 0.4]
        for level in range(5):  # 5 levels of std to constuct heatmap
            stdv = stdv_list[level]  # 3/(12-level)
            for i in range(hmFocus_size):
                for j in range(hmFocus_size):
                    distanceFromCenter = 2 * \
                        np.linalg.norm(np.array([i-int(hmFocus_size/2),
                                                 j-int(hmFocus_size/2)]))/((hmFocus_size)/2)
                    gauss_prob = gauss(distanceFromCenter, stdv)
                    HM_FOCUS_IM[level, i, j, 0] = gauss_prob
        HM_FOCUS_IM[level, :, :, 0] /= np.sum(HM_FOCUS_IM[level, :, :, 0])
        heatmap_im = convert_image_dtype(HM_FOCUS_IM[0, :, :, :], tf.float32)
        heatmap_im = pad_to_bounding_box(heatmap_im,
                                         int(label[0]*config.scale+config.hm_size/2-hmFocus_size/2),
                                         int(label[1]*config.scale+config.hm_size/2-hmFocus_size/2),
                                         config.hm_size, config.hm_size)
        label = heatmap_im

    if (config.arc == 'SAGE'):
        return (orientation, eyelandmark, leye_im, reye_im, label)
    else:
        return (orientation, face_grid_im, face_im, leye_im, reye_im, label)


def decode_img_enhanced(leye_img, reye_img, region, label):
    precision_type = tf.float16
    region = tf.cast(region, tf.int32)

    leye_im = tf.io.decode_jpeg(leye_img)
    reye_im = tf.io.decode_jpeg(reye_img)

    '''Convert to float16/32 in the [0,1] range'''
    leye_im = convert_image_dtype(leye_im, precision_type)
    reye_im = convert_image_dtype(reye_im, precision_type)
    '''Resize'''
    leye_im = resize(leye_im, [config.eyeIm_size, config.eyeIm_size])
    reye_im = resize(reye_im, [config.eyeIm_size, config.eyeIm_size])
    '''Normalize'''
    # leye_im = tf.image.per_image_standardization(leye_im)
    # reye_im = tf.image.per_image_standardization(reye_im)

    orientation = tf.cast(tf.one_hot(region[24], depth=3), precision_type)

    eyelandmark = tf.cast(tf.concat([region[8:11], region[13:16]], 0), tf.float32)/640.0

    '''Create heatmap label'''
    if (config.heatmap):
        hmFocus_size = 17 if (config.mobile) else 9  # tablet focus_size=9

        HM_FOCUS_IM = np.zeros((5, hmFocus_size, hmFocus_size, 1))

        stdv_list = [0.2, 0.25, 0.3, 0.35, 0.4]
        for level in range(5):  # 5 levels of std to constuct heatmap
            stdv = stdv_list[level]  # 3/(12-level)
            for i in range(hmFocus_size):
                for j in range(hmFocus_size):
                    distanceFromCenter = 2 * \
                        np.linalg.norm(np.array([i-int(hmFocus_size/2),
                                                 j-int(hmFocus_size/2)]))/((hmFocus_size)/2)
                    gauss_prob = gauss(distanceFromCenter, stdv)
                    HM_FOCUS_IM[level, i, j, 0] = gauss_prob
        HM_FOCUS_IM[level, :, :, 0] /= np.sum(HM_FOCUS_IM[level, :, :, 0])

        heatmap_im = convert_image_dtype(HM_FOCUS_IM[0, :, :, :], tf.float32)
        heatmap_im = pad_to_bounding_box(heatmap_im,
                                         int(label[0]*config.scale+config.hm_size/2-hmFocus_size/2),
                                         int(label[1]*config.scale+config.hm_size/2-hmFocus_size/2),
                                         config.hm_size, config.hm_size)
        label = heatmap_im

    return (orientation, eyelandmark, leye_im, reye_im, label)
