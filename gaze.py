import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import Model, Sequential, load_model
from math import exp, sqrt, pi, floor
import numpy as np
import time
import pandas as pd
import tensorflow as tf
import losses
import generator
import gaze_models
import config
import decode_utils
from mitdata_utils import load_meta_data, split_meta_data

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def prep_meta_data():
    dots, regions, df_info = load_meta_data(config.path+'processed/')

    '''Sort dataset by subject and frame ID'''
    frameIDs = regions[:, 0]*100000 + regions[:, 1]
    sorted_index = frameIDs.argsort()
    regions = regions[sorted_index, :]
    dots = dots[sorted_index, :]

    '''Filter out invalid subjectID'''
    dots = dots[regions[:, 0] != 208, :]
    regions = regions[regions[:, 0] != 208, :]

    dots = dots[(regions[:, 0] != 2109) |
                ((regions[:, 1] < 341) | (regions[:, 1] > 344)), :]
    regions = regions[(regions[:, 0] != 2109) |
                      ((regions[:, 1] < 341) | (regions[:, 1] > 344)), :]

    dataset_dict = split_meta_data(dots, regions, df_info)

    return dataset_dict


def process_path(file_path, label, region):
    sampleID = region[:2]  # subjectID, frameID
    img = tf.io.read_file(file_path)  # load image from the file as a string
    decoded_im_tuple = decode_utils.decode_img(img, region, label)
    return (sampleID,) + decoded_im_tuple


def process_path_enhanced(file_path, label, region):
    sampleID = region[:2]  # subjectID, frameID
    leye_img = tf.io.read_file(file_path[0])  # load image from the file as a string
    reye_img = tf.io.read_file(file_path[1])  # load image from the file as a string
    decoded_im_tuple = decode_utils.decode_img_enhanced(leye_img, reye_img, region, label)
    return (sampleID,) + decoded_im_tuple


def make_tf_dataset(dots, regions, shuffle=False):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    indices = np.arange(len(dots))
    if shuffle:
        np.random.shuffle(indices)
    shuffled_dots = dots[indices, :]
    shuffled_regions = regions[indices, :]

    filenames = []
    for i in range(len(shuffled_dots)):
        region = shuffled_regions[i, :].astype(int)
        filenames.append(decode_utils.get_frame_path(region[0], region[1]))

    list_ds = tf.data.Dataset.from_tensor_slices(
        (filenames, shuffled_dots[:, -3:-1], shuffled_regions))
    if config.enhanced:
        ds = list_ds.map(process_path_enhanced, num_parallel_calls=AUTOTUNE)
    else:
        ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    return ds


def square_euclidean_pred(pred, scope):
    # pred: euclidean prediction (x,y)
    # scope: scope of screen/ an instance from losses.get_pred_scope
    if (pred[0] > scope['max'][0]):
        pred[0] = scope['max'][0]
    if (pred[1] > scope['max'][1]):
        pred[1] = scope['max'][1]
    if (pred[0] < scope['min'][0]):
        pred[0] = scope['min'][0]
    if (pred[1] < scope['min'][1]):
        pred[1] = scope['min'][1]
    return pred


def test_euclidean(model, dots_test, regions_test, df_info_test,
                   dots_train, regions_train, df_info_train):
    '''Euclidean test'''
    t = time.time()
    ds_test = make_tf_dataset(dots_test, regions_test, shuffle=False)
    test_generator = generator.TFDataFeeder(ds_test, batch_size=config.batch_size,
                                            dataset_len=len(dots_test))
    preds = model.predict(test_generator.reset(),
                          steps=np.floor(len(dots_test)/config.batch_size), verbose=1)

    unique_devices = pd.unique(df_info_test['DeviceName'])
    for device in unique_devices:
        train_subjectIDs = list(df_info_train.loc[df_info_train['DeviceName'] == device,
                                                  'subjectID'])
        test_subjectIDs = list(df_info_test.loc[df_info_test['DeviceName'] == device,
                                                'subjectID'])
        orientations = [1, 3, 4] if config.mobile else [1, 2, 3, 4]
        for ori in orientations:
            train_indices = np.where((regions_train[:, 24] == ori) &
                                     (np.isin(regions_train[:, 0], train_subjectIDs)))[0]
            pred_max = np.max((dots_train[train_indices, -3:-1]), axis=0)
            pred_min = np.min((dots_train[train_indices, -3:-1]), axis=0)
            tmp_indices = np.where((regions_test[-len(preds):, 24] == ori) &
                                   (np.isin(regions_test[-len(preds):, 0],
                                    test_subjectIDs)))[0]
            tmp_preds = preds[tmp_indices, :]
            tmp_preds[(tmp_preds[:, 0] > pred_max[0]), 0] = pred_max[0]
            tmp_preds[(tmp_preds[:, 1] > pred_max[1]), 1] = pred_max[1]
            tmp_preds[(tmp_preds[:, 0] < pred_min[0]), 0] = pred_min[0]
            tmp_preds[(tmp_preds[:, 1] < pred_min[1]), 1] = pred_min[1]
            preds[tmp_indices, :] = tmp_preds

    single_error = losses.euclidean_error(preds, dots_test[-len(preds):, -3:-1])
    averaged_test_error = losses.euclidean_fixation_error(preds, dots_test[-len(preds):, :],
                                                          regions_test[-len(preds):, :])
    print('Single error', len(preds), single_error)
    print('Averaged error', averaged_test_error)


def test_heatmap(model, dots_test, regions_test, df_info_test,
                 dots_train, regions_train, df_info_train):
    '''Heatmap test'''
    df_pred_scopes = losses.get_pred_scope(df_info_train, regions_train,
                                           dots_train, df_info_test)
    subjectIDs, subjectID_indices = np.unique(regions_test[:, 0], return_index=True)
    ecl_errs = []
    pred_count = 0
    invalid_frames = []
    valid_frames = []
    dot_errs = []
    dot_errs1 = []

    single_preds = []
    step = 10
    for subjectID_idx in range(0, len(subjectIDs), step):
        subjectID_list = subjectIDs[subjectID_idx:subjectID_idx+step]
        print('subjectID_list', subjectID_list)
        dots = dots_test[np.isin(dots_test[:, 0], subjectID_list), :]
        regions = regions_test[np.isin(dots_test[:, 0], subjectID_list), :]
        ds = make_tf_dataset(dots, regions, shuffle=False)
        batch_size = 64 if (len(dots) > 64) else 64
        subject_generator = generator.TFDataFeeder(ds, batch_size=batch_size,
                                                   dataset_len=len(dots))
        preds = model.predict(subject_generator, verbose=1)
        preds = preds[:, :, :, 0]
        dots = dots[-len(preds):, :]
        regions = regions[-len(preds):, :]
        for subjectID in subjectID_list:
            if(subjectID == 2032):
                continue
            subject_dots = dots[dots[:, 0] == subjectID, :]
            subject_regions = regions[dots[:, 0] == subjectID, :]
            subject_preds = preds[dots[:, 0] == subjectID, :, :]
            pred_count += len(subject_preds)
            device = df_info_test.loc[df_info_test['subjectID'] == subjectID, 'DeviceName']
            device = list(device)[0]
            subject_ecl_errs = []

            subject_ecl_preds = np.zeros((len(subject_dots), 2))
            valid_indices = []

            '''Single frame error'''
            for i, pred_hm in enumerate(subject_preds):
                ori = subject_regions[i, 24]
                scope = df_pred_scopes.loc[device, ori]
                filter_mask = losses.get_heatmap_filter_mask(subject_regions[i, :2],
                                                             df_pred_scopes,
                                                             df_info_test, subject_regions)
                pred_hm *= (filter_mask*128**2)
                pred_hm = (pred_hm*128**2) * (filter_mask*128**2)
                if (np.sum(pred_hm) == 0):
                    invalid_frames.append(subject_regions[i, :2])
                    pred_hm = filter_mask
                    continue
                pred_hm /= np.sum(pred_hm)
                pred_ecl = np.array(losses.get_matrix_central(pred_hm))
                pred_ecl = (pred_ecl - config.hm_size/2)/config.scale
                # cut-off
                pred_ecl = square_euclidean_pred(pred_ecl, scope)
                ecl_err = losses.euclidean_error(subject_dots[i, -3:-1], pred_ecl)
                # print(i, ori, ecl_err, pred_ecl, dots[i, -3:-1], scope['max'], scope['min'])
                # break
                if (~np.isnan(ecl_err)):
                    subject_ecl_errs.append(ecl_err)
                    ecl_errs.append(ecl_err)

                    # add
                    valid_frames.append(subject_regions[i, :2])

                    # to calibrate
                    single_preds.append([subject_regions[i, 0], subject_regions[i, 1],
                                         pred_ecl[0], pred_ecl[1],
                                         subject_dots[i, -3], subject_dots[i, -2]])
                    # to compute fixation
                    subject_ecl_preds[i, :] = pred_ecl
                    valid_indices.append(i)

                # print(np.mean(ecl_err), np.max(pred_hm))
            print(subjectID, device, preds.shape, np.mean(subject_ecl_errs),
                  np.mean(ecl_errs), len(ecl_errs), pred_count)
            no_pts, subject_pt_err = losses.euclidean_fixation_error(subject_ecl_preds[valid_indices, :],
                                                                  subject_dots[valid_indices, :],
                                                                  subject_regions[valid_indices, :])
            if (~np.isnan(subject_pt_err)):
                dot_errs1.append(subject_pt_err)
            print(subject_pt_err, np.mean(dot_errs1))

            aggregate = True
            if (aggregate):
                '''Aggregated error'''
                ## add orientation
                gazePt_list = np.unique(subject_dots[:, -3:-1], axis=0)
                for gazePt in gazePt_list:
                    tmp_dot_pred = []
                    dot_indices = np.where((subject_dots[:, -3] == gazePt[0]) &
                                           (subject_dots[:, -2] == gazePt[1]))[0]
                    dot_pred_hm = np.ones((config.hm_size, config.hm_size))
                    for i in dot_indices:
                        ori = subject_regions[i, 24]
                        scope = df_pred_scopes.loc[device, ori]
                        pred_hm = subject_preds[i, :, :]
                        filter_mask = losses.get_heatmap_filter_mask(subject_regions[i, :2],
                                                                    df_pred_scopes,
                                                                    df_info_test, subject_regions)
                        pred_hm *= filter_mask
                        if (np.sum(pred_hm) == 0):
                            continue
                        dot_pred_hm = (dot_pred_hm) * (pred_hm) # + filter_mask*10/128**2)
                        # print(gazePt, '###', np.sum(dot_pred_hm))
                        dot_pred_hm /= np.sum(dot_pred_hm)
                        pred_hm /= np.sum(pred_hm)
                        pred_ecl = np.array(losses.get_matrix_central(pred_hm))
                        pred_ecl = (pred_ecl - config.hm_size/2)/config.scale
                        # cut-off
                        pred_ecl = square_euclidean_pred(pred_ecl, scope)
                        tmp_dot_pred.append(pred_ecl)
                    if (len(tmp_dot_pred) > 0):
                        ecl_err = losses.euclidean_error(subject_dots[i, -3:-1],
                                                         np.mean(np.array(pred_ecl), axis=0))
                        if (~np.isnan(ecl_err)):
                            dot_errs1.append(ecl_err)
                    # print('#', np.mean(tmp_dot_err))
                    dot_pred_hm *= filter_mask
                    pred_ecl = np.array(losses.get_matrix_central(dot_pred_hm))
                    pred_ecl = (pred_ecl - config.hm_size/2)/config.scale
                    # cut-off
                    pred_ecl = square_euclidean_pred(pred_ecl, scope)
                    ecl_err = losses.euclidean_error(gazePt, pred_ecl)
                    # print(gazePt, dot_indices, ecl_err)
                    if (~np.isnan(ecl_err)):
                        dot_errs.append(ecl_err)
                print(np.mean(dot_errs), len(dot_errs), np.mean(dot_errs1), len(dot_errs1))

    # np.save(config.path + '/processed/mobile_single_preds.npy', np.array(single_preds))
    # np.save(config.path + 'invalid_frames_a0.2-0.4.npy', np.array(invalid_frames))


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr


def evaluate(model, ds_test, test_size):
    t = time.time()
    test_batch_size = 512
    ds_test = ds_test.batch(test_batch_size)
    iterator = iter(ds_test)
    errs = []
    for i in range(int(test_size/test_batch_size)+1):
        batch_sampleID, batch_orientation, batch_eyelandmark,\
            batch_leye_im, batch_reye_im, batch_label = iterator.get_next()
        tmp_preds = model([batch_orientation, batch_eyelandmark, batch_leye_im, batch_reye_im]).numpy()
        tmp_errs = np.mean(np.sqrt(np.sum(np.square(tmp_preds - batch_label), axis=1)))
        errs.append(tmp_errs)
        print(i, tmp_errs)
    err = np.mean(errs)
    print('predicting time:', time.time()-t, err)
    return err


def main():
    print('#Architecture', config.arc, ' #Heatmap', config.heatmap,
          ' # Mobile', config.mobile, ' #Test ', config.test)
    '''LOAD META DATA'''
    t = time.time()
    dataset_dict = prep_meta_data()
    dots_train, regions_train, df_info_train = dataset_dict['train']
    dots_val, regions_val, df_info_val = dataset_dict['val']
    dots_train = np.concatenate([dots_train, dots_val])
    regions_train = np.concatenate([regions_train, regions_val])
    df_info_train = pd.concat([df_info_train, df_info_val])
    dots_val, regions_val, df_info_val = dataset_dict['test']
    print('train data:', dots_train.shape, regions_train.shape, df_info_train.shape)
    print('val data:', dots_val.shape, regions_val.shape, df_info_val.shape)
    '''LOAD MODEL'''
    base_model = config.base_model
    mobile_str = 'm' if config.mobile else 't'
    weights_str = 'scratch' if (config.weights is None) else str(config.weights)
    model_name = config.arc + '_' + base_model + '_' + weights_str + '_' + \
        mobile_str + '_' + \
        str(config.faceIm_size) + '_' + str(config.eyeIm_size) + '_' + \
        str(config.channel) + '_' + config.regions
    if (config.enhanced):
        model_name += '_enhanced'
    # Create folder to store model
    model_path = config.path + 'model/euclidean/'
    if (config.heatmap):
        model_path = config.path + 'model/heatmap/'
        model_name += '_hm_' + str(config.r)
    model_path = model_path + base_model + '/'
    if(not os.path.exists(model_path)):
        os.makedirs(model_path)
    print('Base model: ', base_model, '-', model_name, ' Model path: ', model_path)

    strategy = tf.distribute.MirroredStrategy()  # multiple gpus
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    current_lr = config.current_lr
    with strategy.scope():
        if(config.arc == 'iTracker'):
            model = gaze_models.get_iTracker_custom(base_model)
        elif(config.arc == 'SAGE'):
            model = gaze_models.get_SAGE(base_model)
        model.save(model_path + model_name + '.h5')
        '''Load FULL or PARTIAL weights from checkpoint model'''
        if (config.pretrained_model is not None):
            print('Pretrained model: ', config.pretrained_model)
            model.load_weights(config.pretrained_model, by_name=True)
        loss = losses.heatmap_loss if config.heatmap else losses.euclidean_loss
        adam = tf.keras.optimizers.Adam(learning_rate=current_lr)
        lr_metric = get_lr_metric(adam)
        model.compile(loss=loss, optimizer=adam, metrics=[lr_metric])
        # print(model.summary())
        print('Model\'s total params: ', f'{model.count_params():,}')
    print('loading metadata and model time:', time.time()-t)

    '''TRAIN/TEST'''
    if (config.test):
        if (config.heatmap):
            test_heatmap(model, dots_val, regions_val, df_info_val,
                         dots_train, regions_train, df_info_train)
        else:
            test_euclidean(model, dots_val, regions_val, df_info_val,
                           dots_train, regions_train, df_info_train)
        return 0

    '''Val dataset'''
    ds_val = make_tf_dataset(dots_val, regions_val, shuffle=False)
    val_generator = generator.TFDataFeeder(ds_val, batch_size=config.batch_size,
                                           dataset_len=len(dots_val))
    best_val_loss = config.current_best_val_loss
    for i in range(config.current_training_round, 1000):
        print('#### Round', i)
        ds_train = make_tf_dataset(dots_train, regions_train, shuffle=True)
        train_generator = generator.TFDataFeeder(ds_train, batch_size=config.batch_size,
                                                 dataset_len=len(dots_train))
        model.fit(train_generator, steps_per_epoch=config.steps_per_epoch,
                  epochs=config.epochs, verbose=1)

        if config.mobile & ((i % 3 != 0) | (i < 0)):
            continue

        # update learning rate
        current_lr *= 0.8
        adam = tf.keras.optimizers.Adam(learning_rate=current_lr)
        lr_metric = get_lr_metric(adam)
        model.compile(loss=loss, optimizer=adam, metrics=[lr_metric])

        val_loss = model.evaluate(val_generator.reset(),
                                  steps=np.floor(len(dots_val)/config.batch_size), verbose=1)

        if (best_val_loss > val_loss[0]):
            best_val_loss = val_loss[0]
            model.save_weights(model_path + model_name + '_' + str(round(val_loss[0], 4))
                               + '_round' + str(i) + '_' + str(config.batch_size) + 'x'
                               + str(config.steps_per_epoch) + 'x' + str(config.epochs)
                               + '.hdf5')


if __name__ == "__main__":
    main()
    # generate_img()
