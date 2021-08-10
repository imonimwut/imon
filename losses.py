import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
import config
'''Loss in tensorflow, erorr in numpy'''


def euclidean_loss(y_true, y_pred):
    # y_pred shape: (steps, batch_size, 2)
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1, keepdims=True))


def heatmap_loss(y_true, y_pred):
    if (K.sum(y_pred) == 0):
        return 1.0
    else:
        y_pred /= K.sum(y_pred)
        y_true /= K.sum(y_true)
        return K.sum(K.abs(y_pred - y_true))/2


def euclidean_error(y_pred, y_true):
    return np.mean(np.sqrt(np.sum(np.square(y_true - y_pred), axis=-1)))


def heatmap_error(y_pred, y_true):
    # there could be some cases that have zero sum??
    if(np.sum(y_pred) != 0):
        y_pred /= np.sum(y_pred)
    if(np.sum(y_true) != 0):
        y_true /= np.sum(y_true)

    return np.sum(np.abs(y_pred - y_true))


def euclidean_average_shift_error(y_pred, dots, regions):
    '''
    calculate the error for the pair model
    dots: 1st gaze point, subjectID, 1st dotID, 2nd gaze point, subjectID, 2nd dotID
    '''
    y_true = dots[:, 4:6] - dots[:, :2]
    subject_dot_ID = dots[:, [6, 4, 5]]

    df_pred = pd.DataFrame(subject_dot_ID, columns=['subjectID', 'y1_true', 'y2_true'])
    df_pred['orientation'] = regions[:, 24]
    df_pred['y1_pred'] = y_pred[:, 0]
    df_pred['y2_pred'] = y_pred[:, 1]
    df_pred['y1_true'] = y_true[:, 0]
    df_pred['y2_true'] = y_true[:, 1]
    # print(df_pred)
    tmp = np.array(df_pred.groupby(['subjectID', 'y1_true', 'y2_true', 'orientation'])
                   ['y1_pred', 'y2_pred', 'y1_true', 'y2_true'].apply(pd.Series.mean))

    return euclidean_error(tmp[:, :2], tmp[:, 2:])


def euclidean_average_endpoint_error(y_pred, dots, regions):
    '''
    calculate the error for the pair model
    dots: 1st gaze point, subjectID, 1st dotID, 2nd gaze point, subjectID, 2nd dotID
    '''
    y_true = dots[:, 4:6]
    subject_dot_ID = dots[:, [6, 4, 5]]

    df_pred = pd.DataFrame(subject_dot_ID, columns=['subjectID', 'y1_true', 'y2_true'])
    df_pred['orientation'] = regions[:, 24]
    df_pred['y1_pred'] = y_pred[:, 0]
    df_pred['y2_pred'] = y_pred[:, 1]
    df_pred['y1_true'] = y_true[:, 0]
    df_pred['y2_true'] = y_true[:, 1]
    # print(df_pred)
    tmp = np.array(df_pred.groupby(['subjectID', 'y1_true', 'y2_true', 'orientation'])
                   ['y1_pred', 'y2_pred', 'y1_true', 'y2_true'].apply(pd.Series.mean))
    print(y_pred.shape, tmp.shape)

    return euclidean_error(tmp[:, :2], tmp[:, 2:])


def euclidean_average_endpoint_dense_error(y_pred, dots, second_regions):
    '''
    calculate the error for the pair model
    dots: 1st gaze point, subjectID, 1st dotID, 2nd gaze point, subjectID, 2nd dotID
    '''
    y_true = dots[:, 4:6]
    subject_dot_ID = dots[:, [6]]

    df_pred = pd.DataFrame(subject_dot_ID, columns=['subjectID'])
    df_pred['second_frameID'] = second_regions[:, 1]
    df_pred['y1_pred'] = y_pred[:, 0]
    df_pred['y2_pred'] = y_pred[:, 1]
    df_pred['y1_true'] = y_true[:, 0]
    df_pred['y2_true'] = y_true[:, 1]
    # print(df_pred)
    tmp = np.array(df_pred.groupby(['subjectID', 'second_frameID'])
                   ['y1_pred', 'y2_pred', 'y1_true', 'y2_true'].apply(pd.Series.mean))
    print(y_pred.shape, tmp.shape)

    return euclidean_error(tmp[:, :2], tmp[:, 2:])


def euclidean_fixation_error(y_pred, dots, regions):
    '''
    dots, regions: original gazecapture format
    '''
    y_true = dots[:, -3:-1]
    # subject_dot_ID = dots[:,[0,2]] # subjectID, dotID
    # sujectID, gaze point: there are multiple dotIDs with same coordination
    subject_dot_ID = dots[:, [0, -3, -2]]

    df_pred = pd.DataFrame(subject_dot_ID, columns=['subjectID', 'y1_true', 'y2_true'])
    df_pred['orientation'] = regions[:, 24]
    df_pred['y1_pred'] = y_pred[:, 0]
    df_pred['y2_pred'] = y_pred[:, 1]
    df_pred['y1_true'] = y_true[:, 0]
    df_pred['y2_true'] = y_true[:, 1]
    # print(df_pred)
    tmp = np.array(df_pred.groupby(['subjectID', 'y1_true', 'y2_true', 'orientation'])
                   ['y1_pred', 'y2_pred', 'y1_true', 'y2_true'].apply(pd.Series.mean))
    print('shape: ', tmp.shape)

    return len(tmp), euclidean_error(tmp[:, :2], tmp[:, 2:])


def get_matrix_central(matrix):
    matrix_indices = np.indices(matrix.shape)
    row_indices = matrix_indices[0, :, :]
    col_indices = matrix_indices[1, :, :]
    return round(np.sum(row_indices*matrix), 4), round(np.sum(col_indices*matrix), 4)


def get_pred_scope(df_info_train, regions_train, dots_train, df_info_test):
    # Get scope of each test device
    unique_devices = pd.unique(df_info_test['DeviceName'])
    pred_scopes = []
    for device in unique_devices:  # each device has different dimension measurement
        train_subjectIDs = list(df_info_train.loc[df_info_train['DeviceName'] == device,
                                                  'subjectID'])
        orientations = [1, 3, 4] if config.mobile else [1, 2, 3, 4]
        for ori in orientations:
            train_indices = np.where((regions_train[:, 24] == ori) &
                                     (np.isin(regions_train[:, 0], train_subjectIDs)))[0]
            pred_max = np.max(dots_train[train_indices, -3: -1], axis=0)
            pred_min = np.min(dots_train[train_indices, -3: -1], axis=0)
            pred_scopes.append([device, ori, pred_max, pred_min])

    df_pred_scopes = pd.DataFrame(pred_scopes)
    df_pred_scopes.columns = ['device', 'orientation', 'max', 'min']
    df_pred_scopes = df_pred_scopes.set_index(['device', 'orientation'])

    return df_pred_scopes


def get_heatmap_filter_mask(frameID, df_pred_scopes, df_info_test, regions_test):
    '''
        A mask to filter out out-of-range heatmap region
            Each (device_name, orientation) has a unique mask
    '''

    from tabulate import tabulate
    # Define mask based on the scope
    device_name = df_info_test.loc[df_info_test['subjectID'] == frameID[0], 'DeviceName'].iloc[0]
    orientation = regions_test[(regions_test[:, 0] == frameID[0]) &
                               (regions_test[:, 1] == frameID[1]), 24][0]
    pred_scope = df_pred_scopes.loc[device_name, orientation]*config.scale + config.hm_size/2
    # print('df pred scopes:\n', tabulate(df_pred_scopes))
    pred_scope_min = pred_scope['min'].astype(int)
    pred_scope_max = pred_scope['max'].astype(int)
    # Create a filtering mask
    filter_mask = np.zeros((config.hm_size, config.hm_size))
    filter_mask[pred_scope_min[0]: pred_scope_max[0], pred_scope_min[1]: pred_scope_max[1]] = 1

    return filter_mask
