import numpy as np
import pandas as pd
import json
import time
import cv2
import glob
import tarfile
from PIL import Image
# from matplotlib import pyplot as plt
from decode_utils import get_frame_path
import config


def preprocess_image_data(path, regions):
    '''TO debug invalid images with invalid region/offset'''
    t = time.time()
    grid_im = np.zeros((25, 25), dtype=np.uint8)
    for i in range(0, len(regions)):
        region = regions[i, :].astype(int)
        subjectID = region[0]
        frameID = region[1]

        frame_path = get_frame_path(subjectID, frameID)
        im = np.array(Image.open(frame_path), dtype=np.uint8)  # 31s/10,000 images

        # GazeCapture default format
        grid_im[region[18]:region[18]+region[20], region[17]:region[17]+region[19]] = 1
        face_im = im[region[5]:(region[5]+region[2]), region[4]:(region[4]+region[3]), :]
        leye_im = im[region[10]:(region[10]+region[7]), region[9]:(region[9]+region[8]), :]
        reye_im = im[region[15]:(region[15]+region[12]), region[14]:(region[14]+region[13]), :]

        if((min(face_im.shape) == 0) | (min(leye_im.shape) == 0) | (min(reye_im.shape) == 0)):
            print(i, region)
            plt.imshow(face_im)
            plt.show()
            plt.imshow(leye_im)
            plt.show()
            plt.imshow(reye_im)
            plt.show()

        #print(reye_x, reye_y, reye_h, reye_w, region[16])
        face_im = (cv2.resize(face_im, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)).astype(np.uint8)
        leye_im = (cv2.resize(leye_im, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)).astype(np.uint8)
        reye_im = (cv2.resize(reye_im, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)).astype(np.uint8)

        if(i % 10000 == 0):
            print(i)
            # break
    print('preprocess_image_data() time: ', time.time() - t)


def split_meta_data(dots, regions, df_info):
    '''Split meta data into train/val/test'''
    dataset_dict = {}
    for target in ['train', 'val', 'test']:
        df_info_target = df_info.loc[df_info['Dataset'] == target, :]
        dots_target = dots[np.isin(dots[:, 0], df_info_target['subjectID']), :]
        regions_target = regions[np.isin(regions[:, 0], df_info_target['subjectID']), :]
        dataset_dict[target] = [dots_target, regions_target, df_info_target]

    return dataset_dict


def load_meta_data(path='./', pipeline=False):
    '''
    prefix: m~mobile, t~tablet
    df_info: subjectID, TotalFrames, NumFaceDetections, NumEyeDetections, Dataset, DeviceName, is Mobile
    dots: subjectID, frameID, DotNum (DotIndex), Xpts (distance to screen top-left), YPt, XCam (distance to camera), YCam, Time
    regions: subjectID, frameID, face(h,w,x,y,valid), left_eye(), right_eye(), face_grid(x,y,w,h), scr_h, scr_w, scr_ori
        w=h for face/eye regions as they are square
        x,y in the image is reverse of as in numpy array
    eyeLandmarks:  subjectID, frameID, leye_landmarks (6 points/coordinations), reye_landmarks, valid_bit

    file:
        regions_ori: default regions with orientation
        regions_ori_v10: FAN regions RLEYE_LAPLACE_VAR_THRES = 0.9
    '''
    df_info = pd.read_csv(path + 'df_info.csv')
    dots = np.load(path + 'dots.npy')
    regions_dict = {'default': 'regions_ori',
                    'fan': 'regions_ori_v10'}
    regions = np.load(path + regions_dict[config.regions] + '.npy').astype(int)
    face_grid_size = 25

    # '''Keep the GazeCapture 'valid' status for validation data'''
    # default_regions = np.load(path + 'regions_ori.npy').astype(int)
    # df_info_val = df_info.loc[df_info['Dataset'] == 'val',:]
    # regions[np.isin(regions[:,0], df_info_val['subjectID']),11] *= default_regions[np.isin(regions[:,0], df_info_val['subjectID']),11]
    # df_info_test = df_info.loc[df_info['Dataset'] == 'test',:]
    # regions[np.isin(regions[:,0], df_info_test['subjectID']),11] *= default_regions[np.isin(regions[:,0], df_info_test['subjectID']),11]
    # print('all dataset:',dots.shape, regions.shape, df_info.shape, time.time()-t)

    df_info = df_info.loc[df_info['isMobile'] == int(config.mobile), :].reset_index(drop=True)
    subjectIDs = list(df_info['subjectID'])
    dots = dots[np.isin(dots[:, 0], subjectIDs), :]
    regions = regions[np.isin(regions[:, 0], subjectIDs), :]

    if(pipeline):
        return dots, regions, df_info

    # filter out samples with invalid face or eye regions
    dots = dots[(regions[:, 11] == 1), :]
    regions = regions[(regions[:, 11] == 1), :]
    unique_subjectID_list = list(np.unique(regions[:, 0]))
    df_info = df_info.loc[df_info['subjectID'].isin(unique_subjectID_list), :]
    # print('valid frames:',dots.shape, regions.shape, df_info.shape)

    # fix grid offset
    regions[:, 17:19] = regions[:, 17:19] - 1
    regions[(regions[:, 19] + regions[:, 17]-1) > face_grid_size, 17] -= 1
    regions[(regions[:, 19] + regions[:, 18]-1) > face_grid_size, 18] -= 1
    tmp_indices = np.where((regions[:, 19] + regions[:, 17]) > face_grid_size)[0]
    regions[tmp_indices, 19] -= (regions[tmp_indices, 19] + regions[tmp_indices, 17] - face_grid_size)
    tmp_indices = np.where((regions[:, 19] + regions[:, 18]) > face_grid_size)[0]
    regions[tmp_indices, 19] -= (regions[tmp_indices, 19] + regions[tmp_indices, 18] - face_grid_size)
    regions[:, 20] = regions[:, 19]
    regions[regions < 0] = 0  # replace all the coordinate value of face/eye/grid that < 0 by 0

    # fix face offset
    tmp_indices = np.where(regions[:, 2] + regions[:, 4] > 640)[0]
    regions[tmp_indices, 2] = 640 - regions[tmp_indices, 4]
    tmp_indices = np.where(regions[:, 2] + regions[:, 5] > 640)[0]
    regions[tmp_indices, 2] = 640 - regions[tmp_indices, 5]
    tmp_indices = np.where((regions[:, 24] < 3) & (regions[:, 2] + regions[:, 4] > 480))[0]
    regions[tmp_indices, 2] = 480 - regions[tmp_indices, 4]
    tmp_indices = np.where((regions[:, 24] > 2) & (regions[:, 2] + regions[:, 5] > 480))[0]
    regions[tmp_indices, 2] = 480 - regions[tmp_indices, 5]
    regions[:, 3] = regions[:, 2]

    # fix eye offset: used for iTracker default region
    if (config.regions == 'default'):
        face_x = regions[:, 4]
        face_y = regions[:, 5]
        # left eye
        tmp_indices = np.where(regions[:, 8] + face_x + regions[:, 9] > 640)[0]
        regions[tmp_indices, 8] = 640 - regions[tmp_indices, 4] - regions[tmp_indices, 9]
        tmp_indices = np.where(regions[:, 8] + face_x + regions[:, 10] > 640)[0]
        regions[tmp_indices, 8] = 640 - regions[tmp_indices, 5] - regions[tmp_indices, 10]
        tmp_indices = np.where((regions[:, -1] < 3) &
                               (regions[:, 8] + face_x + regions[:, 9] > 480))[0]
        regions[tmp_indices, 8] = 480 - regions[tmp_indices, 4] - regions[tmp_indices, 9]
        tmp_indices = np.where((regions[:, -1] > 2) &
                               (regions[:, 8] + face_y + regions[:, 10] > 480))[0]
        regions[tmp_indices, 8] = 480 - regions[tmp_indices, 5] - regions[tmp_indices, 10]
        # right eye
        tmp_indices = np.where(regions[:, 13] + face_x + regions[:, 14] > 640)[0]
        regions[tmp_indices, 13] = 640 - regions[tmp_indices, 4] - regions[tmp_indices, 14]
        tmp_indices = np.where(regions[:, 13] + face_y + regions[:, 15] > 640)[0]
        regions[tmp_indices, 13] = 640 - regions[tmp_indices, 5] - regions[tmp_indices, 15]
        tmp_indices = np.where((regions[:, -1] < 3) &
                               (regions[:, 13] + face_x + regions[:, 14] > 480))[0]
        regions[tmp_indices, 13] = 480 - regions[tmp_indices, 4] - regions[tmp_indices, 14]
        tmp_indices = np.where((regions[:, -1] > 2) &
                               (regions[:, 13] + face_y + regions[:, 15] > 480))[0]
        regions[tmp_indices, 13] = 480 - regions[tmp_indices, 5] - regions[tmp_indices, 15]
    if (config.regions == 'fan'):
        # for FAN regions
        tmp_indices = np.where(regions[:, 7] + regions[:, 9] > 640)[0]
        regions[tmp_indices, 7] = 640 - regions[tmp_indices, 9]
        tmp_indices = np.where(regions[:, 7] + regions[:, 10] > 640)[0]
        regions[tmp_indices, 7] = 640 - regions[tmp_indices, 10]
        tmp_indices = np.where((regions[:, 24] < 3) & (regions[:, 7] + regions[:, 9] > 480))[0]
        regions[tmp_indices, 7] = 480 - regions[tmp_indices, 9]
        tmp_indices = np.where((regions[:, 24] > 2) & (regions[:, 7] + regions[:, 10] > 480))[0]
        regions[tmp_indices, 7] = 480 - regions[tmp_indices, 10]
        regions[:, 8] = regions[:, 7]

        tmp_indices = np.where((regions[:, 12] + regions[:, 14] > 640))[0]
        regions[tmp_indices, 12] = 640 - regions[tmp_indices, 14]
        tmp_indices = np.where(regions[:, 12] + regions[:, 15] > 640)[0]
        regions[tmp_indices, 12] = 640 - regions[tmp_indices, 15]
        tmp_indices = np.where((regions[:, 24] < 3) & (regions[:, 12] + regions[:, 14] > 480))[0]
        regions[tmp_indices, 12] = 480 - regions[tmp_indices, 14]
        tmp_indices = np.where((regions[:, 24] > 2) & (regions[:, 12] + regions[:, 15] > 480))[0]
        regions[tmp_indices, 12] = 480 - regions[tmp_indices, 15]
        regions[:, 13] = regions[:, 12]

    return dots, regions, df_info


######################################## /*ONE-TIME RUN TO PROCESS GAZECAPTURE META DATA ##########################################################################
def extract_file(path='./', des_path='./'):
    '''Extract raw data from compressed files'''
    files = [f for f in glob.glob(path + "**/*.tar.gz", recursive=True)]
    for f in files[:-1]:
        print(f)
        if (f.endswith("tar.gz")):
            tar = tarfile.open(f, "r:gz")
            tar.extractall(des_path)
            tar.close()


def aggregate_meta_data(path='./', des_path='./'):
    dirs = [d for d in glob.glob(path + "/*/", recursive=True)]
    print(len(dirs))
    info_list = []  # TotalFrames, NumFaceDetections, NumEyeDetections, Dataset, DeviceName
    for d in dirs:
        file_path = d + 'info.json'

        with open(file_path) as json_file:
            data = json.load(json_file)
            # print(type(data), data)
            subjectID = int(file_path.split('/')[-2])
            # print(subjectID)
            info_list.append([subjectID] + list(data.values()))
    print(len(info_list))
    df_info = pd.DataFrame(info_list)
    print(df_info.iloc[:20, :])
    df_info.columns = ['subjectID', 'TotalFrames', 'NumFaceDetections',
                       'NumEyeDetections', 'Dataset', 'DeviceName']
    df_info['isMobile'] = 0
    # print(df_info.iloc[0:10,:])
    for device in pd.unique(df_info.loc[:, 'DeviceName']):
        if (device[:3] == 'iPh'):
            df_info.loc[df_info.loc[:, 'DeviceName'] == device, 'isMobile'] = 1

    df_info.to_csv(des_path + 'df_info.csv', index=False)


def aggregate_dot_data(path='./', des_path='./'):
    dirs = [d for d in glob.glob(path + "/*/", recursive=True)]
    # dots fortmat: subjectID, frameID, DotNum, Xpts, YPts, XCam, YCam, Time
    dots = np.zeros((2500000, 8), dtype=np.float32)
    count = 0
    for d in dirs:
        file_path = d + 'dotInfo.json'
        # print(d)
        with open(file_path) as json_file:
            data = json.load(json_file)
            dotInfo = np.transpose(np.array(list(data.values())))
            # tmp = np.copy(dotInfo[:,1])
            # dotInfo[:,1] = dotInfo[:,4]
            # dotInfo[:,4] = tmp
            subjectID = int(file_path.split('/')[-2])
            # print(subjectID)
            subjectID_col = np.ones((len(dotInfo), 1)) * subjectID
            frameID_col = np.arange(0, len(dotInfo))
            frameID_col = np.expand_dims(frameID_col, axis=1)
            # print(dotInfo.shape,subjectID_col.shape, frameID_col.shape)
            subjectID_col, frameID_col
            dotInfo = np.hstack((subjectID_col, frameID_col, dotInfo))
            dots[count:count+len(dotInfo), :] = dotInfo
            count += len(dotInfo)
    dots = dots[:count, ]
    np.save(des_path + 'dots.npy', dots)


def aggregate_region_data(path='./', des_path='./'):
    # Apple face and eye region
    # subjectID, frameID, face(x,y,w,h,valid), left_eye(), right_eye(), face_grid(x,y,w,h,valid)
    faceRegions = np.zeros((2500000, 22), dtype=np.float32)
    dirs = [d for d in glob.glob(path + "/*/", recursive=True)]
    count = 0

    for d in dirs:
        face_path = d + 'appleFace.json'
        Leye_path = d + 'appleLeftEye.json'
        Reye_path = d + 'appleRightEye.json'
        grid_path = d + 'faceGrid.json'

        with open(face_path) as json_file:
            data = json.load(json_file)
            faceInfo = np.transpose(np.array(list(data.values())))
            # tmp = np.copy(faceInfo[:,-1])
            # faceInfo[:,-1] = faceInfo[:,-2]
            # faceInfo[:,-2] = tmp
            subjectID = int(face_path.split('/')[-2])

            subjectID_col = np.ones((len(faceInfo), 1)) * subjectID
            frameID_col = np.arange(0, len(faceInfo))
            frameID_col = np.expand_dims(frameID_col, axis=1)

            faceInfo = np.hstack((subjectID_col, frameID_col, faceInfo))

            with open(Leye_path) as json_file:
                data = json.load(json_file)
                LeyeInfo = np.transpose(np.array(list(data.values())))
                # tmp = np.copy(LeyeInfo[:,-1])
                # LeyeInfo[:,-1] = LeyeInfo[:,-2]
                # LeyeInfo[:,-2] = tmp
                faceInfo = np.hstack((faceInfo, LeyeInfo))

            with open(Reye_path) as json_file:
                data = json.load(json_file)
                ReyeInfo = np.transpose(np.array(list(data.values())))
                faceInfo = np.hstack((faceInfo, ReyeInfo))

            with open(grid_path) as json_file:
                data = json.load(json_file)
                grid = np.transpose(np.array(list(data.values())))
                faceInfo = np.hstack((faceInfo, grid))

            faceRegions[count:count+len(faceInfo), :] = faceInfo
            count += len(faceInfo)
#             print(count)
    faceRegions = faceRegions[:count, :]
    faceRegions.shape
    np.save(des_path+'regions.npy', faceRegions)


def aggregate_screen_data(path='./', des_path='./'):
    # create a dict with subjectID key and screen orientation value as one-hot vector
    # add orienetation and screen size
    faceRegions = np.load(des_path+'regions.npy')
    df_info = pd.read_csv(des_path+'df_info.csv')
    subjectID_list = list(df_info.loc[:, 'subjectID'])

    screenInfo = np.empty((1, 3), dtype=np.float32)
    for subjectID in subjectID_list:
        #         print('subjectID',subjectID)
        screen_path = path + '{:0>5d}'.format(int(subjectID)) + '/screen.json'

        with open(screen_path) as json_file:
            data = json.load(json_file)
            screenInfo = np.vstack((screenInfo, np.transpose(np.array(list(data.values())))))
    screenInfo = screenInfo[1:, ]

    faceRegions = np.hstack((faceRegions, screenInfo))
    np.save(des_path+'regions_ori.npy', faceRegions)
