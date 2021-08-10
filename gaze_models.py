import tensorflow as tf
if (tf.version.VERSION == '2.3.0'):
    from tensorflow.keras.mixed_precision import experimental as mixed_precision
else:
    from tensorflow.keras import mixed_precision
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate, SeparableConv2D, DepthwiseConv2D, Input, GlobalMaxPooling2D, Activation, Conv2D, Conv3D, Reshape, AveragePooling3D, AveragePooling2D, GlobalAveragePooling3D, GlobalAveragePooling2D, GlobalAveragePooling1D, MaxPooling2D, LSTM, Embedding, Dense, Dropout, Flatten, BatchNormalization, add, UpSampling2D, Conv2DTranspose
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4
from losses import euclidean_loss, heatmap_loss
import config


def conv_bn(input_x, conv):
    x = conv(input_x)
    x = Activation('relu')(x)
    bn_name = 'single_v10_' + input_x.name + '_' + conv.name + '_bn'
    bn_name = bn_name.replace(':', '_')
    x = BatchNormalization(name=bn_name)(x)
    return x


def relu_bn(input_x):
    x = Activation('relu')(input_x)
    bn_name = 'single_v10_' + input_x.name + '_bn'
    bn_name = bn_name.replace(':', '_')
    x = BatchNormalization(name=bn_name)(x)
    return x


policy = mixed_precision.Policy('mixed_float16')
if (tf.version.VERSION == '2.3.0'):
    mixed_precision.set_policy(policy)
else:
    mixed_precision.set_global_policy(policy)


grid_dense1 = Dense(256, activation="relu", name='grid_dense1')
grid_dense2 = Dense(128, activation="relu", name='grid_dense2')
grid_dense3 = Dense(128, activation="relu", name='grid_dense3')
heatmap_conv1 = Conv2D(1, (7, 7), padding='same', name='heatmap_conv1')
heatmap_conv2 = Conv2D(1, (7, 7), padding='same', name='heatmap_conv2')
heatmap_conv3 = Conv2D(1, (3, 3), padding='same', name='heatmap_conv3')


def get_SAGE(base_model='MobileNetV2', heatmap=False):
    base_model_list = ['AlexNet', 'MobileNetV2', 'EfficientNetB0', 'EfficientNetB1',
                       'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4']
    if (base_model not in base_model_list):
        print('base_model --' + base_model + '-- does not exist')

    # print('Compute dtype: %s' % policy.compute_dtype,
    #       'Variable dtype: %s' % policy.variable_dtype)

    dropout_rate = 0.5

    '''SAGE architecture'''
    eyeIm_shape = (config.eyeIm_size, config.eyeIm_size, config.channel)
    input_leye = Input(shape=eyeIm_shape, name='input_leye')
    input_reye = Input(shape=eyeIm_shape, name='input_reye')
    input_eyelandmark = Input(shape=(6,), name='input_eyelandmark')
    input_orientation = Input(shape=(3,), name='input_orientation')

    if base_model == 'AlexNet':
        eye_conv1 = Conv2D(96, (11, 11), strides=2, name='eye_conv1')
        eye_conv2 = Conv2D(256, (5, 5), name='eye_conv2')
        eye_conv3 = Conv2D(384, (3, 3), name='eye_conv3')
        eye_conv4 = Conv2D(64, (1, 1), name='eye_conv4')
        eye_dense1 = Dense(128, activation="relu", name='eye_dense1')

        # left eye
        leye = eye_conv1(input_leye)
        leye = Activation('relu')(leye)
        leye = MaxPooling2D(pool_size=(3, 3), strides=2)(leye)
        leye = BatchNormalization(name='bn_leye_conv1')(leye)  # momentum=0.75, epsilon=0.0001
        leye = eye_conv2(leye)
        leye = Activation('relu')(leye)
        leye = MaxPooling2D(pool_size=(3, 3), strides=2)(leye)
        leye = BatchNormalization(name='bn_leye_conv2')(leye)
        leye = eye_conv3(leye)
        leye = Activation('relu')(leye)
        leye = BatchNormalization(name='bn_leye_conv3')(leye)
        leye = eye_conv4(leye)
        leye = Activation('relu')(leye)
        leye = BatchNormalization(name='bn_leye_conv4')(leye)
        leye = Flatten()(leye)
        leye = eye_dense1(leye)
        leye = BatchNormalization(name='bn_eye_dense')(leye)

    else:
        leye_block = globals()[base_model](input_shape=eyeIm_shape, include_top=False,
                                           weights=config.weights, input_tensor=input_leye)
        # rename layers to avoid duplicated name
        for layer in leye_block.layers:
            layer._name = 'leye_' + layer.name

        leye_dense = Dense(128, activation="relu", name='leye_dense')
        leye = leye_block.output
        leye = GlobalAveragePooling2D()(leye)
        leye = Dropout(dropout_rate)(leye)
        leye = BatchNormalization()(leye)
        leye = leye_dense(leye)
        leye = BatchNormalization()(leye)

    eye_model = Model(input_leye, leye)
    reye = eye_model(input_reye)

    # eye_model.trainable = False

    landmark_dense1 = Dense(64, activation="relu", name='lm_dense1')
    landmark_dense2 = Dense(128, activation="relu", name='lm_dense2')
    landmark_dense3 = Dense(16, activation="relu", name='lm_dense3')
    landmark = concatenate([input_orientation, input_eyelandmark])
    landmark = landmark_dense1(landmark)
    landmark = BatchNormalization()(landmark)
    landmark = landmark_dense2(landmark)
    landmark = BatchNormalization()(landmark)
    landmark = landmark_dense3(landmark)
    landmark = BatchNormalization()(landmark)
    # landmark = Dropout(0.5)(landmark)
    # merge
    if (config.heatmap):
        merge_dense1 = Dense(128, activation="relu", name='hm_merge_dense1')
        merge_dense2 = Dense(128, activation='relu', name='hm_merge_dense2')
        merge = concatenate([input_orientation, landmark, leye, reye])
        merge = merge_dense1(merge)
        merge = BatchNormalization(name='bn_hm_merge_dense1')(merge)
        merge = merge_dense2(merge)
        merge = BatchNormalization(name='bn_hm_merge_dense2')(merge)
        heatmap_dense = Dense(int(config.hm_size**2/4),
                              activation='relu', name='heatmap_dense')
        merge = heatmap_dense(merge)
        merge = BatchNormalization(name='bn_hm_dense')(merge)
        heatmap = Reshape(target_shape=(
            int(config.hm_size/2), int(config.hm_size/2), 1))(merge)
        heatmap = conv_bn(heatmap, heatmap_conv1)
        heatmap = UpSampling2D()(heatmap)
        heatmap = heatmap_conv2(heatmap)
        heatmap = Activation('relu', dtype='float32')(heatmap)

        model = Model(inputs=[input_orientation, input_eyelandmark, input_leye,
                              input_reye], outputs=heatmap)
        model.compile(loss=heatmap_loss, optimizer='adam')
    else:
        merge_dense1 = Dense(128, activation="relu", name='merge_dense1')
        merge_dense2 = Dense(2, name='merge_dense2')
        merge = concatenate([input_orientation, landmark, leye, reye])
        merge = merge_dense1(merge)
        merge = BatchNormalization(name='bn_merge_dense1')(merge)
        merge = concatenate([input_orientation, merge])
        # merge = Dropout(0.2)(merge)
        merge = merge_dense2(merge)
        merge = Activation('linear', dtype='float32')(merge)

        model = Model(inputs=[input_orientation, input_eyelandmark, input_leye,
                              input_reye], outputs=merge)
        model.compile(loss=euclidean_loss, optimizer='adam')
    return model


def get_iTracker(orientation=False):
    print('Compute dtype: %s' % policy.compute_dtype)
    print('Variable dtype: %s' % policy.variable_dtype)

    dropout_rate = 0.75

    ### iTracker architecture ###
    input_face = Input(shape=(config.faceIm_size,
                              config.faceIm_size, config.channel), name='input_face')
    input_leye = Input(
        shape=(config.eyeIm_size, config.eyeIm_size, config.channel), name='input_leye')
    input_reye = Input(
        shape=(config.eyeIm_size, config.eyeIm_size, config.channel), name='input_reye')
    input_grid = Input(
        shape=(faceGrid_size, faceGrid_size, 1), name='input_grid')
    if orientation:
        input_orientation = Input(shape=(3,), name='input_orientation')

    eye_conv1 = Conv2D(96, (11, 11), strides=4, name='eye_conv1')
    eye_conv2 = Conv2D(256, (5, 5), name='eye_conv2')
    eye_conv3 = Conv2D(384, (3, 3), name='eye_conv3')
    eye_conv4 = Conv2D(64, (1, 1), name='eye_conv4')
    eye_dense1 = Dense(128, activation="relu", name='eye_dense1')

    face_conv1 = Conv2D(96, (11, 11), strides=4, name='face_conv1')
    face_conv2 = Conv2D(256, (5, 5), name='face_conv2')
    face_conv3 = Conv2D(384, (3, 3), name='face_conv3')
    face_conv4 = Conv2D(64, (1, 1), name='face_conv4')
    face_dense1 = Dense(128, activation="relu", name='face_dense1')
    face_dense2 = Dense(64, activation="relu", name='face_dense2')

    grid_dense1 = Dense(256, activation="relu", name='grid_dense1')
    grid_dense2 = Dense(128, activation="relu", name='grid_dense2')

    # left eye
    leye = eye_conv1(input_leye)
    leye = Activation('relu')(leye)
    leye = MaxPooling2D(pool_size=(3, 3), strides=2)(leye)
    leye = BatchNormalization(name='bn_leye_conv1')(
        leye)  # momentum=0.75, epsilon=0.0001
    leye = eye_conv2(leye)
    leye = Activation('relu')(leye)
    leye = MaxPooling2D(pool_size=(3, 3), strides=2)(leye)
    leye = BatchNormalization(name='bn_leye_conv2')(leye)
    leye = eye_conv3(leye)
    leye = Activation('relu')(leye)
    leye = BatchNormalization(name='bn_leye_conv3')(leye)
    leye = eye_conv4(leye)
    leye = Activation('relu')(leye)
    leye = BatchNormalization(name='bn_leye_conv4')(leye)

    # right eye
    reye = eye_conv1(input_reye)
    reye = Activation('relu')(reye)
    reye = MaxPooling2D(pool_size=(3, 3), strides=2)(reye)
    reye = BatchNormalization(name='bn_reye_conv1')(reye)
    reye = eye_conv2(reye)
    reye = Activation('relu')(reye)
    reye = MaxPooling2D(pool_size=(3, 3), strides=2)(reye)
    reye = BatchNormalization(name='bn_reye_conv2')(reye)
    reye = eye_conv3(reye)
    reye = Activation('relu')(reye)
    reye = BatchNormalization(name='bn_reye_conv3')(reye)
    reye = eye_conv4(reye)
    reye = Activation('relu')(reye)
    reye = BatchNormalization(name='bn_reye_conv4')(reye)

    eyes = concatenate([Dropout(dropout_rate)(leye), Dropout(
        dropout_rate)(reye)])  # by default using axis=-1
    eyes = Flatten()(eyes)
    eyes = eye_dense1(eyes)
    eyes = BatchNormalization(name='bn_eye_dense')(eyes)

    # face
    face = face_conv1(input_face)
    face = Activation('relu')(face)
    face = MaxPooling2D(pool_size=(3, 3), strides=2)(face)
    face = BatchNormalization(name='bn_face_conv1')(face)
    face = face_conv2(face)
    face = Activation('relu')(face)
    face = MaxPooling2D(pool_size=(3, 3), strides=2)(face)
    face = BatchNormalization(name='bn_face_conv2')(face)
    face = face_conv3(face)
    face = Activation('relu')(face)
    face = BatchNormalization(name='bn_face_conv3')(face)
    face = face_conv4(face)
    face = Activation('relu')(face)
    face = BatchNormalization(name='bn_face_conv4')(face)

    face = Flatten()(face)
    face = Dropout(dropout_rate)(face)
    face = face_dense1(face)
    face = BatchNormalization(name='bn_face_dense1')(face)
    face = face_dense2(face)
    face = BatchNormalization(name='bn_face_dense2')(face)

    # face grid
    grid = Flatten()(input_grid)
    grid = Dropout(dropout_rate)(grid)
    grid = grid_dense1(grid)
    grid = BatchNormalization(name='bn_grid_dense1')(grid)
    grid = grid_dense2(grid)
    grid = BatchNormalization(name='bn_grid_dense2')(grid)

    # merge
    dense1 = Dense(128, activation="relu", name='dense1')
    dense2 = Dense(2, name='dense2')
    if orientation:
        merge = concatenate([input_orientation, eyes, face, grid])
    else:
        merge = concatenate([eyes, face, grid])
    merge = dense1(merge)
    merge = BatchNormalization(name='merge_dense1')(merge)
    if orientation:
        merge = concatenate([input_orientation, merge])
    merge = dense2(merge)
    merge = Activation('linear', dtype='float32')(merge)

    if orientation:
        model = Model(inputs=[input_orientation, input_leye, input_reye, input_face, input_grid],
                      outputs=merge)
    else:
        model = Model(inputs=[input_leye, input_reye, input_face, input_grid],
                      outputs=merge)
    model.compile(loss=euclidean_loss, optimizer='adam')
    return model
