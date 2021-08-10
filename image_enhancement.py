import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import concatenate, SeparableConv2D, DepthwiseConv2D, Input, GlobalMaxPooling2D, Activation, Conv2D, Conv3D, Reshape, AveragePooling3D, AveragePooling2D, GlobalAveragePooling3D, GlobalAveragePooling2D, GlobalAveragePooling1D, MaxPooling2D, LSTM, Embedding, Dense, Dropout, Flatten, BatchNormalization, add, UpSampling2D, Conv2DTranspose
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import generator
from sklearn.utils import shuffle
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
import tensorflow.keras.backend as K
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input


def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)


def get_gram_matrix_loss(pred, label):
    pred_gram = gram_matrix(pred)
    label_gram = gram_matrix(label)
    return K.mean(K.abs(pred_gram - label_gram))


def get_resnet_model():
    model = ResNet50V2(weights='imagenet', include_top=False)
    # for i, layer in enumerate(model.layers):
    #     print(i, layer.name)
    layer_indices = [9, 36, 82, 150, 185]  # activation indices ~ scale 0.015
    layer_indices = [37, 83, 151]  # max_pool indices ~ scale 0.018
    outputs = None
    for idx in layer_indices:
        features = model.layers[idx].output
        features = Flatten()(features)
        if (idx == layer_indices[0]):
            outputs = features
        else:
            outputs = concatenate([outputs, features])
    resnet_model = Model(model.inputs, outputs)
    return resnet_model


def get_vgg_model():
    model = VGG16(weights='imagenet', include_top=False)
    # for i, layer in enumerate(model.layers):
    #     print(i, layer.name)
    layer_indices = [2, 5, 9, 13, 17]  # activation indices ~ scale 6
    layer_indices = [3, 6, 10, 14, 18]  # activation indices ~ scale 0.6
    outputs = None
    for idx in layer_indices:
        features = model.layers[idx].output
        features = Flatten()(features)
        if (idx == layer_indices[0]):
            outputs = features
        else:
            outputs = concatenate([outputs, features])
    vgg_model = Model(model.inputs, outputs)
    return vgg_model


vgg_model = get_vgg_model()
resnet_model = get_resnet_model()


@tf.function
def custom_loss(pred, label):
    pred_vgg_input = tf.keras.applications.vgg16.preprocess_input(pred)
    label_vgg_input = tf.keras.applications.vgg16.preprocess_input(label)
    pred_vgg_features = vgg_model(pred_vgg_input)
    label_vgg_features = vgg_model(label_vgg_input)
    vgg_loss = K.mean(K.abs(pred_vgg_features - label_vgg_features))/0.6

    pred_resnet_input = tf.keras.applications.resnet_v2.preprocess_input(pred)
    label_resnet_input = tf.keras.applications.resnet_v2.preprocess_input(label)
    pred_resnet_features = resnet_model(pred_resnet_input)
    label_resnet_features = resnet_model(label_resnet_input)
    resnet_loss = K.mean(K.abs(pred_resnet_features - label_resnet_features))/0.018

    gram_matrix_loss = get_gram_matrix_loss(pred, label)/0.065
    pixel_loss = K.mean(K.abs(pred - label))/0.065
    return 100*(3*vgg_loss + 5*resnet_loss + pixel_loss + gram_matrix_loss)


def conv_block(x, filters):
    x = Activation("relu")(x)
    x = SeparableConv2D(filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    return x


def get_model(img_size, num_classes, dropout=True, link=True):
    # policy = mixed_precision.Policy('mixed_float16')
    # mixed_precision.set_policy(policy)
    # print('Compute dtype: %s' % policy.compute_dtype,
    #       'Variable dtype: %s' % policy.variable_dtype)

    inputs = Input(shape=img_size + (3,), name='input')

    '''[First half of the network: downsampling inputs]'''
    # Entry block
    x = Conv2D(64, 3, strides=2, padding="same")(inputs)  # 112x32
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = conv_block(x, 64)
    x = conv_block(x, 64)

    if (dropout):
        x = Dropout(0.2)(x)
    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    residuals = []
    residuals.append(x)
    for filters in [128, 256, 512]: #[64, 96, 128]
        for layer_no in range(20):
            x = conv_block(x, filters)
        x = MaxPooling2D(3, strides=2, padding="same")(x)  # 56x64, 28x128, 14x256

        # Project residual
        residual = Conv2D(filters, 1, strides=2, padding="same")(previous_block_activation)
        x = add([x, residual])  # Add back residual
        if (dropout):
            x = Dropout(0.2)(x)
        residuals.append(x)
        previous_block_activation = x  # Set aside next residual

    '''[Second half of the network: upsampling inputs]'''
    for i, filters in enumerate([512, 256, 128, 64]): #[128, 96, 64, 64]
        for layer_no in range(20):
            x = conv_block(x, filters)
        if (link):
            x = add([x, residuals[3-i]])
        x = UpSampling2D(2)(x)

        # Project residual
        residual = UpSampling2D(2)(previous_block_activation)
        residual = Conv2D(filters, 1, padding="same")(residual)

        x = add([x, residual])  # Add back residual
        if (dropout):
            x = Dropout(0.2)(x)

        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = Conv2D(num_classes, 3, padding="same")(x)
    outputs = Activation('linear', dtype='float32')(outputs)

    # Define the model
    model = Model(inputs, outputs)
    return model


def resize(dir_path, size):
    origin_path = dir_path + '/1024'
    des_path = dir_path + '/' + str(size) + '/'
    crappified_path = dir_path + '/crappified_' + str(size) + '/'
    if(not os.path.exists(des_path)):
        os.makedirs(des_path)
    if(not os.path.exists(crappified_path)):
        os.makedirs(crappified_path)
    file_list = os.listdir(origin_path)
    for f in file_list[:]:
        file_path = origin_path + '/' + f
        im = np.array(Image.open(file_path), dtype=np.uint8)
        crappified_im = (cv2.resize(im, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)).astype(np.uint8)
        for tmp_im, prefix in [(im, ''), (crappified_im, 'crappified_')]:
            resized_im = (cv2.resize(tmp_im, dsize=(size, size), interpolation=cv2.INTER_CUBIC)).astype(np.uint8)
            resized_im = cv2.cvtColor(resized_im, cv2.COLOR_BGR2RGB)
            cv2.imwrite(dir_path + '/' + prefix + str(size) + '/' + prefix + f, resized_im, [cv2.IMWRITE_JPEG_QUALITY, 100])


def crappify(dir_path, size):
    origin_path = dir_path + '/' + str(size)
    crappified_path = dir_path + '/crappified_' + str(size) + '/'
    if(not os.path.exists(crappified_path)):
        os.makedirs(crappified_path)

    file_list = os.listdir(origin_path)
    for f in file_list[:]:
        file_path = origin_path + '/' + f
        im = np.array(Image.open(file_path), dtype=np.uint8)
        crappified_im = (cv2.resize(im, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)).astype(np.uint8)

        # '''Add glare'''
        # gl_x = np.random.randint(1, 50)
        # gl_y = np.random.randint(1, 50)
        # gl_x_size = np.random.randint(15, 35)
        # gl_y_size = np.random.randint(15, 35)
        # gl_x_size = (63 - gl_x) if (gl_x + gl_x_size > 63) else gl_x_size
        # gl_y_size = (63 - gl_y) if (gl_y + gl_y_size > 63) else gl_y_size
        # crappified_im[gl_x:gl_x+gl_x_size, gl_y:gl_y+gl_y_size, 0] = np.random.randint(100, 255)
        # crappified_im[gl_x:gl_x+gl_x_size, gl_y:gl_y+gl_y_size, 1] = np.random.randint(100, 255)
        # crappified_im[gl_x:gl_x+gl_x_size, gl_y:gl_y+gl_y_size, 2] = np.random.randint(100, 255)

        '''Add blur'''
        blur_mode = np.random.randint(0, 4)
        kernel_size = 3
        if (blur_mode == 0):
            crappified_im = cv2.blur(crappified_im, (kernel_size, kernel_size))
        elif (blur_mode == 1):
            crappified_im = cv2.medianBlur(crappified_im, kernel_size)
        elif (blur_mode == 2):
            crappified_im = cv2.GaussianBlur(crappified_im, (kernel_size, kernel_size), 0)
        '''Add motion blur'''
        kernel_v_size = np.random.randint(1, 32)
        kernel_h_size = np.random.randint(1, 32)
        kernel_o_size = np.random.randint(1, 32)
        v_kernel = np.zeros((kernel_v_size, kernel_v_size))
        h_kernel = np.zeros((kernel_h_size, kernel_h_size))
        o_kernel = np.zeros((kernel_o_size, kernel_o_size))
        # Fill the middle row with ones.
        v_kernel[:, int((kernel_v_size - 1)/2)] = np.ones(kernel_v_size)
        h_kernel[int((kernel_h_size - 1)/2), :] = np.ones(kernel_h_size)
        o_kernel[:, int((kernel_o_size - 1)/2)] = np.ones(kernel_o_size)
        o_kernel[int((kernel_o_size - 1)/2), :] = np.ones(kernel_o_size)
        # Normalize.
        v_kernel /= kernel_v_size
        h_kernel /= kernel_h_size
        o_kernel /= np.sum(o_kernel)

        motion_blur_mode = np.random.randint(1, 3)
        if (motion_blur_mode == 0):
            mb = cv2.filter2D(crappified_im, -1, v_kernel)
        elif (motion_blur_mode == 1):
            mb = cv2.filter2D(crappified_im, -1, h_kernel)
        elif (motion_blur_mode == 2):
            mb = cv2.filter2D(crappified_im, -1, o_kernel)
        else:
            mb = crappified_im

        resized_im = (cv2.resize(mb, dsize=(size, size), interpolation=cv2.INTER_CUBIC)).astype(np.uint8)
        resized_im = cv2.cvtColor(resized_im, cv2.COLOR_BGR2RGB)
        jpeg_quality = np.random.randint(10, 30)
        cv2.imwrite(dir_path + '/crappified_' + str(size) + '/' + f, resized_im, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])



def load_data(paths, im_size):
    no_sample = len(paths)
    ims = np.zeros((no_sample, im_size, im_size, 3))
    for i in range(no_sample):
        im = np.array(Image.open(paths[i]), dtype=np.float32)/255.
        ims[i, :, :, :] = im
    return ims


def process_path(input_path, label_path):
    precision_type = tf.float32
    input_im = tf.io.decode_image(tf.io.read_file(input_path))
    label_im = tf.io.decode_image(tf.io.read_file(label_path))
    input_im = tf.image.convert_image_dtype(input_im, precision_type)
    label_im = tf.image.convert_image_dtype(label_im, precision_type)
    return (input_im, label_im)


def make_tf_dataset(input_paths, label_paths):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    list_ds = tf.data.Dataset.from_tensor_slices((input_paths, label_paths))
    ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    return ds


def get_file_list(path, im_size, no_val_samples=128):
    des_path = path + '/' + str(im_size) + '/'
    crappified_path = path + '/crappified_' + str(im_size) + '/'
    file_list = os.listdir(des_path)
    input_paths = []
    label_paths = []
    for f in file_list:
        # input_paths.append(crappified_path + 'crappified_' + f)
        input_paths.append(crappified_path + f)
        label_paths.append(des_path + f)

    train_input_paths = input_paths[:-no_val_samples]
    train_label_paths = label_paths[:-no_val_samples]
    val_input_paths = input_paths[-no_val_samples:]
    val_label_paths = label_paths[-no_val_samples:]

    return train_input_paths, train_label_paths, val_input_paths, val_label_paths


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr


def main():
    print('---SUPERRES---')
    path = 'PATH_TO_CELEB_DATASET'
    im_size = 112
    crappify(path, im_size)
    # return 0
    # resize(path, im_size)
    strategy = tf.distribute.MirroredStrategy()  # multiple gpus
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    # learning rate
    adam = tf.keras.optimizers.Adam(learning_rate=0.001)
    lr_metric = get_lr_metric(adam)
    with strategy.scope():
        model = get_model((im_size, im_size), 3, dropout=True, link=True)
        # model.compile(optimizer='adam', loss='mean_squared_error')
        model.compile(optimizer=adam, loss=custom_loss, metrics=[lr_metric])
        model.save('MODEL.h5')
    # print(model.summary())
    '''Show model graph'''
    # dot_im_file = 'model.png'
    # tf.keras.utils.plot_model(model, to_file=dot_im_file)
    # model_im = Image.open('model.png')
    # model_im.show()

    batch_size = 32
    train_inputs, train_labels, val_inputs, val_labels = get_file_list(path, im_size)
    test_path = 'PATH_TO_TEST_IMAGES' + str(im_size) + '/'
    test_inputs = [test_path + f for f in os.listdir(test_path)]
    print('train size:', len(train_inputs), ' val size:', len(val_inputs))
    val_input_ims = load_data(val_inputs, im_size)
    val_label_ims = load_data(val_labels, im_size)
    test_input_ims = load_data(test_inputs, im_size)
    best_val_loss = 800
    for iteration in range(1000):
        '''Call crappify once in awhile'''
        train_inputs, train_labels = shuffle(train_inputs, train_labels)
        train_ds = make_tf_dataset(train_inputs, train_labels)
        train_generator = generator.TFDataFeeder(train_ds, batch_size=batch_size,
                                                 dataset_len=len(train_inputs))
        model.fit(train_generator.reset(), steps_per_epoch=200, epochs=1)
        val_loss = model.evaluate(x=val_input_ims, y=val_label_ims, verbose=1)
        if (val_loss[0] < best_val_loss):
            best_val_loss = val_loss[0]
            val_preds = model.predict(val_input_ims)
            test_preds = model.predict(test_input_ims)
            for i in range(20):
                val_pred = val_preds[i, :, :, :][:, :, [2, 1, 0]]
                target = val_label_ims[i, :, :, :][:, :, [2, 1, 0]]
                origin = val_input_ims[i, :, :, :][:, :, [2, 1, 0]]
                combined_im = np.concatenate([origin, val_pred, target], axis=1)*255
                # cv2.imwrite(path + '/pred/' + str(i) + '_' + str(int(best_val_loss))
                            # + '_.jpg', combined_im)

                test_pred = test_preds[i, :, :, :][:, :, [2, 1, 0]]
                origin = test_input_ims[i, :, :, :][:, :, [2, 1, 0]]
                combined_im = np.concatenate([origin, test_pred], axis=1)*255
                cv2.imwrite(path + '/hr_' + str(im_size) + '/' + '_' + str(i) + '_' + str(int(best_val_loss))
                            + '_.jpg', combined_im)
            model.save_weights('MODEL_WEIGHTS.hdf5')

        # update learning rate
        K.set_value(model.optimizer.learning_rate, val_loss[1]*0.988)
        if (iteration % 50 == 0 and iteration > 0):
            print('crappifying')
            crappify(path, im_size)


def test():
    im_size = 224
    test_path = 'PATH_TO_TEST_IMAGE'
    test_inputs = [test_path + f for f in os.listdir(test_path)]
    # load model

    strategy = tf.distribute.MirroredStrategy()  # multiple gpus
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    adam = tf.keras.optimizers.Adam(learning_rate=0.001)
    lr_metric = get_lr_metric(adam)
    with strategy.scope():
        model = get_model((im_size, im_size), 3, dropout=True, link=True)
        # model = load_model('/home/sinh/Downloads/celeb/model/do_link_128_96_64.h5')
        model.load_weights('MODEL_WEIGHTS.hdf5', by_name=True)
        model.compile(optimizer=adam, loss=custom_loss, metrics=[lr_metric])

    print(len(test_inputs))
    current_idx = 0
    batch_size = 250
    while (True):  # (current_idx + 1000 < len(test_inputs)):
        print('current_idx', current_idx)
        test_input_ims = load_data(test_inputs[current_idx: current_idx+batch_size], im_size)
        test_preds = model.predict(test_input_ims)
        for i in range(batch_size):
            frameID = test_inputs[current_idx + i]
            frameID = frameID.replace('lr', 'hr')
            test_pred = test_preds[i, :, :, :][:, :, [2, 1, 0]]*255
            # origin = test_input_ims[i, :, :, :][:, :, [2, 1, 0]]
            # combined_im = np.concatenate([origin, test_pred], axis=1)*255
            cv2.imwrite(frameID, test_pred)
        current_idx += batch_size
        # break


def save_model():
    import coremltools
    im_size = 224
    model = get_model((im_size,im_size), 3, dropout=True, link=True)
    model_input = coremltools.converters.ImageType('input', shape=(1,im_size,im_size,3))
    coreml_model = coremltools.converters.convert(model, source='tensorflow', inputs=[model_input])
    coreml_model.save('/model/coreml/COREML_MODEL.mlmodel')


if __name__ == '__main__':
    # main()
    # test()
    save_model()
