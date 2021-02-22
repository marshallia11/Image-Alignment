import numpy as np
# from tensorflow.keras import backend as keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout,Cropping2D, Conv2DTranspose, UpSampling2D,Input
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import cv2
import json
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import concatenate
from tensorflow.keras.optimizers import Adam
import util
# def get_model(img_size,num_classes):
#     inputs = layers.Input([843, 1055, 1])
#
#     ### [First half of the network: downsampling inputs] ###
#
#     # Entry block
#     x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation("relu")(x)
#
#     previous_block_activation = x  # Set aside residual
#
#     # Blocks 1, 2, 3 are identical apart from the feature depth.
#     for filters in [64, 128, 256]:
#         x = layers.Activation("relu")(x)
#         x = layers.SeparableConv2D(filters, 3, padding="same")(x)
#         x = layers.BatchNormalization()(x)
#
#         x = layers.Activation("relu")(x)
#         x = layers.SeparableConv2D(filters, 3, padding="same")(x)
#         x = layers.BatchNormalization()(x)
#
#         x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
#
#         # Project residual
#         residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
#             previous_block_activation
#         )
#         x = layers.add([x, residual])  # Add back residual
#         previous_block_activation = x  # Set aside next residual
#
#     ### [Second half of the network: upsampling inputs] ###
#
#     for filters in [256, 128, 64, 32]:
#         x = layers.Activation("relu")(x)
#         x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
#         x = layers.BatchNormalization()(x)
#
#         x = layers.Activation("relu")(x)
#         x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
#         x = layers.BatchNormalization()(x)
#
#         x = layers.UpSampling2D(2)(x)
#
#         # Project residual
#         residual = layers.UpSampling2D(2)(previous_block_activation)
#         residual = layers.Conv2D(filters, 1, padding="same")(residual)
#         x = layers.add([x, residual])  # Add back residual
#         previous_block_activation = x  # Set aside next residual
#     outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)
#     # Define the model
#     model = keras.Model(inputs, outputs)
#     return model


def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), \
               kernel_initializer='he_normal', padding='same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), \
               kernel_initializer='he_normal', padding='same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x

def unet(pretrained_weights = None,input_size = (842,1054,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    print('conv1',conv1.type_spec.shape)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    print('conv2',conv2.type_spec.shape)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    print('conv3',conv3.type_spec.shape)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    print('drop4',drop4.type_spec.shape)
    print('pool4',pool4.type_spec.shape)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    print('up6',up6.type_spec.shape)
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs, conv10)

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()


def get_unet(input_img, start_neurons=64, dropout=0.1, batchnorm=True):
    # Contracting Path
    inputs = keras.Input(shape=[848, 1056, 3])
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(inputs)
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(0.25)(pool1)
    print('conv1',conv1.type_spec.shape)

    conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(0.5)(pool2)
    print('conv2',conv2.type_spec.shape)

    conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(0.5)(pool3)
    print('conv3',conv3.type_spec.shape)

    conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(pool3)
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(0.5)(pool4)
    print('conv4',conv4.type_spec.shape)

    # Middle
    convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(pool4)
    convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(convm)
    print(convm.type_spec.shape)

    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    print('conv4',deconv4.type_spec.shape)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(0.5)(uconv4)
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)

    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    print('conv3',deconv3.type_spec.shape)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(0.5)(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)
    #
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2),padding="same")(uconv3)
    print('deconv2',deconv2.type_spec.shape)

    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(0.5)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)

    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    print('deconv1',deconv1.type_spec.shape)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(0.5)(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)

    output_layer = Conv2D(4, (1, 1), padding="same", activation="sigmoid")(uconv1)

    model = Model(inputs, output_layer)
    return model

if __name__ == '__main__':
    keras.backend.clear_session()
    x = np.load('/home/kuro/project/Image-Alignment/input/imagesClean3.npy',allow_pickle=True)
    y = np.load('/home/kuro/project/Image-Alignment/input/masksClean3.npy',allow_pickle=True)
    # x_train = x[:10]
    # y_train = y[:10]
    # x_val = x[10:13]
    # y_val = y[10:13]
    # x_test = x[13:]
    # y_test = y[13:]
    x_train = x[:7]
    y_train = y[:7]
    x_val = x[7:10]
    y_val = y[7:10]
    x_test = x[10:]
    y_test = y[10:]

    # cv2.imshow('img', y[14])
    # cv2.imshow('mask', x[14])
    # cv2.waitKey(0)

    model = get_unet((848,1056,3))
    model.compile(optimizer=Adam(learning_rate=.0001), loss="sparse_categorical_crossentropy", metrics=['accuracy'],)
    # model.build((842,1054,1))
    model.summary()
    earlyStopping=tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=5)
    plt.imshow(y_train[0])
    plt.show()
    history =model.fit(x= x_train,y= y_train, epochs=20, validation_data=(x_val, y_val), batch_size=1, callbacks=[earlyStopping])
    val_predics=model.predict(x_test)
    df = pd.DataFrame({'accuracy': history.history['accuracy'],
                       'val_accuracy': history.history['val_accuracy'],
                       'loss': history.history['loss'],
                       'val_loss':history.history['val_loss']})
    cv2.waitKey(0)

    df.to_csv('/home/kuro/project/Image-Alignment/output/unetc3.csv', index=False)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    # f, axarr = plt.subplots(2, 4)
    # axarr[0, 0].imshow(y_test[0])
    # axarr[0, 1].imshow(x_test[0]*255)
    # # axarr[0, 2].imshow(cv2.cvtColor(cv2.COLOR_RGBA2RGB,val_predics[0]))
    # axarr[0, 3].imshow(val_predics[0]*255)
    # axarr[1, 0].imshow(y_test[1])
    # axarr[1, 1].imshow(x_test[1]*255)
    # axarr[1, 2].imshow(cv2.cvtColor(cv2.COLOR_RGBA2RGB,val_predics[1]))
    # axarr[1, 3].imshow(val_predics[1]*255)
    # plt.show()
    util.output('/home/kuro/project/Image-Alignment/output/c3mask.png',y_test[0])
    util.output('/home/kuro/project/Image-Alignment/output/c3mask1.png',y_test[1])
    util.output('/home/kuro/project/Image-Alignment/output/c3predicted.png',val_predics[0]*64)
    util.output('/home/kuro/project/Image-Alignment/output/c3predicted1.png',val_predics[1]*64)
    util.output('/home/kuro/project/Image-Alignment/output/c3predicted3.png',val_predics[0])
    util.output('/home/kuro/project/Image-Alignment/output/c3predicted4.png',val_predics[1])
    util.output('/home/kuro/project/Image-Alignment/output/c3ori.png',x_test[0]*255)
    util.output('/home/kuro/project/Image-Alignment/output/c3ori1.png',x_test[1]*255)

