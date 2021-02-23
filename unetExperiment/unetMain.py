import numpy as np
import tensorflow as tf
import pandas as pd
import util
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Conv2DTranspose
from tensorflow import keras
import matplotlib.pyplot as plt

def get_unet(input_img, num_class, start_neurons=64, dropout=0.1 ):
    inputs = keras.Input(shape=input_img)
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(inputs)
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(dropout)(pool1)

    conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(dropout)(pool2)

    conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(dropout)(pool3)

    conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(pool3)
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(dropout)(pool4)

    # Middle
    convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(pool4)
    convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(convm)

    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(dropout)(uconv4)
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)

    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(dropout)(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)

    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)

    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(dropout)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)

    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(dropout)(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)

    output_layer = Conv2D(num_class, (1, 1), padding="same", activation="sigmoid")(uconv1)

    model = Model(inputs, output_layer)
    return model


def train(x_train, y_train, x_val, y_val, image_shape, classes, epoch, batchSize, modelName):
    model = get_unet(image_shape, num_class=classes)
    model.compile(optimizer=Adam(learning_rate=.0001), loss="sparse_categorical_crossentropy", metrics=['accuracy'], )

    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)

    history = model.fit(x=x_train, y=y_train, epochs=epoch, validation_data=(x_val, y_val), batch_size=batchSize,
                        callbacks=[earlyStopping])
    model.save(pathModel+modelName+'.h5')

    df = pd.DataFrame({'accuracy': history.history['accuracy'],
                       'val_accuracy': history.history['val_accuracy'],
                       'loss': history.history['loss'],
                       'val_loss': history.history['val_loss']})
    df.to_csv(pathOutput+'unetc3.csv', index=False)


def predict(path_model, x_test, y_test):
    model = keras.models.load_model(path_model+'.h5')
    val_predicts = model.predict(x_test)
    util.output(pathOutput + 'c3mask.png', y_test[0])
    util.output(pathOutput + 'c3mask1.png', y_test[1])
    util.output(pathOutput + 'c3predicted.png', val_predicts[0] * 64)
    util.output(pathOutput + 'c3predicted1.png', val_predicts[1] * 64)
    util.output(pathOutput + 'c3predicted3.png', val_predicts[0])
    util.output(pathOutput + 'c3predicted4.png', val_predicts[1])
    util.output(pathOutput + 'c3ori.png', x_test[0] * 255)
    util.output(pathOutput + 'c3ori1.png', x_test[1] * 255)
    return val_predicts

def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255

if __name__ == '__main__':
    keras.backend.clear_session()
    pathInput = '/home/kuro/project/Image-Alignment/input/'
    pathOutput = '/home/kuro/project/Image-Alignment/output/'
    pathModel = '/home/kuro/project/Image-Alignment/model/'

    imageShape = [848, 1056, 3]
    numClasses = 4
    epoch = 20
    batchSize = 1

    x = np.load('/home/kuro/project/Image-Alignment/input/imagesClean3.npy', allow_pickle=True)
    y = np.load('/home/kuro/project/Image-Alignment/input/masksClean3.npy', allow_pickle=True)
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

    # train(x_train, y_train, x_val, y_val, imageShape, numClasses, epoch, batchSize, 'unetd3')
    val_predicts = predict(pathModel + 'unetd3', x_test, y_test)
    result = []
    for img in val_predicts:
        imgOut = np.argmax(img, axis= -1)
        # print(img[1,1,:],img[1,1,1],img[1,1,2],img[1,1,3])
        # imgOut = img[:,:,1]+img[:,:,2]+img[:,:,3]
        result.append(imgOut)

