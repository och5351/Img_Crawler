from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def gray_scale(maleImages, femaleImages):
    gray_images_male = []
    gray_images_female = []

    for i in maleImages:
            src = cv2.imread(i, cv2.IMREAD_COLOR)
            src2 = cv2.resize(src, (100, 200), interpolation=cv2.INTER_CUBIC)
            gray_images_male.append(cv2.cvtColor(src2, cv2.COLOR_BGR2GRAY))

    for i in femaleImages:
            src = cv2.imread(i, cv2.IMREAD_COLOR)
            src2 = cv2.resize(src, (100, 200), interpolation=cv2.INTER_CUBIC)
            gray_images_female.append(cv2.cvtColor(src2, cv2.COLOR_BGR2GRAY))


    return gray_images_male, gray_images_female

if __name__ == '__main__':

    np.random.seed(777)
    tf.random.set_seed(777)
    '''
    (x,y), (x_v,y_v) = mnist.load_data()
    print(x[0].ndim)
    print(x[0].shape)
    x = x.reshape(x.shape[0], 28, 28, 1)
    print(x[0].ndim)
    print(x[0].shape)
    '''

    maleImages = glob.glob('C://Users/och5351/Desktop/github_och/Img_Crawler/male/*.png')
    images = glob.glob('C://Users/och5351/Desktop/github_och/Img_Crawler/male/*.jpg')
    maleImages = maleImages + images

    femaleImages = glob.glob('C://Users/och5351/Desktop/github_och/Img_Crawler/female/*.png')
    images = glob.glob('C://Users/och5351/Desktop/github_och/Img_Crawler/female/*.jpg')
    femaleImages = femaleImages + images

    maleImages, femaleImages = gray_scale(maleImages, femaleImages)
    maleImages, femaleImages = maleImages[:1001], femaleImages[:1001]

    x_train, y_train = np.array(maleImages[:800] + femaleImages[:800]), to_categorical(np.hstack([np.ones(800), np.zeros(800)]),2)
    x_val, y_val = np.array(maleImages[801:] + femaleImages[801:]), to_categorical(np.hstack([np.ones(200), np.zeros(200)]),2)

    x_train, x_val = x_train/255, x_val/255
    #print(x_train[0].ndim, x_train[0].shape)
    x_train = x_train.reshape(x_train.shape[0], 100, 200, 1)
    #print(x_train[0].ndim, x_train[0].shape)
    x_val = x_val.reshape(x_val.shape[0], 100, 200, 1)

    model = Sequential()
    model.add(Conv2D(30, kernel_size=(10, 10), input_shape=(100, 200, 1), activation='relu'))
    model.add(Conv2D(5, kernel_size=(10, 10), input_shape=(100, 200, 1), activation='relu'))
    model.add(MaxPool2D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(60, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    modelpath = './modelFile/{epoch:02d}-{val_loss:.4f}.hdf5'
    checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=30, batch_size=200, verbose=0, callbacks=[early_stopping_callback, checkpointer])
    print("\n Test Accuracy: %.4f" % (model.evaluate(x_val, y_val)[1]))
    y_vloss = history.history['val_loss']
    y_loss = history.history['loss']
    x_len = np.array(len(y_loss))
    plt.plot(x_len, y_vloss, marker='.', c='red', label='Testset_loss')
    plt.plot(x_len, y_loss, marker='.', c='blue', label="Trainset_loss")

    plt.legend(loc='upper right')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
