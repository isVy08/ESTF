#!/usr/bin/env python3


import numpy as np
from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from tensorflow.keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint

def main(row, col, horizon=20, batch_size=60, epochs=50):
    dir = './data/'

    x = {}
    y = {}
    for split in ('train', 'val', 'test', 'full'):
        data = np.load(dir + f'{split}.npz')
        x[split] = data['x'].reshape(-1, horizon, row, col, 1)
        y[split] = data['y'].reshape(-1, horizon, row, col, 1)


    seq = Sequential()
    seq.add(ConvLSTM2D(filters=col, kernel_size=(3, 3),
                       input_shape=(None, row, col, 1),
                       padding='same', return_sequences=True))
    seq.add(BatchNormalization())

    seq.add(ConvLSTM2D(filters=col, kernel_size=(3, 3),
                       padding='same', return_sequences=True))
    seq.add(BatchNormalization())

    seq.add(ConvLSTM2D(filters=col, kernel_size=(3, 3),
                       padding='same', return_sequences=True))
    seq.add(BatchNormalization())

    seq.add(ConvLSTM2D(filters=col, kernel_size=(3, 3),
                       padding='same', return_sequences=True))
    seq.add(BatchNormalization())

    seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
                   activation='sigmoid',
                   padding='same', data_format='channels_last'))

    seq.compile(loss='mean_squared_error', optimizer='adadelta')

    # Train the network

    model_path = 'ConvLSTM.h5'

    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    seq.fit(x['train'], y['train'], batch_size=batch_size, epochs=epochs, 
                validation_data=(x['val'], y['val']),
                callbacks = [checkpoint])
                


if __name__ == '__main__':
    main()
