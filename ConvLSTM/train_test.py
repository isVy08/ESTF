#!/usr/bin/env python3
import os, sys
import numpy as np
from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from tensorflow.keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint

def main():
    dir = sys.argv[1]
    horizon = int(sys.argv[2])
    batch_size = int(sys.argv[3])
    
    epochs = int(sys.argv[4])
    row, col = 10, 3

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
    name = dir.split('/')[1]
    model_path = f'./model/{name}.h5'


    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    seq.fit(x['train'], y['train'], batch_size=batch_size, epochs=epochs, 
                validation_data=(x['val'], y['val']),
                callbacks = [checkpoint])
    predictions = seq.predict(x['full'], verbose=1, batch_size=batch_size)
    np.savez_compressed(
    os.path.join(sys.argv[5]),
    input=x["full"].squeeze(-1),
    truth=y["full"].squeeze(-1),
    prediction=predictions.squeeze(-1)
    
    )
    

if __name__ == '__main__':
    main()
