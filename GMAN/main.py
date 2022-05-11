import argparse
import time
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from utils_ import log_string
from utils_ import count_parameters, load_data

from model_ import GMAN
from train import train
from test import test

parser = argparse.ArgumentParser()
# parser.add_argument('--mode', type=str, default='train',
#                     help='whether to train or evaluate')
parser.add_argument('--time_slot', type=int, default=5,
                    help='a time step is 5 mins')
parser.add_argument('--num_his', type=int, default=20,
                    help='history steps')
parser.add_argument('--num_pred', type=int, default=20,
                    help='prediction steps')
parser.add_argument('--L', type=int, default=1,
                    help='number of STAtt Blocks')
parser.add_argument('--K', type=int, default=8,
                    help='number of attention heads')
parser.add_argument('--d', type=int, default=8,
                    help='dims of each head attention outputs')
parser.add_argument('--train_size', type=int, default=300,
                    help='training set [default : 300]')
parser.add_argument('--val_ratio', type=float, default=0.1,
                    help='validation set [default : 0.1]')
parser.add_argument('--test_ratio', type=float, default=0.2,
                    help='testing set [default : 0.2]')
parser.add_argument('--batch_size', type=int, default=50,
                    help='batch size')
parser.add_argument('--max_epoch', type=int, default=1,
                    help='epoch to run')
parser.add_argument('--patience', type=int, default=10,
                    help='patience for early stop')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=10,
                    help='decay epoch')
parser.add_argument('--traffic_file', default='./data/stvar.h5',
                    help='traffic file')
parser.add_argument('--SE_file', default='./data/SE.txt',
                    help='spatial embedding file')
parser.add_argument('--model_file', default='./model/GMAN.pt',
                    help='save the model to disk')
parser.add_argument('--log_file', default='./data/log',
                    help='log file')
parser.add_argument('--output_file', default='./data',
                    help='output file')
args = parser.parse_args()
log = open(args.log_file, 'w')
log_string(log, str(args)[10: -1])
T = 24 * 60 // args.time_slot  # Number of time steps in one day
# load data
log_string(log, 'loading data...')
(trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY, fullX, fullY, fullTE, SE) = load_data(args)
log_string(log, f'fullX: {fullX.shape}\t\t fullY: {fullY.shape}')
log_string(log, f'trainX: {trainX.shape}\t\t trainY: {trainY.shape}')
log_string(log, f'valX:   {valX.shape}\t\tvalY:   {valY.shape}')
log_string(log, f'testX:   {testX.shape}\t\ttestY:   {testY.shape}')
log_string(log, f'trainTE:   {trainTE.shape}\t\tvalTE:   {valTE.shape}')
log_string(log, f'testTE:   {testTE.shape}\t\tfullTE:   {fullTE.shape}')
log_string(log, 'data loaded!')
del trainX, trainTE, valX, valTE, testX, testTE
# build model
log_string(log, 'compiling model...')

model = GMAN(SE, args, bn_decay=0.1)
loss_criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), args.learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer,
                                      step_size=args.decay_epoch,
                                      gamma=0.9)
parameters = count_parameters(model)
log_string(log, 'trainable parameters: {:,}'.format(parameters))

if __name__ == '__main__':
    
    loss_train, loss_val = train(model, args, log, loss_criterion, optimizer, scheduler) 
    print(loss_train, loss_val)    
    test(args, log)
