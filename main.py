import os, torch, sys
from utils import *
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader


def load_model(model, optimizer, model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def moving_average_standardize(W, n):
    T = W.shape[0]
    std_W = (W[:n, :] - W[:n, :].mean())/W[:n, :].std() 
    for i in range(n,T):
        ref = W[i+1-n:i+1, :]
        w = (W[i:i+1, :] - ref.mean())/ref.std()
        std_W = torch.cat((std_W,w))
    return std_W

def basis_function(d, shape):
    m = d.shape[0]
    sorted_d = np.sort(d)
    g = []
    for i in range(m):
        if shape == 'monotone_inc':
            a = (d >= sorted_d[i]).astype('float')
            b = int(sorted_d[i] <= 0.0) 
            g.append(a - b)
        elif shape == 'concave_inc':
            a = (d <= sorted_d[i]).astype('float')
            gx = np.multiply(d-sorted_d[i], a) + sorted_d[i] * int(sorted_d[i] >= 0.0) 
            g.append(gx)

    return np.stack(g, axis=1)

def generate_data(X, p):
    T = X.size(1)
    input = []
    target = []
    # indices = []
    for i in range(p, T):
        target.append(X[:, i])
        input.append(X[:, i-p:i])
        # indices.append(torch.arange(i-p, i))
    return torch.stack(input), torch.stack(target)


class Model(nn.Module):
    def __init__(self, input_size, g):
        super(Model, self).__init__()

        self.g = g
        self.N = input_size

        # Defining some parameters
        w = torch.empty(input_size * input_size, 1)
        self.weights = nn.Parameter(nn.init.xavier_normal_(w))
    def forward(self, dx):
        self.g.requires_grad = False
        w = torch.matmul(self.g, self.weights ** 2) 
        # w = torch.matmul(g, weights ** 2) 
        
        f = w.reshape(self.N, self.N) # add exponential term
        ef = torch.softmax(-f, -1)
        z = ef @ dx
        return z, w



def train(DX, d, p, model_path, batch_size, epochs, lr, shape, device='cpu'):
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    g = basis_function(d, shape)
    g = torch.from_numpy(g).float()
    
    #  Intialize model
    N, T = DX.shape   
    
    model = Model(N, g)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    if os.path.isfile(model_path):
        load_model(model, optimizer, model_path, device)
    else:
        model.to(device)

    loss_fn = nn.MSELoss()

    # Generate data 
    input, target = generate_data(DX, p)
    loader = DataLoader(list(range(T-p)), batch_size=batch_size, shuffle=False)

    for epoch in range(1, epochs + 1):
        losses = 0
        for idx in tqdm(loader): 
            y = target[idx,]
            dx = input[idx, ]
            pred, _ = model(dx)
            pred = pred.squeeze(-1)
            optimizer.zero_grad()
            loss = loss_fn(pred, y)
            loss.backward()
            
            optimizer.step()
            losses += loss.item()

        train_loss = losses / len(loader)
        msg = f"Epoch: {epoch}, Train loss: {train_loss:.3f}"
        print(msg)

        torch.save({'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),}, model_path)

def forecast(X, d, p, model_path, forecast_path, shape, device='cpu'):

    g = basis_function(d, shape)
    g = torch.from_numpy(g).float()
    
    #  Intialize model
    N, _ = X.shape 
    
    model = Model(N, g)
    load_model(model, None, model_path, device)
    model.eval()

    Xts = torch.from_numpy(X).float()
    input, target = generate_data(Xts, p)
    loss_fn = nn.MSELoss()
    
    print('Predicting ...')
    pred, F = model(input)
    pred = pred.squeeze(-1)
    loss = loss_fn(pred, target)
 
    pred = torch.t(pred)
    out = torch.cat((Xts[:, :1], pred), dim=-1)

    out = out.detach().numpy()

    print(loss)
    if forecast_path:
        write_pickle([X, out, F], forecast_path)
    
    
def scale(X, max_, min_):
    X_std = (X - X.min(axis=1).reshape(-1,1)) / ((X.max(axis=1) - X.min(axis=1)).reshape(-1,1))
    X_std = X_std * (max_ - min_) + min_
    return X_std

if __name__ == "__main__":

    sample_path = 'data/sample.pickle'
    data_path = 'data/sim.npy'
    model_path = 'model/sim.pt'
    forecast_path = 'output/sim.pickle'


    train_size = 400
    batch_size = 300
    epochs = 5000
    lr = 0.1
    
    p = 1


    X = np.load(data_path)
    _, d, _ = load_pickle(sample_path)

    # X_std = scale(X, 0.3, 0)
    X_std = X

    # d_norm = scale(d.reshape(1,-1), 1, 0).reshape(-1)
    d_norm = d
    
        
    X_train = X_std[:, :train_size]
    DX = X_train
    # DX = X_train[:, 1:] - X_train[:, :-1] 
    DX = torch.from_numpy(DX).float()

    shape = 'monotone_inc'

    if sys.argv[1] == 'train':
        train(DX, d_norm, p, model_path, batch_size, epochs, lr, shape, device='cpu')
    else:
        forecast(X_std, d_norm, p, model_path, forecast_path, shape)





