import os, torch
from utils import *
import numpy as np
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

def basis_function(d):
    m = d.shape[0]
    sorted_d = np.sort(d)
    g = []
    for i in range(m):
        a = d > sorted_d[i]
        g.append(a.astype('float'))

    return np.stack(g, axis=1)

def generate_data(X, p):
    T = X.size(1)
    input = []
    target = []
    indices = []
    for i in range(p, T):
        target.append(X[:, i])
        input.append(X[:, i-p:i])
        indices.append(torch.arange(i-p, i))
    return torch.stack(input), torch.stack(target), torch.stack(indices)


class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()

        # Defining some parameters
        w = torch.empty(input_size, output_size)
        self.weights = nn.Parameter(nn.init.xavier_uniform_(w))
    def forward(self, g):
        x = torch.matmul(g, self.weights ** 2)
        return x

def batch_predict(input, idx, indices, F, N, train=True):
    pred = []
    if not train:
        idx = tqdm(idx)
    for b in idx:
    # select component time steps
        t = indices[b, :]
        # compute weight for each step
        w = torch.t(F[:, t]).reshape(-1, N, N)
        # normalize W
        w = torch.softmax(-w, dim=1) 
        # select batch input 
        dx = torch.t(input[b, :, :]).unsqueeze(-1)
        # compute sum(WX)
        Z = w @ dx 
        Z = moving_average_standardize(Z, p)
        pred.append(Z.sum(0).reshape(1,-1))
    pred = torch.cat(pred, axis=0)
    return pred



def train(DX, d, p, model_path, batch_size, epochs, lr, device='cpu'):
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    g = basis_function(d)
    g = torch.from_numpy(g).float()
    
    #  Intialize model
    N, T = DX.shape   
    
    model = Model(N * N, T)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    if os.path.isfile(model_path):
        load_model(model, optimizer, model_path, device)
    else:
        model.to(device)

    loss_fn = nn.MSELoss()

    # Generate data 
    input, target, indices = generate_data(DX, p)
    train_loader = DataLoader(list(range(T-p)), batch_size=batch_size)

    for epoch in range(1, epochs + 1):
        losses = 0
        for idx in tqdm(train_loader): 
            F = model(g)
            pred = batch_predict(input, idx, indices, F, N)
            y = target[idx,]
            optimizer.zero_grad()
            loss = loss_fn(pred, y)
            loss.backward()
            
            optimizer.step()
            losses += loss.item()

        train_loss = losses / len(train_loader)
        msg = f"Epoch: {epoch}, Train loss: {train_loss:.3f}"
        print(msg)

        torch.save({'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),}, model_path)

def forecast(X, DX, d, p, model_path, forecast_path, until=1000, device='cpu'):

    g = basis_function(d)
    g = torch.from_numpy(g).float()
    
    #  Intialize model
    N, T = DX.shape 
    
    model = Model(N * N, T)
    load_model(model, None, model_path, device)
    model.eval()

    input, target, indices = generate_data(DX, p)
    ds = input.size(0)
    loss_fn = nn.MSELoss()
    F = model(g)
    
    print('Predicting ...')
    pred = batch_predict(input, range(ds), indices, F, N, False)
    loss = loss_fn(pred, target)

    # Obtain the last 5 weights
    print('Forecasting ...')
    W = torch.t(F[:, -p:]).reshape(-1, N, N)
    W = torch.softmax(-W, dim=1) 
    # Forecast the next [until] steps
    for _ in tqdm(range(until)):
        dx = pred[-p:, :].unsqueeze(-1)
        Z = W @ dx 
        Z = moving_average_standardize(Z, p)
        new_pred = Z.sum(0).reshape(1,-1)
        pred = torch.cat((pred, new_pred), dim=0)
        
    
    pred = pred.detach().numpy()
    upper = ds + until + p
    
    fc = X[:, p:upper] + pred.transpose()
    prev = X[:, :p+1]

    out = np.concatenate((prev, fc), axis=1)

    print(loss)
    if forecast_path:
        write_pickle([X[:, :(upper + 1)], out, F], forecast_path)
    
    

if __name__ == "__main__":

    sample_path = 'sample_small.pickle'
    data_path = 'data/data_small.npy'
    model_path = 'model/sm.pt'
    forecast_path = 'output/sm.pickle'

    train_size = 4000
    batch_size = 100
    epochs = 300
    lr = 0.01
    until = 1000

    
    _, d, _ = load_pickle(sample_path)
    X = np.load(data_path)
    p = 5
    X_train = X[:, :train_size]
    DX = X_train[:, 1:] - X_train[:, :-1] 
    DX = torch.from_numpy(DX).float()

   

    # train(DX, d, p, model_path, batch_size, epochs, lr, device='cpu')

    
    forecast(X, DX, d, p, model_path, None, until)





