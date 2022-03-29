import os, torch, sys
from utils import *
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

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
    def forward(self, x):
        self.g.requires_grad = False
        w = torch.matmul(self.g, self.weights ** 2) 
        # w = torch.matmul(g, weights ** 2) 
        
        f = w.reshape(self.N, self.N) # add exponential term
        ef = torch.softmax(-f, -1)
        z = ef @ x
        z = z.sum(-1)
        return z, w



def train(X, d, p, batch_size, epochs, lr, shape, device='cpu'):
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    g = basis_function(d, shape)
    g = torch.from_numpy(g).float()
    
    #  Intialize model
    N, T = X.shape   
    
    model = Model(N, g)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    model.to(device)

    loss_fn = nn.MSELoss()

    # Generate data 
    input, target = generate_data(X, p)
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

    return model

def forecast(X, d, p, model, forecast_path, shape, device='cpu'):

    g = basis_function(d, shape)
    g = torch.from_numpy(g).float()
    
    #  Intialize model
    N, _ = X.shape 
    
    model.eval()

    Xts = torch.from_numpy(X).float()
    input, target = generate_data(Xts, p)
    loss_fn = nn.MSELoss()
    
    print('Predicting ...')
    pred, F = model(input)
    pred = pred.squeeze(-1)
    loss = loss_fn(pred, target)
 
    pred = torch.t(pred)
    out = torch.cat((Xts[:, :p], pred), dim=-1)

    out = out.detach().numpy()
    print(loss)
    np.save(forecast_path, out)    
    


if __name__ == "__main__":

    sample_path = 'data/sample.pickle'
    data_path = sys.argv[1]
    forecast_path = sys.argv[2]

    df = pd.read_csv(data_path)
    X = df.iloc[:, 1:].to_numpy()

    train_size = 300
    batch_size = 50
    epochs = 100
    lr = 0.01
    
    p = 1

    _, d, _ = load_pickle(sample_path)

            
    X_train = X[:, :train_size]  
    X_train = torch.from_numpy(X_train).float()

    shape = 'monotone_inc'

    
    model = train(X_train, d, p, batch_size, epochs, lr, shape, device='cpu')
    forecast(X, d, p, model, forecast_path, shape)





