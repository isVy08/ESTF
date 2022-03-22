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

def generate_data(X, p, alpha_bound):
    T = X.size(1)
    input, target = [], []
    input_indices = []
    alphas = [alpha_bound[0]] + [(1 - (t-1)/T) * alpha_bound[0] + (t-1)/T * alpha_bound[1] for t in range(2, T+1)]
    target_indices = torch.arange(p+1, T+1)

    for i in range(p, T):        
        target.append(X[:, i])
        input.append(X[:, i-p:i])
        input_indices.append(torch.arange(i-p+1, i+1))
    
    return torch.stack(input), torch.stack(target), torch.stack(input_indices), target_indices, torch.Tensor(alphas)




class Model(nn.Module):
    def __init__(self, N, T, g, alphas):
        
        super(Model, self).__init__()

        self.g = g
        self.N = N
        self.alphas = alphas.unsqueeze(-1)

        # Defining some parameters        
        self.weights = nn.Parameter(nn.init.xavier_uniform_(torch.empty(T, N * N)))
        self.mus = nn.Parameter(nn.init.xavier_uniform_(torch.empty(T, N)))
        

    def forward(self, x, x_i, y_i):
        """
        x : [b, N, p]
        x_i : [b, p]
        y_i : [b]
        """
        self.g.requires_grad = False
        # y_i_rep = torch.repeat_interleave(y_i.unsqueeze(-1), N, dim=-1)
        mu_y = mus[y_i, :]

        # Shape function
        W = torch.matmul(self.weights ** 2, g) # [T, N ** 2]
        W = torch.mul(W, self.alphas)
        wg = W.reshape(-1, self.N, self.N) #[T, N, N]
        f = torch.softmax(-wg, -1) # [T, N, N]
        
        Z = 0
        for p in range(x.size(-1)):
            steps = x_i[:, p] - 1
            
            # coefs_ = coefs[steps, ] # [b, N]
            # x_i_rep = torch.repeat_interleave(x_i[:,p].unsqueeze(1), N, dim=1) # [b, N]
            mu_x = mus[steps, ]
            f_ = f[steps, :]

            a = (x[:, :, p] - mu_x).unsqueeze(-1)
            z = torch.matmul(f_, a)
            z.squeeze(-1) + mu_y
        
        return Z, W



def train(X, d, p, model_path, batch_size, epochs, lr, shape, device='cpu'):
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    g = basis_function(d, shape)
    g = torch.from_numpy(g).float() # [N ** 2, N ** 2]

    # F = np.log(d+1)
    # F = torch.from_numpy(F)
    
    
    N, T = X.shape   

    # Generate data 
    # input :  [T - 1, N, 1], target: [T - 1, N], input_indices: [T-1, p], target_indices: [T], alphas: [T]
    input, target, input_indices, target_indices, alphas = generate_data(X, p, [0.01, 3])
    loader = DataLoader(list(range(T-p)), batch_size=batch_size, shuffle=False)

    #  Intialize model
    model = Model(N, g, k)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    if os.path.isfile(model_path):
        load_model(model, optimizer, model_path, device)
    else:
        model.to(device)

    loss_fn = nn.MSELoss()
    prev_loss = 1.0

    for epoch in range(1, epochs + 1):
        train_losses = 0
        val_losses = 0
        for idx in tqdm(loader): 
            y = target[idx, ]
            x = input[idx, ]
            x_i, y_i = input_indices[idx, ], target_indices[idx, ]

            pred, _ = model(x, x_mask, x_i, y_i)
            # W = W.squeeze(-1)
            optimizer.zero_grad()
            loss = loss_fn(pred, y)
            loss.backward()
            
            optimizer.step()
            train_losses += loss.item()
            # val_losses += loss_fn(W, F).item()

        train_loss = train_losses / len(loader)
        val_loss = val_losses / len(loader)
        msg = f"Epoch: {epoch}, Train loss: {train_loss:.5f}, Val loss: {val_loss:.5f}"
        print(msg)
        if train_loss < prev_loss:
            print('Saving model ...')
            torch.save({'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),}, model_path)
            prev_loss = train_loss

def forecast(X, d, p, model_path, forecast_path, shape, device='cpu'):

    g = basis_function(d, shape)
    g = torch.from_numpy(g).float()
    
    N, _ = X.shape 

    input, target, input_mask, input_indices, target_indices, k = generate_data(X, p)
    
    model = Model(N, g, k)
    load_model(model, None, model_path, device)
    model.eval()

    loss_fn = nn.MSELoss()
    
    print('Predicting ...')
    input_mask = torch.eye(k)[input_mask]
    pred, W = model(input, input_mask, input_indices, target_indices)
    pred = pred.squeeze(-1)
    loss = loss_fn(pred, target)
 
    pred = torch.t(pred)
    out = torch.cat((X[:, :p], pred), dim=-1)

    out = out.detach().numpy()
    W = W.detach().numpy()

    print(loss)
    print(model.alphas ** 2)
    if forecast_path:
        X = X.detach().numpy()
        write_pickle([X, out, W], forecast_path)
    
    
def scale(X, max_, min_):
    X_std = (X - X.min(axis=1).reshape(-1,1)) / ((X.max(axis=1) - X.min(axis=1)).reshape(-1,1))
    X_std = X_std * (max_ - min_) + min_
    return X_std

if __name__ == "__main__":


    sample_path = 'data/sample.pickle'
    data_path = 'data/nst_sim_data.csv'
    model_path = 'model/nst_sim.pt'
    forecast_path = 'output/nst_sim.pickle'


    df = pd.read_csv(data_path)
    X = df.iloc[:, 1:].to_numpy()
    X = torch.from_numpy(X).float()
     _, d, _ = load_pickle(sample_path)


    train_size = 3000
    batch_size = 100
    epochs = 100
    lr = 1e-2
    
    p = 1
    
        
    X_train = X[:, :train_size]
    

    shape = 'monotone_inc'

    if sys.argv[1] == 'train':
        train(X, d, p, model_path, batch_size, epochs, lr, shape, device='cpu')
    else:
        forecast(X, d, p, model_path, forecast_path, shape)





