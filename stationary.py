import os, sys
from utils import *
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

def generate_data(X, p):
    T = X.size(1)
    input = []
    target = []
    for i in range(p, T):
        target.append(X[:, i])
        input.append(X[:, i-p:i])
    return torch.stack(input), torch.stack(target)


class Model(nn.Module):
    def __init__(self, input_size, g):
        super(Model, self).__init__()

        self.g = g
        self.N = input_size

        # Defining some parameters
        w = torch.empty(input_size * input_size, 1)
        self.weights = nn.Parameter(nn.init.xavier_normal_(w, gain=0.1))
    def forward(self, x):
        self.g.requires_grad = False
        w = torch.matmul(self.g, self.weights ** 2) 
        
        f = w.reshape(self.N, self.N) 
        f = torch.softmax(f, -1) # remove minus sign if decreasing
        z = f @ x
        z = z.sum(-1)
        return z, w




def train(X, d, p, batch_size, epochs, lr, model_path, shape, device='cpu'):
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    g = basis_function(d, shape)
    g = torch.from_numpy(g).float()
    
    #  Intialize model
    N, T = X.shape   
    
    model = Model(N, g)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    if os.path.isfile(model_path):
        load_model(model, optimizer, model_path, device)
    else:
        model.to(device)
    
    model.to(device)

    loss_fn = nn.MSELoss()
    prev_loss = 1e+10


    # Generate data 
    input, target = generate_data(X, p)
    loader = DataLoader(list(range(T-p)), batch_size=batch_size, shuffle=False)

    for epoch in range(1, epochs + 1):
        train_losses = 0
        for idx in tqdm(loader): 
            y = target[idx,]
            dx = input[idx, ]
            pred, F_hat = model(dx)
            pred = pred.squeeze(-1)
            optimizer.zero_grad()
            loss = loss_fn(pred, y)
            loss.backward()
                        
            optimizer.step()
            train_losses += loss.item()
            

        train_loss = train_losses / len(loader)
        msg = f"Epoch: {epoch}, Train loss: {train_loss:.5f}"
        print(msg)
        if train_loss < prev_loss:
            print('Saving model ...')
            torch.save({'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),}, model_path)
            prev_loss = train_loss
        elif train_loss > prev_loss:
            break


def forecast(X, d, p, model_path, forecast_path, shape, device='cpu'):

    g = basis_function(d, shape)
    g = torch.from_numpy(g).float()
    
    #  Intialize model
    N, T = X.shape 
    
    model = Model(N, g)
    load_model(model, None, model_path, device)

    Xts = torch.from_numpy(X).float()
    input, target = generate_data(Xts, p)
    loss_fn = nn.MSELoss()
    
    print('Predicting ...')
    pred, F = model(input)
    pred = pred.squeeze(-1)
    loss = loss_fn(pred, target)
 
    pred = torch.t(pred)
    out = torch.cat((Xts[:, :p], pred), dim=-1)
    print(loss.item())
    
    out = out.detach().numpy()
    
    F = F.detach().numpy()
    write_pickle([X, out, F], forecast_path) 
    
    


if __name__ == "__main__":

    sample_path = 'data/sample.pickle'
    data_path = sys.argv[1]
    forecast_path = sys.argv[2]
    model_path = sys.argv[3]
    
    df = pd.read_csv(data_path)
    X = df.iloc[:, 1:].to_numpy()

    train_size = 300
    batch_size = 50
    epochs = 300
    lr = 0.001
    
    p = 1

    _, d = load_pickle(sample_path)

            
    X_train = X[:, :train_size]  
    X_train = torch.from_numpy(X_train).float()

    shape = 'convex_dec'

    train(X_train, d, p, batch_size, epochs, lr, model_path, shape, device='cpu')
    forecast(X, d, p, model_path, forecast_path, shape)





