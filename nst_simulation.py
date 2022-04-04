import os, sys
from utils import *
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader


def generate_data(X, p):
    
    input, target = [], []
    input_indices = []
    
    T = X.size(1)
    target_indices = torch.arange(p, T) 

    for i in range(p, T):        
        target.append(X[:, i])
        input.append(X[:, i-p:i])
        input_indices.append(torch.arange(i-p, i))
    
    return torch.stack(input), torch.stack(target), torch.stack(input_indices), target_indices

def generate_alphas(lower, upper, AT=4000):
    alphas = [[lower]] + [ [ (1 - (t-1)/AT) * lower + (t-1)/AT * upper ]  for t in range(2, AT+1)]
    return torch.Tensor(alphas)

class Model(nn.Module):
    def __init__(self, N, T, g):
        
        super(Model, self).__init__()

        self.g = g
        self.N = N
        self.T = T

        # Defining some parameters        
        self.weights = nn.Parameter(nn.init.uniform_(torch.empty(N * N, 1)))

        self.alphas = nn.Parameter(nn.init.normal_(torch.empty(T, 1)))

    def forward(self, x, x_i):
        """
        x : [b, N, p]
        x_i : [b, p]
        """
        self.g.requires_grad = False
    
        # Shape function
        F = torch.matmul(self.g, self.weights ** 2) # [N ** 2, 1]
        W = torch.matmul(self.alphas ** 2, -F.t()) # [T, N ** 2]
        
        wg = W.reshape(-1, self.N, self.N) #[T, N, N]
        f = torch.softmax(wg, -1) # [T, N, N]
        
        Z = 0
        for p in range(x.size(-1)):
            steps = x_i[:, p]
        
            f_ = f[steps, :]

            a = (x[:, :, p]).unsqueeze(-1)
            z = torch.matmul(f_, a)
            Z += z.squeeze(-1)
        
        return Z, F



def train(X, d, p, model_path, batch_size, epochs, lr, shape, device='cpu'):
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    g = basis_function(d, shape)
    g = torch.from_numpy(g).float() # [N ** 2, N ** 2]

    F = np.log(d+1)
    F = torch.from_numpy(F)
    
    
    N, T = X.shape   

    # Generate data 
    # input :  [T - 1, N, 1], target: [T - 1, N], input_indices: [T-1, p], target_indices: [T], alphas: [T]
    input, target, input_indices, target_indices = generate_data(X, p)
    loader = DataLoader(list(range(T-p)), batch_size=batch_size, shuffle=False)

    #  Intialize model
    model = Model(N, T, g)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

    if os.path.isfile(model_path):
        load_model(model, optimizer, model_path, device)
    else:
        model.to(device)

    loss_fn = nn.MSELoss()
    prev_loss = 1e+10

    for epoch in range(1, epochs + 1):
        train_losses = 0
        val_losses = 0
        for idx in tqdm(loader): 
            y = target[idx, ]
            x = input[idx, ]
            x_i = input_indices[idx, ]

            pred, F_hat = model(x, x_i)
            optimizer.zero_grad()
            loss = loss_fn(pred, y)
            loss.backward()
            
            optimizer.step()
            train_losses += loss.item()
            val_losses += loss_fn(F_hat[:, 0], F).item()

        train_loss = train_losses / len(loader)
        # val_loss = val_losses / len(loader)
        val_loss = train_losses 
        msg = f"Epoch: {epoch}, Train loss: {train_loss:.5f}, Val loss: {val_loss:.5f}"
        print(msg)
        if val_loss < prev_loss:
            print('Saving model ...')
            torch.save({'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),}, model_path)
            prev_loss = val_loss
        elif val_loss > prev_loss: 
            print(f'Early stopping at epoch {epoch}')
            break


def update(X_new, p, epochs, model, optimizer, loss_fn):
    for _ in tqdm(range(epochs)):
        x, y, x_i, _ = generate_data(X_new, p)
        y_hat, _ = model(x, x_i)
        optimizer.zero_grad()
        loss = loss_fn(y_hat, y)
        loss.backward(retain_graph=True)        
        optimizer.step()
    return model, optimizer

def forecast(X, d, p, train_size, lr, until, epochs, 
            model_path, forecast_path, 
            shape, device):

    g = basis_function(d, shape)
    g = torch.from_numpy(g).float()
    
    N, T = X.shape[0], train_size 

    input, target, input_indices, target_indices = generate_data(X, p)
    
    model = Model(N, T, g)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    load_model(model, optimizer, model_path, device)
    loss_fn = nn.MSELoss()

    
    # Dynamic forecasting
    preds, F = model(input[:T, ], input_indices[:T, ]) 
    complete = False

    while not complete:
        with torch.no_grad():            
            print(f'Forecasting the next {train_size} steps ...')
            for t in range(train_size):
                i = t + T
                x = X.t()[i - p: i, :].reshape(1, N, -1)
                x_i = torch.LongTensor([[t]])
                y_hat, _ = model(x, x_i)
                preds = torch.cat((preds, y_hat))

                L = preds.size(0)
                remaining = max(0, until + train_size - L) 
                print(f'{remaining} steps until completion')
        
                if L >= until + train_size:
                    complete = True
                    break
        T = L
        if not complete:
            # Update model
            X_new = preds[-train_size:, ].t()
            X_new.requires_grad = True
            print('Updating model ...')
            model, optimizer = update(X_new, p, epochs, model, optimizer, loss_fn)
    
    out = preds.t()
    X = X[:, :T]
    
    loss = loss_fn(out, X)
    print(loss.item())
 
    out = out.detach().numpy()  
    # X = X.detach().numpy()
    # F = F.detach().numpy()
    # write_pickle([X, out, F], forecast_path) 
    np.save(forecast_path, out)    

if __name__ == "__main__":


    sample_path = 'data/sample.pickle'

    data_path = sys.argv[1]
    forecast_path = sys.argv[2]
    model_path = sys.argv[3]

    # data_path = 'data/nst_sim/csv/s0.csv'
    # model_path = 'model/nst_sim/model0.pt'


    df = pd.read_csv(data_path)
    X = df.iloc[:, 1:].to_numpy()

    X = torch.from_numpy(X).float()
    _, d, _ = load_pickle(sample_path)
    

    train_size = 300
    batch_size = 300
    epochs = 1000
    lr = 0.01
    p = 1

    
    shape = 'monotone_inc'

    X_train = X[:, :train_size]


        
    train(X_train, d, p, model_path, batch_size, epochs, lr, shape, device='cpu')
    until = 200
    forecast(X, d, p, train_size, lr, until, 100, model_path, forecast_path, shape, device='cpu')




