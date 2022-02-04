import torch, os
import numpy as np
from utils import *
from rnn import RNN
from pygam import GAM, s
from tqdm import tqdm
from trainer import train_epoch
from main import load_model
from torch.utils.data import DataLoader
from estimate import calDiff, estimate


def generate_weight(x, f):
    N = x.shape[0]
    fd = f.reshape((N, N))
    fd = np.exp(-fd)
    w = fd/np.sum(fd, axis=0)        
    return torch.from_numpy(w @ x).float() 

def generate_weight_batch(X, F):
    W = []
    T = X.shape[1]
    
    for t in range(T):
        z = generate_weight(X[:,t], F[:,t])
        W.append(z)

    return torch.stack(W)    

def moving_average_standardize(W, n):
    T = W.shape[0]
    std_W = (W[:n, :] - W[:n, :].mean())/W[:n, :].std() 
    for i in range(n,T):
        ref = W[i+1-n:i+1, :]
        w = (W[i:i+1, :] - ref.mean())/ref.std()
        std_W = torch.cat((std_W,w))
    return std_W

def generate_input(W, agg_W, std_W, p, s):
    """
    Generate input for the next observation
    W is raw weights
    """ 
    # Aggregated weight for the next observation
    w = W[-p:,].sum(dim=0).unsqueeze(0)
    agg_W = torch.cat((agg_W, w))

    # Standardize w by p moving average on agg_W
    std_w = (w - agg_W[-s:,].mean())/agg_W[-s:,].std()
    std_W = torch.cat((std_W, std_w))
    return std_W[-p:,].unsqueeze(0), agg_W, std_W

def generate_diff_input(X, order=1):
    if order == 1:
        return X[:, 1:] - X[:, :-1] 

class DataGenerator():
    def __init__(self, X_slice, F_slice, window_size, slide_size):
        super(DataGenerator).__init__()
        """
        F : f(D_ij) ( N * N x T)
        X : ( N x T)
        """

        X_slice = generate_diff_input(X_slice)
        # F_slice = F_slice[:, 1:] ???
        self.N, self.T = X_slice.shape
        self.p = window_size
        self.s = slide_size
            
    
        self.W = generate_weight_batch(X_slice, F_slice)   
        self.agg_W = self.transform(self.W)
        
        self.std_W = moving_average_standardize(self.agg_W, self.s)
        self.Z = self.generate_input(self.std_W)
        
        input_X = torch.from_numpy(X_slice.transpose()).float()
        self.X = self.generate_label(input_X)
        
    def transform(self, W):
        T, N = W.shape
        A = W[:1, :]
        A = torch.zeros((1, N))
        for i in range(1, T):
            a = W[:i,:] if i < self.p else W[(i-self.p):i,:]
            a = a.sum(dim=0).unsqueeze(dim=0)
            A = torch.cat((A, a), dim=0)
        return A
    
    def generate_input(self, W):

        Z = W[self.p:(self.p*2), :].unsqueeze(dim=0)
        for i in range(self.p*2+1, self.T):
            z = W[(i-self.p):i, :].unsqueeze(dim=0)
            Z = torch.cat((Z, z), dim=0)
        return Z
    
    def generate_label(self, input_X):
        L = []
        T = input_X.shape[0]
        for i in range(self.p*2, T):
            l = input_X[(i-self.p+1):(i+1), ]
            L.append(l)
        return torch.stack(L)

def train(X_slice, F_slice, window_size, slide_size, device, batch_size, 
                input_size, hidden_size, n_layers, 
                model_path, lr, epochs):

    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # load data
    dg = DataGenerator(X_slice, F_slice, window_size, slide_size)
    indices = list(range(dg.Z.size(0)))
    train_loader = DataLoader(indices, batch_size=batch_size)

    
    # load model, optimizer, loss function
    model = RNN(input_size, input_size, hidden_size, n_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if os.path.isfile(model_path):
        load_model(model, optimizer, model_path, device)
    else:
        model.to(device)

    
    # training
    print("Training begins")
    loss_fn = torch.nn.MSELoss()
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, optimizer, train_loader, dg.Z, dg.X, loss_fn)        
        msg = f"Epoch: {epoch}, Train loss: {train_loss:.3f}"
        print(msg)

        if epoch % 20 == 0:
            torch.save({'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),}, model_path)

    torch.save({'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),}, model_path)
    return dg, model

def forecast(d, dg, model, constraint, until):
    
    W, agg_W, std_W = dg.W, dg.agg_W, dg.std_W
    results = []
    for _ in tqdm(range(until)): 
        
        # generate input for the next T+1
        z, agg_W, std_W = generate_input(W, agg_W, std_W, window_size, slide_size)

        # predict X_(T+1)
        x = model(z)[0][:, -1, :]
        x = x.detach().numpy().transpose()
        results.append(x)
    
        # calculate difference in obs to estimate f_(T+1)
        diff = calDiff(x)
        diff = abs(diff) 

        # estimation f_(T+1)
        f = estimate(d, diff, constraint)
        
        # calculate W.X at T+1
        w = generate_weight(x, f)
        
        # update weight matrix, size (T+1, n)
        W = torch.cat((W, torch.t(w)), dim=0)
    
    return results

        
if __name__ == "__main__":
    
    # load data
    X = np.load('data/data_sim.npy')
    F = np.load('data/estimates_sim.npy')
    _, d, _ = load_pickle('sample_sim.pickle')
    
    # params
    constraint = 'concave'
    window_size = 10
    slide_size = 10
    device = 'cpu'
    batch_size = 200
    input_size, hidden_size, n_layers, lr = 100, 128, 3, 0.0001  
    model_path = "model/sim_128_3L.pt"
    forecast_path = 'output/sim_128_3L.npy'
    
    train_size = 3000
    slice = 5
    max_step = 200
    max_sample = X.shape[1] - slice

    if os.path.isfile(forecast_path):
        collector = np.load(forecast_path)
        t = collector.shape[1] - train_size + slice
        counter = t / slice
        collector = np.expand_dims(collector.T, 2)
        collector = list(collector)
    else:
        collector = []
        t, counter = 0, 0
    
    # run
    
    epochs = 30000

    while counter < max_step:
        print(f"LEARNING AT STEP {counter} for {epochs} EPOCHS")

        if counter > 0:
            epochs = 1000
            model_path = "model/sim_128_3L_fc.pt"

        X_slice = X[:, t : t + train_size]
        F_slice = F[:, t : t + train_size]
        dg, model = train(X_slice, F_slice, window_size, slide_size, device, batch_size, 
                        input_size, hidden_size, n_layers, model_path, lr, epochs)
        
        print("Forecasting begins ...")
        results = forecast(d, dg, model, constraint, slice)
        
        # Save forecasting results    
        curr = collector[-1] if len(collector) > 0 else X_slice[:, -1:]
        for pred in results:
            curr = curr + pred
            collector.append(curr)
        
        out = np.concatenate(collector, axis=1)
        np.save(forecast_path, out)

        t += slice
        counter += 1

        if t + train_size > max_sample:
            break
    
    






