import torch
import numpy as np
from utils import *
from model import RNN
from tqdm import tqdm
from data_generator import *
from main import load_model
from estimate import estimate, calDiff


def forecast(dg, model, config, until=2000):

    sp, d, y = load_pickle(config.sample_path)
    if config.weight_method == 'static':
        F = d
    else:
        F = np.load(config.estimate_path) 
        F = F[:, :config.train_size]
    
    label = np.load(config.data_path)
    label = label[:, :config.train_size]
    X_actual = torch.from_numpy(label[:, config.window_size*2:].transpose())
    X = model(dg.Z)[0][:,-1,:]
    print(torch.nn.MSELoss()(X, X_actual))
    
    X = X.detach().numpy().transpose()
    
    W = generate_weight_batch(X, F, config.weight_method)  
    agg_W = dg.transform(W)
    std_W =  moving_average_standardize(agg_W, config.window_size)
    
    for _ in tqdm(range(until)): 
        
        # generate input for the next T+1
        z, agg_W, std_W = generate_input(W, agg_W, std_W, config.window_size, config.slide_size)

        # predict X_(T+1)
        x = model(z)[0][:, -1, :]
        x = x.detach().numpy().transpose()
        X = np.concatenate((X, x), axis=1)
    
        if config.weight_method != 'static':
            # calculate difference in obs to estimate f_(T+1)
            diff = calDiff(x)
            diff = abs(diff) # make it non-negative

            # estimation f_(T+1)
            f = estimate(d, diff, config.constraint)
            w = generate_weight(x, f)
        else: 
            w = generate_weight(x, d)

        
        # update weight matrix, size (T+1, n)
        W = torch.cat((W, torch.t(w)), dim=0)
    
    return X

if __name__ == "__main__":

    # load config file
    config=get_config('config/config.json')
    params = ['static', 'invsqr', 'exp', 'original']
    model_names = ['model/sti.pt', 'model/dyi.pt', 'model/dye.pt', 'model/dyo.pt']
    forecast_paths = ['output/sti.npy', 'output/dyi.npy', 'output/dye.npy', 'output/dyo.npy']
    n = len(params)
    for i in range(n):

        # update config
        config.model_path = model_names[i]
        config.forecast_path = forecast_paths[i]
        print(config.model_path)
        config.weight_method = params[i]
        
        
        # load data
        dg = DataGenerator(config)

        # load model
        N = config.input_size
        model = RNN(N, N, config.hidden_size, config.n_layers)
        load_model(model, None, config.model_path, config.device)
        model.eval()
        
        out = forecast(dg, model, config, 600)
        np.save(config.forecast_path, out)







