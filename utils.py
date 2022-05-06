import pickle
import numpy as np
import torch

def load(datadir):
    with open(datadir, encoding='utf-8') as f:
        data = f.read().splitlines()
    return data

def load_pickle(datadir):
  file = open(datadir, 'rb')
  data = pickle.load(file)
  return data

def write_pickle(data, savedir):
  file = open(savedir, 'wb')
  pickle.dump(data, file)
  file.close()

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
        if shape == 'monotone_inc': #2
            a = (d >= sorted_d[i]).astype('float')
            b = int(sorted_d[i] <= 0.0) 
            g.append(a - b)
        elif shape == 'concave_inc': #7
            a = (d <= sorted_d[i]).astype('float')
            gx = np.multiply(d-sorted_d[i], a) + sorted_d[i] * int(sorted_d[i] >= 0.0) 
            g.append(gx)
        elif shape == 'monotone_dec': #3
            a = (d <= sorted_d[i]).astype('float')
            b = 0 # int(sorted_d[i] > 0.0) 
            g.append(a - b)
        elif shape == 'convex_dec': #6
            a = (d <= sorted_d[i]).astype('float')
            gx = np.multiply(sorted_d[i]-d, a) - sorted_d[i] * 0 # int(sorted_d[i] >= 0.0) 
            g.append(gx)
        else:
          raise ValueError("Unknown shape!")

    return np.stack(g, axis=1)

def scale(X, max_=1, min_=0):
  """
  X shape : [N, T]
  """
  from sklearn.preprocessing import MinMaxScaler
  scaler = MinMaxScaler(feature_range=(min_, max_))
  X = scaler.fit_transform(X.transpose()).transpose()
  return X

def normalize(X):
  """
  X shape : [N, T]
  """
  from sklearn.preprocessing import StandardScaler
  scaler = StandardScaler()
  X = scaler.fit_transform(X.transpose()).transpose()
  return X