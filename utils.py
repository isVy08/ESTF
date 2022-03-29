import json, pickle

class Namespace:
  def __init__(self, **kwargs):
    self.__dict__.update(kwargs)

def get_config(config_file): 
  with open(config_file) as f:
        config = json.load(f)
  n = Namespace()
  n.__dict__.update(config)
  return n

def standardize(ts, dim, means=None, stds=None):
  if means is None:
    means = ts.mean(dim=dim, keepdim=True)
  if stds is None:
    stds = ts.std(dim=dim, keepdim=True)
  return (ts - means) / stds

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
        if shape == 'monotone_inc':
            a = (d >= sorted_d[i]).astype('float')
            b = int(sorted_d[i] <= 0.0) 
            g.append(a - b)
        elif shape == 'concave_inc':
            a = (d <= sorted_d[i]).astype('float')
            gx = np.multiply(d-sorted_d[i], a) + sorted_d[i] * int(sorted_d[i] >= 0.0) 
            g.append(gx)

    return np.stack(g, axis=1)

def scale(X, max_, min_):
    X_std = (X - X.min(axis=1).reshape(-1,1)) / ((X.max(axis=1) - X.min(axis=1)).reshape(-1,1))
    X_std = X_std * (max_ - min_) + min_
    return X_std

def normalize(X):
  X_std = (X - X.mean(1).reshape(-1, 1)) / X.std(1).reshape(-1, 1)
  return X_std