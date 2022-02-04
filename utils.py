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