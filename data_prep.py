import os
from utils import *
import pandas as pd
from tqdm import tqdm


def calDist(data, ids):
    d = []
    for i in tqdm(ids):
        for j in ids:
            x = (data[i, 0] - data[j, 0])**2
            y =  (data[i, 1] - data[j, 1])**2
            d.append(np.sqrt(x+y))
    return np.array(d)
    
dataset = sys.argv[1] # mine / air

# Select locations and distance

if dataset == 'mine':
    # col 1: lat, col 2: long
    data_path = 'data/mine_data.mat'
    location_path = 'data/sample.pickle'
    name = 'data'
elif dataset == 'air':
    # col 1: index, col 2: lat, col 3: long
    data_path = 'data/air/air.mat'
    location_path = 'data/air/sample.pickle'
    name = 'realcase_air'
else: 
    raise ValueError('Unknown Dataset')


if os.path.isfile(f'data/{dataset}/data.npy'):
    data = np.load(f'data/{dataset}/data.npy')
    ids, d = load_pickle(location_path)
else:
    import scipy.io as sio
    
    matdata = sio.loadmat(data_path)
    data = matdata[name]

    if dataset == 'air':
        data = data[:, 1:]

    # Sample locations
    import random
    random.seed(18)
    N, T = data.shape
    ids = random.sample(range(N), 30)

    d = calDist(data, ids)
    write_pickle((ids, d), location_path)

    # Sample 30 locations given by ids
    data = data[ids, 2:]
    np.save(f'data/{dataset}/data.npy', data)

# Normalize data
# data = scale(data)

# Convert to h5 file
df = pd.DataFrame(data)
df = df.transpose()
T = df.shape[0]
df.columns = ids
t = pd.Timestamp(2022, 3, 1, 0, 0, 0)
index = [t]
for _ in range(1, T):
    m = t.minute + 5
    h = t.hour
    dy = t.day
    if m > 59:
        m = 0
        h = t.hour + 1
    if h > 23:
        m = 0 
        h = 0
        dy = t.day + 1

    t = pd.Timestamp(2022, 3, dy, h, m, 0)
    index.append(t) 

df.index = index
df.to_hdf(f'data/{dataset}/data.h5', key='df')





