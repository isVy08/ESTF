import os
import sys
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
    
def filter(data):
    N, T = data.shape
    selected = []
    for loc in range(N):
        if (data[loc, :] == 0).sum() < (T//2): 
            selected.append(loc)
    return selected


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

    # To get locations with certain conditions i.e., modify function filter() and run
    # ids = filter(data)
    ids = [6, 12, 18, 28, 32, 33, 38, 49, 50, 51, 55, 56, 70, 71, 76, 79, 80, 87, 89, 94, 96, 98, 99, 102, 103, 104, 110, 111, 146, 150]

    d = calDist(data, ids)
    write_pickle((ids, d), location_path)

    # Sample 30 locations given by ids
    data = data[ids, 2:367]
    np.save(f'data/{dataset}/data.npy', data)


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





