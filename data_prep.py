import os

from utils import *
import pandas as pd

ids, d, _ = load_pickle('data/sample.pickle')

if os.path.isfile('data/mine/data.npy'):
    import numpy as np
    data = np.load('data/mine/data.npy')
else:
    import scipy.io as sio
    matdata = sio.loadmat('data/mine_data.mat')
    data = matdata['data']

    # Sample 30 locations given by ids
    data = data[ids, 2:]

# Normalize data
data = normalize(data)

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
df.to_hdf('data/mine/data.h5', key='df')





