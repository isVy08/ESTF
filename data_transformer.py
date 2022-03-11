import sys
import numpy as np
import pandas as pd
from main import scale
from utils import load_pickle
"""
Module to transform data to be compatible with baselines
"""

# Load location indices
ids, d, _ = load_pickle('data/sample.pickle')

dir = sys.argv[1] # Location 

if 'sim' in dir:
    data = np.load('data/sim.npy')
elif 'mine' in dir:
    data_path = 'data/data.npy'
    data = scale(data, 10, 0)


N, T = data.shape

if '1' in sys.argv[2]:

    # Write .h5 file
    print('Writing .h5 file ...')
    df = pd.DataFrame(data.transpose())
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

    df.to_hdf(dir + 'stvar.h5', key='df')

if '2' in sys.argv[2]:

    # write location ids
    file = open(dir + 'graph_location_ids.txt', 'w+')
    for l in range(N):
        if l < N - 1:
            file.write(str(ids[l]) + ',')
        else:
            file.write(str(ids[l]))
    file.close()

if '3' in sys.argv[2]:

    # write distance df
    dis_df = {"from": [], "to": [], "cost": []}
    cnt = 0
    for i in range(N):
        for j in range(N):
            dis_df["from"].append(ids[i])
            dis_df["to"].append(ids[j])
            dis_df["cost"].append(round(d[cnt],3))
            cnt += 1

    pd.DataFrame.from_dict(dis_df).to_csv(dir + 'distances.csv', index=False)


