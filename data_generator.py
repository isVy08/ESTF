import sys
import numpy as np
import pandas as pd
from utils import *
"""
Module to transform data to be compatible with baselines
"""

# Load location indices
ids, d = load_pickle('data/sample.pickle')
# data_dir = './data/st_sim/'
data_dir = sys.argv[1]
N = len(ids)

if '1' in sys.argv[2]:
    # Write .h5 file
    for i in range(100):
        print(f'Writing .h5 file at {i}...')
        df = pd.read_csv(data_dir + f'csv/s{i}.csv')
        df = df.iloc[:, 1:].transpose()
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
        df.to_hdf(data_dir + f'h5/s{i}.h5', key='df')


if '2' in sys.argv[2]:
    print('Writing file graph_location_ids.txt')

    # write location ids
    file = open(data_dir + 'graph_location_ids.txt', 'w+')
    for l in range(N):
        if l < N - 1:
            file.write(str(ids[l]) + ',')
        else:
            file.write(str(ids[l]))
    file.close()

if '3' in sys.argv[2]:

    # write distance df
    print('Writing file distances.csv')
    dis_df = {"from": [], "to": [], "cost": []}
    cnt = 0
    for i in range(N):
        for j in range(N):
            dis_df["from"].append(ids[i])
            dis_df["to"].append(ids[j])
            dis_df["cost"].append(round(d[cnt],3))
            cnt += 1
    pd.DataFrame.from_dict(dis_df).to_csv(data_dir + 'distances.csv', index=False)


