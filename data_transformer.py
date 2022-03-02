import sys
import numpy as np
import pandas as pd

from utils import load_pickle
"""
Module to transform data to be compatible with baselines
"""

# Load original data
ids, d, _ = load_pickle('data/sample.pickle')
dir = 'DCRNN/data/STVAR/'
data = np.load('data/data.npy')
N, T = data.shape

if sys.argv[0] in ['y', 'yes']:
    from main import scale
    data = scale(data, 0.3, 0)
    d = scale(d.reshape(1,-1), 0.3, 0).reshape(-1)

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


# write location ids
file = open(dir + 'graph_location_ids.txt', 'w+')
for l in range(N):
    if l < N - 1:
        file.write(str(sample[0][l]) + ',')
    else:
        file.write(str(sample[0][l]))
file.close()

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

# python -m scripts.generate_training_data --output_dir=data/STVAR --traffic_df_filename=data/STVAR/stvar.h5
# python -m scripts.gen_adj_mx  --sensor_ids_filename=data/STVAR/graph_location_ids.txt --distances_filename=data/STVAR/distances.csv --normalized_k=0.1 --output_pkl_filename=data/STVAR/adj_mx.pkl 
# python dcrnn_train_pytorch.py --config_filename=data/model/stvar.yaml --use_cpu_only=True
# python run_evaluation.py --split=val --use_cpu_only=True --config_filename=data/model/stvar.yaml --output_filename=data/STVAR/dcrnn_val_predictions.npz


