import pandas as pd
import numpy as np

df = pd.read_csv('data/nst_sim_data_clean.csv')
data = df.iloc[:, 1:].to_numpy()
print(data.shape)
np.save('data/nst_sim_clean.npy', data)
