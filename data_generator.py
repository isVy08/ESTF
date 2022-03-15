import scipy.io
import numpy as np
import random, os
from pygam import GAM, s
from utils import *


def calDiff(df):
    diff = []
    n = df.shape[0]
    for i in range(n):
        diff.append(df - df[i,])  
    return np.concatenate(diff)

# Generate sample
def sampling(data, sample_size, to_file, seed=18, location=None):
    
    n = data.shape[0] 
    random.seed(seed)   
    sp = random.sample(range(n), sample_size)

    if location is not None and location not in sp:
        sp[-1] = location
        
    # Calculate difference 
    diff = calDiff(data[sp, :])
    d = np.sqrt(diff[:,0]**2 + diff[:,1]**2)
    y = diff[:, 2:] 
    if to_file:
        write_pickle((sp, d, y), to_file)
    return sp, d, y

def estimate(d, y, cons='convex', dist='normal', link='identity'): 
    gam = GAM(s(0, constraints=cons), distribution=dist, link=link)
    gam.fit(d, y)
    return gam.predict(d)

if __name__ == "__main__":

    input_size = 30
    sample_path = "data/sample.pickle"
    data_path = "data/data.npy"
    source_path = "data/mine_data.mat"

    
    if os.path.isfile(sample_path):
        sp, d, y = load_pickle(sample_path)
        y = abs(y) 
    
    else: 
        mat = scipy.io.loadmat(source_path)
        data = mat['data']

        sp, d, y = sampling(data, input_size, sample_path, 8, None)
        y = abs(y)
        X = data[sp, 2:]
        np.save(data_path, X)

 