from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import yaml

from lib.utils import load_graph_data
from model.pytorch.dcrnn_supervisor import DCRNNSupervisor
import numpy as np


def main(args):
    with open(args.config_filename) as f:
        supervisor_config = yaml.safe_load(f)

        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)

        supervisor = DCRNNSupervisor(adj_mx=adj_mx, **supervisor_config)

        supervisor.train()
        end_training = time.time()
        mean_score, outputs = supervisor.evaluate(args.split)
        np.savez_compressed(args.output_filename, **outputs)
        print("MAE : {}".format(mean_score))
        print('Predictions saved as {}.'.format(args.output_filename))
    return end_training


if __name__ == '__main__':
    import os, psutil, time
    process = psutil.Process(os.getpid())
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default=None, type=str,
                        help='Configuration filename for restoring the model.')
    parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')
    parser.add_argument('--split', default='full', type=str, help='Dataset to evaluate on')
    parser.add_argument('--output_filename', default='data/full_predictions.npz')
    args = parser.parse_args()
    end_training = main(args)
    print(process.memory_info().vms)  # in bytes 
    end = time.time()
    print(f'Computing time: {end - start} seconds\nInference time: {end - end_training}')