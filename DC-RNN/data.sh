#!/bin/bash
python -m scripts.gen_adj_mx  --sensor_ids_filename=../data/st_sim/graph_location_ids.txt --distances_filename=../data/st_sim/distances.csv --normalized_k=0.1 --output_pkl_filename=data/sim/adj_mx.pkl 
