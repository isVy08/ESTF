#!/bin/bash
python -m scripts.generate_training_data --output_dir=data/$1 --traffic_df_filename=data/$1/stvar.h5
python -m scripts.gen_adj_mx  --sensor_ids_filename=data/$1/graph_location_ids.txt --distances_filename=data/$1/distances.csv --normalized_k=0.1 --output_pkl_filename=data/$1/adj_mx.pkl 
