python -m scripts.gen_adj_mx  --sensor_ids_filename=../data/air/graph_location_ids.txt --distances_filename=../data/air/distances.csv --normalized_k=0.1 --output_pkl_filename=data/air/adj_mx.pkl 

python -m scripts.generate_training_data --output_dir=data/air/ --traffic_df_filename=../data/air/data.h5 # specify num train
python train_test.py --config_filename=data/air/stvar.yaml --use_cpu_only=True --output_filename=data/air/preds.npz

