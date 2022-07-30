# python -m scripts.gen_adj_mx  --sensor_ids_filename=../data/so2/graph_location_ids.txt --distances_filename=../data/so2/distances.csv --normalized_k=0.1 --output_pkl_filename=data/so2/adj_mx.pkl 

python -m scripts.generate_training_data --output_dir=data/so2/ --traffic_df_filename=../data/so2/data.h5 # specify num train
python train_test.py --config_filename=data/so2/stvar.yaml --use_cpu_only=True --output_filename=data/so2/preds.npz

