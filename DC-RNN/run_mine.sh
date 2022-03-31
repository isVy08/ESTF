# python -m scripts.gen_adj_mx  --sensor_ids_filename=../data/mine/graph_location_ids.txt --distances_filename=../data/mine/distances.csv --normalized_k=0.1 --output_pkl_filename=data/mine/adj_mx.pkl 

python -m scripts.generate_training_data --output_dir=data/mine/ --traffic_df_filename=../data/mine/data.h5
python train_test.py --config_filename=data/mine/stvar.yaml --use_cpu_only=True --output_filename=data/mine/preds.npz

