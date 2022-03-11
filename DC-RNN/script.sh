python -m scripts.generate_training_data --output_dir=data/stvar --traffic_df_filename=data/stvar/stvar.h5
python -m scripts.gen_adj_mx  --sensor_ids_filename=data/stvar/graph_location_ids.txt --distances_filename=data/stvar/distances.csv --normalized_k=0.1 --output_pkl_filename=data/stvar/adj_mx.pkl 
python dcrnn_train_pytorch.py --config_filename=data/model/stvar.yaml --use_cpu_only=True
python run_evaluation.py --split=full --use_cpu_only=True --config_filename=data/model/stvar.yaml --output_filename=data/stvar/dcrnn_full_predictions.npz

