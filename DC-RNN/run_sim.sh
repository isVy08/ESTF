# python -m scripts.gen_adj_mx  --sensor_ids_filename=../data/st_sim/graph_location_ids.txt --distances_filename=../data/st_sim/distances.csv --normalized_k=0.1 --output_pkl_filename=data/sim/adj_mx.pkl 

for i in {0..1}
do
    python -m scripts.generate_training_data --output_dir=data/sim/ --traffic_df_filename=../data/st_sim/h5/s$i.h5
    python train_test.py --config_filename=data/sim/stvar.yaml --use_cpu_only=True --output_filename=data/sim/output/preds$i.npz
    rm models/sim/*.tar
    rm data/sim/*.npz
done