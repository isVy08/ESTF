# python -m scripts.gen_adj_mx  --sensor_ids_filename=../data/nst_sim/graph_location_ids.txt --distances_filename=../data/nst_sim/distances.csv --normalized_k=0.1 --output_pkl_filename=data/nst_sim/adj_mx.pkl 

for i in {0..99}
do
    python -m scripts.generate_training_data --output_dir=data/nst_sim/ --traffic_df_filename=../data/nst_sim/h5/s$i.h5
    python train_test.py --config_filename=data/nst_sim/stvar.yaml --use_cpu_only=True --output_filename=data/nst_sim/output/preds$i.npz
    rm models/nst_sim/*.tar
    rm data/nst_sim/*.npz
done