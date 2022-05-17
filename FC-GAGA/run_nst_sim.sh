cp ../DC-RNN/data/nst_sim/adj_mx.pkl ./data/nst_sim/
for i in {0..99}
do
    python -m generate_training_data --output_dir=data/nst_sim/ --traffic_df_filename=../data/non_stationary/h5/s$i.h5 --horizon=1 --history_length=1
    python train_test.py train nst_sim
    # val nst_sim (model index) (output file)
    python train_test.py val nst_sim 0 data/nst_sim/output/preds$i.npz
    rm model/nst_sim-0.hdf5
    rm data/nst_sim/*.npz
done