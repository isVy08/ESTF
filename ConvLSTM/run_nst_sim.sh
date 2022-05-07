# # name horizon batch-size epoch
for i in {0..30}
do
    
    python generate_training_data.py --output_dir=data/nst_sim/ --traffic_df_filename=../data/nst_sim/h5/s$i.h5
    python train_test.py data/nst_sim/ 1 50 100 data/nst_sim/output/preds$i.npz
    rm model/nst_sim.h5
    rm data/nst_sim/*.npz
done
