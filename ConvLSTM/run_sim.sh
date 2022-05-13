# # name horizon batch-size epoch lr
for i in {0..30}
do
    
    # python generate_training_data.py --output_dir=data/sim/ --traffic_df_filename=../data/st_sim/h5/s$i.h5
     python generate_training_data.py --output_dir=data/sim/ --traffic_df_filename=../data/stationary/h5/s$i.h5
    python train_test.py data/sim/ 1 50 100 0.01 data/sim/output/preds$i.npz
    rm model/sim.h5
    rm data/sim/*.npz
done
