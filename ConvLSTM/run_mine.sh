# # name horizon batch-size epoch
  
python generate_training_data.py --output_dir=data/mine/ --traffic_df_filename=../data/mine/data.h5 # with horizon specified
python train_test.py data/mine/ 1 50 1000 0.001 data/mine/preds.npz

