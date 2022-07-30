# # name horizon batch-size epoch learning-rate
  
python generate_training_data.py --output_dir=data/so2/ --traffic_df_filename=../data/so2/data.h5 # with horizon specified
python train_test.py data/so2/ 1 50 100 0.01 data/so2/preds.npz

