# # name horizon batch-size epoch learning-rate
  
python generate_training_data.py --output_dir=data/air/ --traffic_df_filename=../data/air/data.h5 # with horizon specified
python train_test.py data/air/ 1 50 100 0.01 data/air/preds.npz

