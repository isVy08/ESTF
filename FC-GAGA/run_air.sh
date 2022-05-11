# cp ../DC-RNN/data/air/adj_mx.pkl ./data/air/

python -m generate_training_data --output_dir=data/air/ --traffic_df_filename=../data/air/data.h5 --horizon=1 --history_length=1
python train_test.py train air
# val sim (model index) (output file)
python train_test.py val air 0 data/air/preds.npz
