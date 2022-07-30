cp ../DC-RNN/data/so2/adj_mx.pkl ./data/so2/

python -m generate_training_data --output_dir=data/so2/ --traffic_df_filename=../data/so2/data.h5 --horizon=1 --history_length=1 # specify train size
python train_test.py train so2
# val sim (model index) (output file)
python train_test.py val so2 0 data/so2/preds.npz
