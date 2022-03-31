# cp ../DC-RNN/data/mine/adj_mx.pkl ./data/mine/

python -m generate_training_data --output_dir=data/mine/ --traffic_df_filename=../data/mine/data.h5 --horizon=5 --history_length=5
python train_test.py train mine
# val sim (model index) (output file)
python train_test.py val mine 0 data/mine/preds.npz
