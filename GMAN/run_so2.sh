# python generateSE.py ./data/so2/Adj.txt ./data/so2/SE.txt ../data/so2/sample.pickle

python main.py --num_his=1 --num_pred=1 --train_size=200 --max_epoch=100 --traffic_file=../data/so2/data.h5 --SE_file=data/so2/SE.txt --model_file=model/so2.pt --log_file=logs/so2_train --learning_rate=1e-2 --output_file=data/so2/preds.npz

