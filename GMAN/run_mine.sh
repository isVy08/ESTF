# python generateSE.py ./data/mine/Adj.txt ./data/mine/SE.txt

python main.py --num_his=5 --num_pred=5 --train_size=3000 --max_epoch=100 --traffic_file=../data/mine/data.h5 --SE_file=data/mine/SE.txt --model_file=model/mine.pt --log_file=logs/mine_train --learning_rate=1e-2 --output_file=data/mine/preds.npz

