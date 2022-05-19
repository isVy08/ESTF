python generateSE.py ./data/air/Adj.txt ./data/air/SE.txt

python main.py --num_his=1 --num_pred=1 --train_size=200 --max_epoch=100 --traffic_file=../data/air/data.h5 --SE_file=data/air/SE.txt --model_file=model/air.pt --log_file=logs/air_train --learning_rate=1e-2 --output_file=data/air/preds.npz

