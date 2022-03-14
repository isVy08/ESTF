# dataset_name train_size epoch (sim = 10, mine = 1000)
python main.py --mode=train --num_his=5 --num_pred=5 --train_size=$2 --max_epoch=$3 --traffic_file=data/$1/stvar.h5 --SE_file=data/$1/SE.txt --model_file=model/$1.pt --log_file=logs/$1_train
python main.py --mode=val --num_his=5 --num_pred=5 --train_size=$2 --max_epoch=$3 --traffic_file=data/$1/stvar.h5 --SE_file=data/$1/SE.txt --model_file=model/$1.pt --log_file=logs/$1_val