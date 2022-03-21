# python generateSE.py ./data/sim/Adj.txt ./data/sim/SE.txt
for i in {0..99}
do
    python main.py --num_his=1 --num_pred=1 --train_size=300 --max_epoch=100 --traffic_file=../data/st_sim/h5/s$i.h5 --SE_file=data/sim/SE.txt --model_file=model/sim.pt --log_file=logs/sim_train --learning_rate=1e-2 --output_file=data/sim/output/preds$i.npz
    rm model/sim.pt
done