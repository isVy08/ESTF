# python generateSE.py ./data/nst_sim/Adj.txt ./data/nst_sim/SE.txt
for i in {0..99}
do
    python main.py --num_his=1 --num_pred=1 --train_size=300 --max_epoch=1000 --traffic_file=../data/nst_sim/h5/s$i.h5 --SE_file=data/nst_sim/SE.txt --model_file=model/nst_sim.pt --log_file=logs/nst_sim_train --learning_rate=1e-2 --output_file=data/nst_sim/output/preds$i.npz
    rm model/nst_sim.pt
done