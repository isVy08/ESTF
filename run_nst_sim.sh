for i in {1..99}
do
    python nst_simulation.py data/nst_sim/csv/s$i.csv output/nst_sim/preds$i.npy model/nst_sim/model$i.pt
done
