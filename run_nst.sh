for i in {0..99}
do
    python non_stationary.py data/nst_sim/csv/s$i.csv output/nst_sim/out$i.pickle model/nst_sim/model$i.pt data/nst_sim/csv/F.npy $i
done
