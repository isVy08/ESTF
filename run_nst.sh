for i in {4..30}
do
    # python non_stationary.py data/nst_sim/csv/s$i.csv output/nst_sim/out$i.pickle model/nst_sim/model$i.pt data/nst_sim/csv/F.npy $i
    python non_stationary.py data/non_stationary/csv/s$i.csv output/nst_sim/out$i.pickle model/nst_sim/model$i.pt data/nst_sim/csv/F.npy $i
done
