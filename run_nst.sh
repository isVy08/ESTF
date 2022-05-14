for i in {0..30}
do
    # python non_stationary.py data/non_stationary/csv/s$i.csv output/nst_sim/out$i.pickle model/nst_sim/model$i.pt data/non_stationary/csv/F.npy $i
    python non_stationary.py data/non_stationary/csv/s$i.csv output/test/out$i.pickle model/test/model$i.pt data/non_stationary/csv/F.npy $i
done
