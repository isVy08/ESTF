for i in {0..30}
do
    python non_stationary.py data/non_stationary/csv/s$i.csv output/non_stationary/out$i.pickle model/non_stationary/model$i.pt data/non_stationary/csv/F.npy $i
done
