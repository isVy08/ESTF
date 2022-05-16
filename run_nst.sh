for i in {0..99}
do
    python non_stationary.py data/non_stationary/csv/s$i.csv output/non_stationary/out$i.pickle model/non_stationary/model$i.pt 100
done
