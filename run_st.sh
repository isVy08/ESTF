for i in {0..30}
do
    # python stationary.py data/stationary/csv/s$i.csv output/stationary/out$i.pickle model/stationary/model$i.pt data/stationary/csv/F.npy $i 
    python stationary.py data/stationary/csv/s$i.csv output/test/out$i.pickle model/test/model$i.pt data/stationary/csv/F.npy $i 
done
