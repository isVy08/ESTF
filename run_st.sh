for i in {21..99}
do
    python stationary.py data/stationary/csv/s$i.csv output/stationary/out$i.pickle model/stationary/model$i.pt 
done
