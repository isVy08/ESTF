for i in {0..19}
do
    # rm model/stationary/model$i.pt
    python st_simulation.py data/stationary/s$i.csv output/stationary/out$i.pickle model/stationary/model$i.pt $i
done
