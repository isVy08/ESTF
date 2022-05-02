shape="quad"
for i in {1..19}
do
    # rm model/stationary/model$i.pt
    python st_simulation.py data/stationary/$shape/s$i.csv output/stationary/$shape/out$i.pickle model/stationary/$shape/model$i.pt data/stationary/$shape/F.npy $i
done
