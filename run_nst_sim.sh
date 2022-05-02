shape="log"
for i in {0..5}
do
    # python nst_simulation.py data/nst_sim/csv/s$i.csv output/test_sim/out$i.pickle model/test_sim/model$i.pt $i
    python nst_simulation.py data/non_stationary/$shape/s$i.csv output/non_stationary/$shape/out$i.pickle model/non_stationary/$shape/model$i.pt data/non_stationary/$shape/F.npy $i
done
