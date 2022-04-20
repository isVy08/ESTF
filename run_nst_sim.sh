for i in {0..3}
do
    # python nst_simulation.py data/nst_sim/csv/s$i.csv output/nst_sim/out$i.pickle model/nst_sim/model$i.pt
    # python nst.py data/quad_sim/csv/s$i.csv output/quad_sim/out$i.pickle model/quad_sim/model$i.pt $i
    python nst.py data/quad_sim/csv_2/s$i.csv output/rand_sim/out$i.pickle model/rand_sim/model$i.pt $i
done
