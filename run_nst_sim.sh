for i in {6..99}
do
    # python nst_simulation.py data/nst_sim/csv/s$i.csv output/nst_sim/out$i.pickle model/nst_sim/model$i.pt
    python nst.py data/quad_sim/csv/s$i.csv output/quad_sim/out$i.pickle model/quad_sim/model$i.pt $i
done
