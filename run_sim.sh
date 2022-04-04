for i in {0..99}
do
    python st_simulation.py data/st_sim/csv/s$i.csv output/st_sim/preds$i.npy
done
