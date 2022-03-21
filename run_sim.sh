for i in {0..99}
do
    python simulation.py data/st_sim/csv/s$i.csv output/st_sim/preds$i.npy
done
