for i in {0..99}
do
    python -m scripts.generate_training_data --output_dir=data/sim/ --traffic_df_filename=../data/st_sim/h5/s$i.h5
    python train_test.py --config_filename=data/sim/stvar.yaml --use_cpu_only=True --output_filename=data/sim/output/preds$i.npz
    rm models/sim/*.tar
    rm data/sim/*.npz
done