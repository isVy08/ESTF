cp ../DC-RNN/data/$1/adj_mx.pkl ./data/$1/
cp ../DC-RNN/data/$1/stvar.h5 ./data/$1/
cp ../DC-RNN/data/$1/distances.csv ./data/$1/
cp ../DC-RNN/data/$1/graph_location_ids.txt ./data/$1/
# dataset-name horizon
python -m generate_training_data --output_dir=data/$1 --traffic_df_filename=data/$1/stvar.h5 --horizon=$2 --history_length=$2
