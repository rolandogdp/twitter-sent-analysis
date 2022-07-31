First, copy the full tweets dataset into this folder <\ br>
Then generate HF dataset by python HF_dataset.py
To run the program: python HF_transformer.py

E.g.: python HF_transformer.py --tag "Test Transformer" --on_cluster --train --test --amount_per_it 50000 --amount_of_data 2000000 --full_data --bs_train 32 --bs_eval 16 --n_epochs 3

See the config file transformer_config.py to find all usable arguments.

