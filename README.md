# Twitter Sentiment Analysis
Team name: AntipathyGlance

Team members: Rolando, Klim, Mengtao, Fabian

## How to use

1) Clone this repo
2) Download the twitter Datasets and extract everything in this folder.
3) Install the python dependencies, You can install them with 
    > pip install -r pip-freeze.txt
>
4) Run the HF_dataset.py
5) Now you can run HF_transformer.py for the best models.
6) You can also run the file baselines.py for the baselines. (you will need to create an empty 'images' folder)

(Quick Note: When running locally or on the cluster the environment and the paths are usually different. For this reason there exists an --on_cluster parameter that allows to switch between environments. You will need to set up some environment variables on the cluster:
> CIL_PROJECT_PATH

> CIL_EXPERIMENTS_PATH → it is recommended to put this one on the scratch partition.

)
### Branches
There are 3 main branches for the final project:
- master: This is where most of the code is, in its final state. It can run all the HuggingFace models using the HF_dataset.py file. The required parameters must be provided for it to run.

- CA-KL: You will need to switch to this branch, check that you still have the correct HF_data.txt file. Then you will be able to run this model with
> python SCAL_transformer.py --cfg configs/default_gpu.yaml 

- bertTweetCustom: You will need to switch to this branch, check that you still have the correct HF_data.txt file. You will be able to run this model in the same way as master. For example:
>  python HF_transformer.py --tag "Test Transformer" --on_cluster --train --test --amount_per_it 50000 --amount_of_data 2000000 --full_data --bs_train 32 --bs_eval 16 --n_epochs 3

### Best Model command
To reproduce the best run we had, you will need to run this:
> python HF_transformer.py --tag "Test BERTTweet" --on_cluster --train --test --n_epochs 3 --lr 1e-5 --fp16 --model_name vinai/bertweet-base --bs_train 32 --bs_eval 32 --tokenizer_max_length 128

It would be advised to run it on the cluster as it can take quite a lot of time... If you run it on the cluster you should set the correct environment variables, if not, please remove the "--on_cluster" parameter.

### HF_transformer.py Usage
>>
>usage: HF_transformer.py [-h] [--tag TAG] [--seed SEED] [--on_cluster] [--autosubmit] [--discord] [--load_model LOAD_MODEL] [--test] [--train]
>                         [--num_labels NUM_LABELS] [--full_data] [--amount_of_data AMOUNT_OF_DATA] [--amount_per_it AMOUNT_PER_IT] [--start_at_it START_AT_IT]
>                         [--use_HF_dataset_format] [--freq_words] [--tokenizer_max_length TOKENIZER_MAX_LENGTH] [--model_name MODEL_NAME]
>                         [--train_val_ratio TRAIN_VAL_RATIO] [--lr LR] [--n_epochs N_EPOCHS] [--weight_decay WEIGHT_DECAY] [--bs_train BS_TRAIN]
>                         [--bs_eval BS_EVAL] [--fp16]
>
>options:
>  -h, --help            show this help message and exit
>  --tag TAG             A custom tag for this experiment
>  --seed SEED           Random number generator seed
>  --on_cluster
>  --autosubmit
>  --discord
>  --load_model LOAD_MODEL
>                        Checkpoint ID of a pretrained model in the experiment folder
>  --test                Load a model and generate output file
>  --train               Train a model
>  --num_labels NUM_LABELS
>                        How many different classes there are to predict
>  --full_data           Use the full dataset
>  --amount_of_data AMOUNT_OF_DATA
>                        Amount of Data
>  --amount_per_it AMOUNT_PER_IT
>                        Amount of data to load on each iteration
>  --start_at_it START_AT_IT
>                        Start at a certain iteration (usefull for resuming a training)
>  --use_HF_dataset_format
>                        Use the cleaned HF dataset we created.
>  --freq_words          Use the most frequent words.
>  --tokenizer_max_length TOKENIZER_MAX_LENGTH
>                        Set the tokenizer max_length parameter.
>  --model_name MODEL_NAME
>                        Default model name to load
>  --train_val_ratio TRAIN_VAL_RATIO
>                        The training/validation ratio to use for the given dataset
>  --lr LR               Learning rate
>  --n_epochs N_EPOCHS   Number of train epochs
>  --weight_decay WEIGHT_DECAY
>                        Weight decay
>  --bs_train BS_TRAIN   Batch size for the training set
>  --bs_eval BS_EVAL     Batch size for validation/test set
>  --fp16                Uses fp16 for training (Not always supported)
>


### Twitter Datasets

Download the tweet datasets from here:
http://www.da.inf.ethz.ch/teaching/2018/CIL/material/exercise/twitter-datasets.zip


The dataset should have the following files:
- sample_submission.csv
- train_neg.txt : a subset of negative training samples
- train_pos.txt: a subset of positive training samples
- test_data.txt:
- train_neg_full.txt: the full negative training samples
- train_pos_full.txt: the full positive training samples



