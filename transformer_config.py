## Code inspired by the 2022 Machine Perception course project skeleton file at ETH Zürich.
# Configuration for the HF_transformer.py file

import argparse
import os
import pprint
import torch

class Constants(object):
    class __Constants:
        def __init__(self):
            # Environment setup
            self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.discord_hook = "https://discord.com/api/webhooks/990270535592181800/g_zbw08Fz-52WZAbeb1Sy5au_ND2h1TmSmn1Cs_BIrHj7ne3Mb8rcnbl3EcrOY-hd_sn"
            self.bot_name = "Klim's bot"
            
    instance = None

    def __new__(cls, *args, **kwargs):
        if not Constants.instance:
            Constants.instance = Constants.__Constants()
        return Constants.instance

    def __getattr__(self, item):
        return getattr(self.instance, item)

    def __setattr__(self, key, value):
        return setattr(self.instance, key, value)


CONSTANTS = Constants()


class Configuration(object):
    "Config parameters to pass via the command line."

    def __init__(self, adict):
        self.__dict__.update(adict)

    def __str__(self):
        return pprint.pformat(vars(self), indent=4)

    @staticmethod
    def parse_cmd():
        parser = argparse.ArgumentParser()

        # General
        parser.add_argument('--tag', default='', help='A custom tag for this experiment')
        parser.add_argument('--seed', type=int, default=12345, help='Random number generator seed')     # for randomness; initially None
        # parser.add_argument('--data_workers', type=int, default=4, help='Number of parallel threads for data loading.')
        # parser.add_argument('--print_every', type=int, default=200, help='Print stats to console every so many iters.')
        # parser.add_argument('--eval_every', type=int, default=400, help='Evaluate validation set every so many iters.')
        parser.add_argument("--on_cluster", default=False, action="store_true")

        # Kaggle
        parser.add_argument("--autosubmit", default=False, action="store_true")

        # Discord notifications bot
        parser.add_argument("--discord", default=False, action="store_true")

        # Run
        parser.add_argument('--load_model', default=None, help='Checkpoint ID of a pretrained model in the experiment folder')
        parser.add_argument("--test", default=False, action="store_true", help='Load a model and generate output file')
        parser.add_argument("--train", default=False, action="store_true", help='Train a model')

        # Data
        parser.add_argument('--num_labels', type=int, default=2, help='How many different classes there are to predict')
        parser.add_argument("--full_data", default=False, action="store_true", help='Use the full dataset')
        parser.add_argument('--amount_of_data', type=int, default=0, help='Amount of Data')
        parser.add_argument('--amount_per_it', type=int, default=10000, help='Amount of data to load on each iteration')
        parser.add_argument('--start_at_it', type=int, default=0, help='Start at a certain iteration (usefull for resuming a training)')
        # parser.add_argument('--aug_data', default=False, action="store_true", help='If using the augmented data engineering or not.')
        parser.add_argument("--use_HF_dataset_format", default=False, action="store_true", help='Use the cleaned HF dataset we created.')
        parser.add_argument("--freq_words", default=False, action="store_true", help='Use the most frequent words.')
        parser.add_argument('--tokenizer_max_length', type=int, default=256, help='Set the tokenizer max_length parameter.')

        # Model
        parser.add_argument('--model_name', type=str, default="bert-base-cased", help='Default model name to load')     # "cardiffnlp/twitter-roberta-base-sentiment-latest"
        
        # Learning args
        parser.add_argument('--train_val_ratio', type=float, default=0.95, help='The training/validation ratio to use for the given dataset')
        parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
        parser.add_argument('--n_epochs', type=int, default=2, help='Number of train epochs')
        parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
        parser.add_argument('--bs_train', type=int, default=16, help='Batch size for the training set')
        parser.add_argument('--bs_eval', type=int, default=16, help='Batch size for validation/test set')
        parser.add_argument("--fp16", default=False, action="store_true", help='Uses fp16 for training (Not always supported)')
        
        config = parser.parse_args()
        
        return Configuration(vars(config))