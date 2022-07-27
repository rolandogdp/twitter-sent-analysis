import os
from os.path import exists

import datasets
from datasets.tasks import TextClassification
from datasets import load_dataset

import numpy as np 
import json

from transformers import AutoTokenizer


logger = datasets.logging.get_logger(__name__)

## Constants
USE_FULL_DATASET = True
PROJECT_PATH = "./"



def _define_columns(example):
    text_splited = example["text"].split('\t')
    return {"text": text_splited[1].strip(), "labels": int(text_splited[0])}

class Sentiment(datasets.GeneratorBasedBuilder):
    def _info(self):
        class_names = ["negative", "positive"]
        return datasets.DatasetInfo(
            description="Our nice dataset in HF format",
            features=datasets.Features(
                {"text": datasets.Value("string"), 
                "labels": datasets.ClassLabel(num_classes=2, names=class_names)}  #  Value("int32")
            ),
            supervised_keys=("text", "labels"),
        )

    def _split_generators(self, _):
        """Returns SplitGenerators."""

        data_dir = "./"

        data = load_dataset("text", data_files="./HF_data.txt")
        data = data.map(_define_columns)

        texts_dataset_clean = data["train"].train_test_split(train_size=0.95, seed=12345)
        # Rename the default "test" split to "validation"
        texts_dataset_clean["validation"] = texts_dataset_clean.pop("test")


        for split, dataset in texts_dataset_clean.items():
            dataset.to_json(data_dir + f"twitter-sentiment-analysis-{split}.jsonl")


        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": os.path.join(data_dir, "twitter-sentiment-analysis-train.jsonl")}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": os.path.join(data_dir, "twitter-sentiment-analysis-validation.jsonl")}),
        ]   

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            for key, row in enumerate(f):
                data = json.loads(row)
                yield key, {
                    "text": data["text"],
                    "labels": data["labels"],
                }


def read_file(file_name_label_tuple):
    fname, label = file_name_label_tuple
    tweets, labels = [], []
    with open(fname, 'r', encoding='utf-8') as f:
        tweets = f.readlines()
    
    labels = [label] * (len(tweets))

    return(tweets, labels)


def load_train_data():

    if USE_FULL_DATASET == True:
        X_train_neg_path = PROJECT_PATH + "train_neg_full.txt"
        X_train_pos_path = PROJECT_PATH + "train_pos_full.txt"
        
    else:
        X_train_neg_path = PROJECT_PATH + "train_neg.txt"
        X_train_pos_path = PROJECT_PATH + "train_pos.txt"
    
    tweets, labels = read_file((X_train_neg_path, 0))
    tweets = list(set(tweets))
    labels = labels[:len(tweets)]
    print("There are ", len(tweets), " negative tweets after removing the duplicates.")
    
    tweets_2, labels_2 = read_file((X_train_pos_path, 1))
    tweets_2 = list(set(tweets_2))
    labels_2 = labels_2[:len(tweets_2)]
    print("There are ", len(tweets_2), " positive tweets after removing the duplicates.")

    
    tweets += tweets_2
    tweets_2 = []
    del(tweets_2)
    labels += labels_2
    labels_2 = []
    del(labels_2)
    print(f"Loaded {len(tweets)} tweets!")

    tweets, labels = np.array(tweets), np.array(labels)
    print(tweets)

    # To shuffle the data before cerating the .txt file dataset
    nb_of_samples = len(tweets)
    shuffled_indices = np.random.permutation(nb_of_samples)
    train_indices = shuffled_indices[:nb_of_samples]
    labels = labels[train_indices]

    print("Number of indices for training: ", len(train_indices))

    return tweets, labels


def create_data_file(tweets, labels):
    
    with open("HF_data.txt", "wb") as f:
        for i in range(len(tweets)):
            # print(tweets[i])
            f.write(f"{labels[i]} \t {tweets[i]}".encode('utf-8'))

    
    

def main():
    tweets, labels = load_train_data()
    create_data_file(tweets, labels)

if __name__ == "__main__":
    main()