## CIL 2022 Project: Sentiment Analysis on tweets

import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import random

from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments     # https://huggingface.co/transformers/v3.0.2/main_classes/trainer.html#transformers.TFTrainingArguments
from transformers import Trainer    # https://huggingface.co/transformers/v3.0.2/main_classes/trainer
from transformers import pipeline
from datasets import load_metric

import time
import requests

# Config file
from transformer_config import Configuration
from transformer_config import CONSTANTS as C

# Constant Values for Discord bot
discord_hook= C.discord_hook
bot_name = C.bot_name
red = 15158332
green = 3066993
orange = 15105570

# Global variables
metric = load_metric("accuracy")    # metric to use
train_dataset, val_dataset = None, None


# DATASET CLASSES
class TrainDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        item['labels'] = self.labels[idx].clone().detach()
        return item

    def __len__(self):
        return len(self.labels)     # number of items in the Dataset

class TestDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])     # number of items in the Dataset


def read_file(file_name_label_tuple, starting_line=0, end_line=0):
    fname, label = file_name_label_tuple
    tweets, labels = [], []
    with open( fname, 'r', encoding='utf-8') as f:
        tweets = [line for line in f.readlines()[starting_line:end_line]]
        labels = [label] * (end_line-starting_line)

    return(tweets, labels)

def load_train_data(amount_per_batch, iteration):
 
    if use_full_dataset == True:
        X_train_neg_path = project_path + "train_neg_full.txt"
        X_train_pos_path = project_path + "train_pos_full.txt"
        
    else:
        X_train_neg_path = project_path + "train_neg.txt"
        X_train_pos_path = project_path + "train_pos.txt"

    amount_per_batch = amount_per_batch // 2
    
    starting_line = iteration * amount_per_batch
    end_line = starting_line + amount_per_batch
    print(f"Going to read {amount_per_batch*2} lines, starting_line:{starting_line}, end_line:{end_line}")
    tweets, labels = read_file((X_train_neg_path, 0), starting_line=starting_line, end_line=end_line)
    tweets_2, labels_2 = read_file((X_train_pos_path, 1), starting_line=starting_line, end_line=end_line)
    tweets += tweets_2
    tweets_2 = []
    del(tweets_2)
    labels += labels_2
    labels_2 = []
    del(labels_2)
    print(f"Loaded {len(tweets)} tweets!")
    
    return np.array(tweets), np.array(labels)

def load_test_data():
    filename = project_path + "test_data.txt"
    tweets = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.partition(",")[2]   # this allows to get the text content only for each line of the file (see that the file starts with "n," where n is the line numbner); it parts the string into three strings as: before the arg, the arg, and after the arg
            tweets.append(line.rstrip())

    return tweets


def vectorize_data(tweets,train_indices,val_indices):
    vectorizer = CountVectorizer(max_features=5000)   # 5000

    # Important: we call fit_transform on the training set, and only transform on the validation set
    X_train = vectorizer.fit_transform(tweets[train_indices])
    X_val = vectorizer.transform(tweets[val_indices])

    return X_train, X_val

def get_train_val_datasets(tweets, labels):
    nb_of_samples = len(tweets)
    shuffled_indices = np.random.permutation(nb_of_samples)
    split_idx = int(train_val_ratio * nb_of_samples)

    train_indices = shuffled_indices[:split_idx]
    val_indices = shuffled_indices[split_idx:]

    print("Number of indices for training: ", len(train_indices))
    print("Number of indices for validation: ", len(val_indices))

    if use_most_freq_words:
        X_train, X_val = vectorize_data(tweets, train_indices, val_indices)
    else:
        X_train, X_val = tweets[train_indices], tweets[val_indices]
    
    Y_train = labels[train_indices]
    Y_val = labels[val_indices]
    
    X_train = tokenizer(X_train.tolist(), max_length=256, padding="max_length", truncation=True)
    X_val = tokenizer(X_val.tolist(), max_length=256, padding="max_length", truncation=True)

    Y_train = torch.tensor(Y_train).clone().detach()
    Y_val = torch.tensor(Y_val).clone().detach()

    train_dataset = TrainDataset(X_train, Y_train)
    val_dataset = TrainDataset(X_val, Y_val)
    
    return train_dataset, val_dataset

def get_test_dataset(tweets):
    nb_of_samples = len(tweets)
    print(f'{nb_of_samples} tweets loaded for testing.\n')
    tweets = tokenizer(tweets, max_length=256, padding="max_length", truncation=True)
    tweets = TestDataset(tweets)

    return tweets

def compute_metrics(eval_pred): 
    logits, labels = eval_pred      # here, we have to get rid of the second element (neutral class) of the logits before taking the softmax IF we want to only predict neg/pos
    predictions = np.argmax(logits, axis=-1)

    return metric.compute(predictions=predictions, references=labels)
    
def train(model, train_dataset_param, val_dataset_param):
    training_args = TrainingArguments(output_dir=checkpoints_path, 
                                    overwrite_output_dir=True,
                                    per_device_train_batch_size=bs_train, 
                                    per_device_eval_batch_size=bs_eval, 
                                    learning_rate=lr, 
                                    evaluation_strategy="steps",    # "steps"
                                    save_strategy="steps",
                                    gradient_accumulation_steps=4,
                                    # gradient_checkpointing=True,
                                    save_total_limit=2,
                                    fp16=fp16,
                                    seed=SEED,
                                    warmup_steps=500,              # number of warmup steps for learning rate scheduler
                                    weight_decay=weight_decay,       # strength of weight decay
                                    logging_dir='./logs',            # directory for storing logs
                                    logging_steps=500,
                                    load_best_model_at_end=True,
                                    num_train_epochs=n_epochs)
    
    if not (train_dataset_param is None) and not (val_dataset_param is None):
        train_dataset, val_dataset = train_dataset_param, val_dataset_param

    trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=train_dataset,
      eval_dataset=val_dataset,
      compute_metrics=compute_metrics
    )

    trainer.train()

def load_model_from_checkpoint(selected_checkpoint):
    model_path = checkpoints_path+f"checkpoint-{selected_checkpoint}"
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=config.num_labels)
    print(f"Loaded model from: {model_path}")
    
    return model

def load_model_from_path(model_path):
    print(f"Loaded model from: {model_path}")
    
    return AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=config.num_labels)

def test(model):
    test_trainer = Trainer(model)
    tweets = get_test_dataset(load_test_data())
    raw_preds, _, _ = test_trainer.predict(tweets)     # only predictions to return, no label ids, no metrics; see HF Trainer doc
    Y_test_pred = np.argmax(raw_preds, axis=1)
    
    return Y_test_pred

def generate_submission(Y_preds):
    nb_of_samples=len(Y_preds)
    results = np.zeros((nb_of_samples, 2))

    results[:,0] = np.arange(1, nb_of_samples+1).astype(np.int32)  # save the ids
    results[:,1] = [-1 if elem == 0 else 1 for elem in Y_preds]  # save the test predictions

    test_results_path = experiments_results_path + "/test_results/"
    print("\nPredictions on the test set saved in: ", test_results_path)
    os.makedirs(test_results_path, exist_ok=True)    # create the folder(s) if needed
    final_filename = f"{experiment_date_for_folder_name}-submission.csv"
    np.savetxt(test_results_path + final_filename, results, fmt="%1d", delimiter=",", header = "Id,Prediction", comments = "")
    
    return final_filename

def load_and_train(model, amount_per_batch, iteration):
    # Load TRAINING-VALIDATION DATA
    
    if model is None:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=config.num_labels)

    tweets, labels = load_train_data(amount_per_batch, iteration)
    train_dataset, val_dataset = get_train_val_datasets(tweets, labels)
    tweets, labels = [], []

    # Free some memory
    del(tweets)
    del(labels)
    
    # TRAINING
    train(model, train_dataset, val_dataset)

    return model

# DISCORD
def send_discord_notif(title, content, color, error=None):
    if not discord_enabled:
        return
    if error is None:
        msg = "Little Update on my status"
    else: 
        msg = error

    data = {
        "username": bot_name,
        "content": msg,
    }
    data["embeds"] = [{
        "title":title,
        "description":content,
        "color":color
        }]

    result = requests.post(discord_hook, json = data)
    
    try:
        result.raise_for_status()
    except requests.exceptions.HTTPError as err:
        print(err)
    else:
       print("Payload delivered successfully, code {}.".format(result.status_code))

# KAGGLE API
def submit_preds_on_Kaggle(submit_filename, msg):
    import kaggle
    api = kaggle.api
    api.get_config_value("username")
    res = api.competition_submit(submit_filename, f"Automatic File submission test: {msg}", "cil-text-classification-2022")
    print("res: ", res)
    
    return res


if __name__ == "__main__":

    # To time the duration of the experiment
    time_run = time.perf_counter()     # better to use perf_counter() than time()

    # Get the config
    config = Configuration.parse_cmd()

    # Prepare the folder where this experiment (i.e., program run) outputs and results will be saved
    experiment_id = int(time_run)
    experiment_date = time.ctime(experiment_id)
    experiment_date_name = experiment_date.replace(" ", "_").replace(":", "h")[:-8] + experiment_date[-8:-5].replace(":", "m") + "s"
    experiment_date_for_folder_name = "experiment-" + experiment_date_name

    # Prepare and set the paths
    if config.on_cluster:
        print("\nRunning on the cluster.")
        project_path = os.environ["CIL_PROJECT_PATH"]   # see cluster .bashrc file for the environment variables
        experiment_path = os.environ["CIL_EXPERIMENTS_PATH"] + "Experiments/"   # see cluster .bashrc file for the environment variables
    else:
        print("\nRunning locally.")
        project_path = "./"
        experiment_path = "./" + "Experiments/"

    experiments_results_path = experiment_path + experiment_date_for_folder_name
    os.makedirs(experiments_results_path, exist_ok=True)    # create the experiment folder(s) needed
    checkpoints_path = experiments_results_path + "/checkpoints/"
    print("The project path is: ", project_path)
    print("The experiment path is: ", experiment_path)
    print("The model checkpoints will be saved at: ", checkpoints_path, "\n")


    # Fix seeds for reproducibility
    SEED = config.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = False

    # Save in the experiment folder the command line that was used to run this program
    cmd = sys.argv[0] + ' ' + ' '.join(sys.argv[1:])
    with open(os.path.join(experiments_results_path, 'cmd.txt'), 'w') as f:  # NOT DONE: need to create a model directory for each model
        f.write(cmd)

    # Data
    use_full_dataset = config.full_data 
    model_name = config.model_name 
    use_most_freq_words = config.freq_words
    train_val_ratio = config.train_val_ratio

    # Calculate values for the number of tweets to work with during model training   
    amount_per_it = config.amount_per_it
    
    if config.amount_of_data > 200000:
        use_full_dataset = True  

    if use_full_dataset:
        total_amount_of_tweets = 2500000    # max number of tweets there are in the full dataset
    else:
        total_amount_of_tweets = 200000     # max number of tweets there are in the size-reduced dataset

    # if we don't want to use the complete dataset (whether the full one or the size-reduced one)
    if config.amount_of_data != 0:
        total_amount_of_tweets = config.amount_of_data
        if total_amount_of_tweets > 2500000:
            total_amount_of_tweets = 2500000 
    
    number_of_iterations = int(np.ceil(total_amount_of_tweets / amount_per_it))
    print(f"We will use {total_amount_of_tweets} tweets in total. {int(train_val_ratio * total_amount_of_tweets)} for training and {int(total_amount_of_tweets * (1-train_val_ratio))} for validation.")
    print(f"{amount_per_it} tweets will be used for each of the {number_of_iterations} iterations (i.e., in each subset that is split in training/validation).")

    # Model
    n_epochs = config.n_epochs
    bs_train = config.bs_train
    bs_eval = config.bs_eval
    lr = config.lr
    fp16 = config.fp16
    weight_decay = config.weight_decay
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if not (config.load_model is None):
        model = load_model_from_checkpoint(config.load_model)

    # Misc
    submit_to_kaggle = config.autosubmit
    discord_enabled =  config.discord

    # Create the model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=config.num_labels)
    model.to(C.DEVICE)  # automatic if use the Trainer()
    print("\nRunning on", C.DEVICE, "\n")

    # --- TRAINING & VALIDATION ---
    if config.train:

        print(f"Going to iterate over {number_of_iterations} subsets of {amount_per_it} samples/tweets (separated for training/validation) to see {total_amount_of_tweets} in total.")

        send_discord_notif("Starting Training", f"Going to iterate over {number_of_iterations} subsets of {amount_per_it} samples/tweets (separated for training/validation) to see {total_amount_of_tweets} in total.", orange, None)

        try:
            for iteration in range(number_of_iterations)[config.start_at_it:]:
                model = load_and_train(model, amount_per_it, iteration)
                send_discord_notif("Continuing Training", f"currently finished iteration: {iteration+1}/{number_of_iterations} ", orange, None)
                print(f"{iteration+1} out of {number_of_iterations} iteration(s) done!")
        except Exception as e:
            print("GOT ERROR:", str(e))
            send_discord_notif("ERROR WHILE TRAINING", str(e), red, f"Got the error at iteration:{iteration+1}/{number_of_iterations}")
            raise(e)
            
        send_discord_notif("Finished Training", f"Used {number_of_iterations} subsets of {amount_per_it} samples (i.e., tweets) without problem. Used {total_amount_of_tweets} in total.", green, None)
        print(f"Finished Training. Used {number_of_iterations} subsets of {amount_per_it} samples (i.e., tweets) without problem. Used {total_amount_of_tweets} tweets in total.\n")

    # --- TESTING ---
    if config.test:
        try:
            Y_test_pred = test(model)
            submit_filename = generate_submission(Y_test_pred)
        except Exception as e:
            print("GOT ERROR:",str(e))
            send_discord_notif("ERROR WHILE TESTING", str(e), red, f"Got the error while trying to predict on test set.")
            raise(e)

        send_discord_notif("Finished to predict on test set", f"Everything ran without issues!", green, None)

    
    if submit_to_kaggle:
        res = submit_preds_on_Kaggle(submit_filename, "")
        send_discord_notif("Submitted results on Kaggle!", f"{res}", green, None)

    time_run = time.perf_counter() - time_run
    print(f"The program took {str(time_run/60)[:6]} minutes to run.")

