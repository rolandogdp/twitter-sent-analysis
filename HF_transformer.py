## CIL 2022 Project: Sentiment Analysis on tweets
## Team: Klim, Rolando, Mengtao, Fabian

import os
import sys

import json

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
from transformers import DataCollatorForLanguageModeling

from transformers import pipeline
from datasets import load_metric
from datasets import load_dataset

import time
import requests
import wandb


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
METRIC = load_metric("accuracy")    # metric to use
val_dataset_global = None

# DATASET CLASSES
class TrainDataset(Dataset):
    '''Pytorch Dataset object used to store the data in a format
    that can be easily sent to the gpu for the models.'''
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels.long()

    def __getitem__(self, idx):
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        item['labels'] = self.labels[idx].clone().detach()
        return item

    def __len__(self):
        return len(self.labels)     # number of items in the Dataset

class TestDataset(Dataset):
    '''Pytorch Dataset object used to store the data in a format
    that can be easily sent to the gpu for the models.'''
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])     # number of items in the Dataset

# Read a file containing tweets and store them as well as their respective labels
def read_file(file_name_label_tuple, starting_line=0, end_line=0):
    '''Helper function to read the default training  files that didn't had the labels in it.
    Takes a tuple: (filename , label_of_this_file), the starting line and the ending line as parameters.
    label_of_this_file being and int to set all the tweets loaded from this file.
    Will return the requested lines in 2 numpy arrays teets, labels.'''
    fname, label = file_name_label_tuple
    tweets, labels = [], []
    with open( fname, 'r', encoding='utf-8') as f:
        tweets = [line for line in f.readlines()[starting_line:end_line]]
        labels = [label] * (end_line-starting_line)

    return(tweets, labels)

# Read a file containing tweets and store them as well as their respective labels 
def read_file_HF(file_name, starting_line=0, end_line=0):
    '''Helper function to read a file with the HF format ( label \t text).
    takes the filename, the starting line and the ending line as parameters.
    Will return the requested lines in 2 numpy arrays teets, labels.'''
    tweets, labels = [], []
    with open( file_name, 'r', encoding='utf-8') as f:
        tweets = [line.split("\t",maxsplit=1) for line in f.readlines()[starting_line:end_line]]
        tweets = np.array(tweets)
        labels = tweets[:,0].astype(np.uint8)
        tweets = tweets[:,1]
        
    return(tweets, labels)

# Store the training data, tweets and labels, in numpy arrays
def load_train_data(amount_per_batch, iteration):
    '''Handles reading the data in batches. Uses the helper function read_file_HF to do so.
    takes as input: the amount of data per subset and the actual iteration number 
    and returns two arrays: tweets, labels  with length equal or lesser than amount_per_batch.'''
    # if use_full_dataset == True:
    #     X_train_neg_path = project_path + "train_neg_full.txt"
    #     X_train_pos_path = project_path + "train_pos_full.txt"
        
    # else:
    #     X_train_neg_path = project_path + "train_neg.txt"
    #     X_train_pos_path = project_path + "train_pos.txt"

    # amount_per_batch = amount_per_batch // 2
    
    # starting_line = iteration * amount_per_batch
    # end_line = starting_line + amount_per_batch
    # print(f"Going to read {amount_per_batch*2} lines ({amount_per_batch} in each of the pos and neg datasets), starting_line:{starting_line}, end_line:{end_line}")
    # tweets, labels = read_file((X_train_neg_path, 0), starting_line=starting_line, end_line=end_line)
    # tweets_2, labels_2 = read_file((X_train_pos_path, 1), starting_line=starting_line, end_line=end_line)
    # tweets += tweets_2
    # tweets_2 = []
    # del(tweets_2)
    # labels += labels_2
    # labels_2 = []
    # del(labels_2)
    # print(f"Loaded {len(tweets)} tweets!")
    # return np.array(tweets), np.array(labels)

    starting_line = iteration * amount_per_batch
    end_line = starting_line + amount_per_batch
    print(f"Going to read {amount_per_batch*2} lines ({amount_per_batch} in each of the pos and neg datasets), starting_line:{starting_line}, end_line:{end_line}")
    tweets, labels = read_file_HF(project_path + "HF_data.txt", starting_line, end_line)
    print(f"Loaded {len(tweets)} tweets!")


    return tweets, labels


    

# Store the testing data (tweets) in a numpy array
def load_test_data():
    '''Loads the testing data and returns it in a numpy array.'''
    filename = project_path + "test_data.txt"
    tweets = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.partition(",")[2]   # this allows to get the text content only for each line of the file (see that the file starts with "n," where n is the line numbner); it parts the string into three strings as: before the arg, the arg, and after the arg
            tweets.append(line.rstrip())

    return tweets

# Vectorize the data 
def vectorize_data(tweets, train_indices, val_indices):
    '''Takes as input:
    tweets: np.array of tweets in text mode.
    train_indices: np.array of the train indices of the tweets array
     val_indices: np array of the validation indices of the tweets array.
     returns X_train,X_val in vectorized form using the CountVectorizer
    '''
    vectorizer = CountVectorizer(max_features=5000)   # 5000

    # Important: we call fit_transform on the training set, and only transform on the validation set
    X_train = vectorizer.fit_transform(tweets[train_indices])
    X_val = vectorizer.transform(tweets[val_indices])

    return X_train, X_val

def get_train_val_data(tweets, labels):
    ''' Takes as input the tweets and labels as numpy arrays.
    The tweets being still in text, it will tokenize it using the corresponding tokenizer of the model.

    If it is the first call, it set ups the validation sets, and the training set.
    On future calls, it reuses the first validation set, and uses the full tweets as training data.
    it returns train_dataset: TrainDataset,val_dataset_global: TrainDataset 
    
    '''
    global val_dataset_global
    if val_dataset_global is None:
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
        
        X_train = tokenizer(X_train.tolist(), max_length=config.tokenizer_max_length, padding="max_length", truncation=True)
        X_val = tokenizer(X_val.tolist(), max_length=config.tokenizer_max_length, padding="max_length", truncation=True)

        Y_train = torch.tensor(Y_train).clone().detach()
        Y_val = torch.tensor(Y_val).clone().detach()

        train_dataset = TrainDataset(X_train, Y_train)
        val_dataset = TrainDataset(X_val, Y_val)
        val_dataset_global = val_dataset
        
        return train_dataset, val_dataset
    else:
        nb_of_samples = len(tweets)
        shuffled_indices = np.random.permutation(nb_of_samples)
        
        train_indices = shuffled_indices
        print("Number of indices for training: ", len(train_indices))

        if use_most_freq_words:
            X_train = vectorize_data(tweets, train_indices, val_indices)
        else:
            X_train = tweets[train_indices]
        
        Y_train = labels[train_indices]
        
        X_train = tokenizer(X_train.tolist(), max_length=config.tokenizer_max_length, padding="max_length", truncation=True)

        Y_train = torch.tensor(Y_train).clone().detach()
        train_dataset = TrainDataset(X_train, Y_train)
        
        return train_dataset,val_dataset_global

def get_test_data(tweets):
    ''' Takes as input the tweets as a numpy array with The tweets still
     being still in text, it will tokenize it using the corresponding tokenizer of the model.
     The tokenizer_max_length must be set from the config either passing the corresponding parameter,
     or the default value. 
     returns the TestDataset object.
    
    '''
    nb_of_samples = len(tweets)
    print(f'{nb_of_samples} tweets loaded for testing.\n')
    tweets = tokenizer(tweets, max_length=config.tokenizer_max_length, padding="max_length", truncation=True)
    tweets = TestDataset(tweets)

    return tweets

def compute_metrics(eval_pred): 
    '''Required function to evaluate the predictions of the model on the evaluation dataset.
     '''
    logits, labels = eval_pred      # here, we have to get rid of the second element (neutral class) of the logits before taking the softmax IF we want to only predict neg/pos
    predictions = np.argmax(logits, axis=-1)

    return METRIC.compute(predictions=predictions, references=labels)


def train(model, train_dataset, val_dataset, iteration):
    '''Helper function to start the training of a model.
    takes as input:
    - model: a huggingface model
    - train_dataset: a TrainDataset object used for the training.
    - val_dataset: a TrainDataset object used for the validation
    - iteration : is used when running with subsets, to save the different models of different versions. 
    
    Creates a Trainer using the training arguments defaults, and the ones passed by commandline.
    Doesn't return anything per se, but the model taken as parameter will be trained after the training.'''

    training_args = TrainingArguments(output_dir=checkpoints_path, 
                                    overwrite_output_dir=True,
                                    per_device_train_batch_size=config.bs_train, 
                                    per_device_eval_batch_size=config.bs_eval, 
                                    learning_rate=config.lr, 
                                    evaluation_strategy="steps",            # "steps"
                                    save_strategy="steps",
                                    gradient_accumulation_steps=4,
                                    # gradient_checkpointing=True,
                                    save_total_limit=2,
                                    fp16=config.fp16,
                                    seed=config.seed,
                                    warmup_steps=500,                       # number of warmup steps for learning rate scheduler
                                    weight_decay=config.weight_decay,       # strength of weight decay
                                    logging_dir='./logs',                   # directory for storing logs
                                    logging_steps=500,
                                    load_best_model_at_end=True,
                                    num_train_epochs=config.n_epochs,
                                    report_to="wandb" # WANDB INTEGRATION
                                    )


    # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True)
        
    trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=train_dataset,
      eval_dataset=val_dataset,
    #   data_collator=data_collator,
    #   tokenizer=tokenizer,
      compute_metrics=compute_metrics
    )

    if len(os.listdir(checkpoints_path)) == 0:
        trainer.train()
    else: 
        trainer.train(checkpoints_path, resume_from_checkpoint=True)    # ??? Working like that?

    best_model_at_iteration_path = f"/best_model/iteration{iteration}"
    trainer.save_model(experiments_results_path + best_model_at_iteration_path)    # save the best model of the current iteration


def load_model_from_checkpoint(path_to_checkpoint):
    ''' Helper function, to load the model from a checkpoint.
    takes as input a path to the checkpoint (from the "experiment-[...]" )
     '''
    full_path_to_model_checkpoint = experiment_path + path_to_checkpoint
    model = AutoModelForSequenceClassification.from_pretrained(full_path_to_model_checkpoint, num_labels=config.num_labels, local_files_only=False, ignore_mismatched_sizes=True)
    print(f"Loaded model from: {full_path_to_model_checkpoint}")
    
    return model


def numpy_2d_softmax(model_preds):
    '''Converts the raw predictions from a HuggingFace model into clean logits.'''
    max = np.max(model_preds, axis=1, keepdims=True)
    e_x = np.exp(model_preds-max)
    sum = np.sum(e_x, axis=1, keepdims=True)
    out = e_x / sum 
    return out

def test(model):
    ''' Takes the trained model as input, and will get the testing data and produce the predictions.
    Also produces the logits.txt file that is used for ensembling the models.
    returns a numpy array with the predictions of the model.
     '''

    test_trainer = Trainer(model)
    tweets = get_test_data(load_test_data())
    raw_preds, _, _ = test_trainer.predict(tweets)     # only predictions to return, no label ids, no metrics; see HF Trainer doc
    Y_test_pred = np.argmax(raw_preds, axis=1)

    # store the logits in a file
    logits = numpy_2d_softmax(raw_preds)    # beer owning line
    print(len(logits))
    print(logits)

    if not(config.model_name is None):
        model_name_for_logits = config.model_name.split("/")[1]
    else: 
        model_name_for_logits = "NoModelNameGiven"
    
    if not(config.load_model is None):
        model_name_for_logits = config.load_model.split("experiment-")[1].split("\\")[0]
    
    np.savetxt(test_results_path + model_name_for_logits + "-" + 'logits.txt', logits, delimiter=",", header = "negative,positive", comments = "") # fmt="%1d"
    
    return Y_test_pred

def generate_submission(Y_preds):
    '''Takes as input a numpy array containing the model predictions, and generates 
    a correctly formatted output csv file for the kaggle competition.'''
    nb_of_samples=len(Y_preds)
    results = np.zeros((nb_of_samples, 2))

    results[:,0] = np.arange(1, nb_of_samples+1).astype(np.int32)  # save the ids
    results[:,1] = [-1 if elem == 0 else 1 for elem in Y_preds]  # save the test predictions

    final_filename = f"{experiment_date_for_folder_name}-submission.csv"
    np.savetxt(test_results_path + final_filename, results, fmt="%1d", delimiter=",", header = "Id,Prediction", comments = "")
    
    return final_filename

def load_and_train(model, amount_per_batch, iteration):
    '''Helper function for training the model using the batches strategie to allow the model the run on systems with low amount of memory.
    takes as input: 
    -model: the model to train (HF model)
    -amount_per_batch: The amount of data to be loaded and used on each iteration.
    -iteration: At which iteration the training is.

    In case the use HF_dataset format has been used, you should not run this function multiple times, only once as 
    all the data will be loaded using HuggingFace's api. There is also no way to select the amount of data to work 
    on when using this parameter.

    
    returns the trained model.'''
    
    if model is None:
        model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=config.num_labels, local_files_only=False, ignore_mismatched_sizes=True)


    if config.use_HF_dataset_format: # Using HuggingFace api to load all the data.
        train_dataset = load_dataset("./HF_dataset.py", split="train")
        val_dataset = load_dataset("./HF_dataset.py", split="validation")

        datasets = load_dataset("./HF_dataset.py")

        def tokenization(sample):
            return tokenizer(sample["text"], max_length=config.tokenizer_max_length, padding="max_length", truncation=True)

        datasets = datasets.map(tokenization, batched=True)

        train_dataset = datasets["train"]
        val_dataset = datasets["validation"]

    else:
        # Load training & validation data
        tweets, labels = load_train_data(amount_per_batch, iteration)
        train_dataset, val_dataset = get_train_val_data(tweets, labels)
        tweets, labels = [], []

        # Free some memory
        del(tweets)
        del(labels)
    
    # TRAINING
    train(model, train_dataset, val_dataset, iteration)

    return model

# DISCORD 
def send_discord_notif(title, content, color, error=None):
    '''Simple function to allow callbacks to the discord channel to get updates when running for a long time.'''
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
    '''Simple helper function to submit the predictions csv file on the test data to the kaggle competition
    takes as input the file path to the csv, and the message to put on the leaderboard.'''
    import kaggle
    api = kaggle.api
    api.get_config_value("username")
    res = api.competition_submit( experiments_results_path + "/test_results/" + submit_filename, f"Automatic File submission test: {msg}", "cil-text-classification-2022")
    print("res: ", res)
    
    return res

# TRAINING
def run_training(model):
    ''' Function to call from the main to start the training, takes as input the model to train.
    It will handle all the requirements for the training.
    It will load all the data required for training, if using subsets it will handle all the iterations over the data.
    If enabled it will send discord notification on training start, and at each subset ending.
    It will also handle the wandb initialisation of a project with name being: cil-{model_name} 
    It will also catch errors that happened during training and send them to the discord hook and proceed to exit. 
    
    returns a trained model. '''
    print(f"Going to iterate over {number_of_iterations} subsets of {amount_per_it} samples/tweets (separated for training/validation) to see {total_amount_of_tweets} in total.")
    send_discord_notif("Starting Training", f"Going to iterate over {number_of_iterations} subsets of {amount_per_it} samples/tweets (separated for training/validation) to see {total_amount_of_tweets} in total.", orange, None)
    wandb_project_name = f"cil-{model_name}".replace("/","-").replace("\\","").replace("?","").replace("%","").replace(":","")
    wandb.init(project=wandb_project_name)
    try:
        total_subsets = range(number_of_iterations)[config.start_at_it:]

        if config.use_HF_dataset_format:
            total_subsets = range(1)

        for iteration in total_subsets:
            trained_model = load_and_train(model, amount_per_it, iteration)
            # torch.cuda.empty_cache()  # can be used if save the trained model before that line and load it again after that line
            send_discord_notif("Continuing Training", f"currently finished subset iteration: {iteration+1}/{number_of_iterations} ", orange, None)
            print(f"{iteration+1} out of {number_of_iterations} subset iteration(s) done!")
    
    except Exception as e:
        print("GOT ERROR:", str(e))
        send_discord_notif("ERROR WHILE TRAINING", str(e), red, f"Got the error at subset iteration: {iteration+1}/{number_of_iterations}")
        raise(e)
        
    send_discord_notif("Finished Training", f"Used {number_of_iterations} subsets of {amount_per_it} samples (i.e., tweets) without problem. Used {total_amount_of_tweets} in total.", green, None)
    print(f"Finished Training. Used {number_of_iterations} subsets of {amount_per_it} samples (i.e., tweets) without problem. Used {total_amount_of_tweets} tweets in total.\n")

    return trained_model


# TESTING
def run_testing(model):
    '''Function to call from main to start the testing of a trained model.
    Takes as input the trained model and returns the output filename.csv'''
    try:
        Y_test_pred = test(model)
        submit_filename = generate_submission(Y_test_pred)
    except Exception as e:
        print("GOT ERROR:", str(e))
        send_discord_notif("ERROR WHILE TESTING", str(e), red, f"Got the error while trying to predict on test set.")
        raise(e)

    send_discord_notif("Finished to predict on test set", f"Everything ran without issues!", green, None)

    return submit_filename

if __name__ == "__main__":

    torch.cuda.empty_cache()    

    # To time the duration of the experiment
    time_run = time.time()     # better to use perf_counter() than time()

    # Get the config
    config = Configuration.parse_cmd()

    # Prepare the folder where this experiment (i.e., program run) outputs and results will be saved
    experiment_id = int(time_run)
    experiment_date = time.ctime(experiment_id)
    print("CURRENT DATE TIME: ", experiment_date)
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

    # for the submission
    test_results_path = experiments_results_path + "/test_results/"
    os.makedirs(test_results_path, exist_ok=True)    # create the folder(s) if needed


    # Fix seeds for reproducibility
    SEED = config.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True # DEBUG: comment this when debugging.

    # Save in the experiment folder the command line that was used to run this program
    cmd = sys.argv[0] + ' ' + ' '.join(sys.argv[1:])
    with open(os.path.join(experiments_results_path, 'cmd.txt'), 'w') as f:
        f.write(cmd)

    # Data
    use_full_dataset = config.full_data 
    model_name = config.model_name 
    use_most_freq_words = config.freq_words
    train_val_ratio = config.train_val_ratio

    # Calculate values for the number of tweets to work with during model training   
    amount_per_it = config.amount_per_it
    full_dataset_size = 2500000
    small_dataset_size = 200000
    
    if config.amount_of_data > small_dataset_size:
        use_full_dataset = True  

    if use_full_dataset:
        total_amount_of_tweets = full_dataset_size    # max number of tweets there are in the full dataset
    else:
        total_amount_of_tweets = small_dataset_size     # max number of tweets there are in the size-reduced dataset

    # if we don't want to use the complete dataset (whether the full one or the size-reduced one)
    if config.amount_of_data != 0:
        total_amount_of_tweets = config.amount_of_data
        if total_amount_of_tweets > full_dataset_size:
            total_amount_of_tweets = full_dataset_size 
    
    number_of_iterations = int(np.ceil(total_amount_of_tweets / amount_per_it))
    print(f"We will use {total_amount_of_tweets} tweets in total. {int(train_val_ratio * total_amount_of_tweets)} for training and {int(total_amount_of_tweets * (1-train_val_ratio))} for validation.")
    print(f"{amount_per_it} tweets will be used for each of the {number_of_iterations} subset iterations (i.e., in each subset that is split in training/validation).")

    # Misc
    submit_to_kaggle = config.autosubmit
    discord_enabled = config.discord

    # Model
    n_epochs = config.n_epochs
    bs_train = config.bs_train
    bs_eval = config.bs_eval
    lr = config.lr
    fp16 = config.fp16
    weight_decay = config.weight_decay
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Create the model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=config.num_labels, local_files_only=False, ignore_mismatched_sizes=True)
    
    # If we need to load a model from a checkpoint or not
    if not (config.load_model is None):
        with open(experiment_path + config.load_model + "/config.json", 'r') as json_file:
            json_dict = json.load(json_file)
        
        model_name = json_dict["_name_or_path"]
        print("Using a checkpoint from the model architecture: ", model_name)

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = load_model_from_checkpoint(config.load_model)   # load_model should be (from a previous exp) e.g.: experiment-Thu_Jul_28_03h29m56s/checkpoints/checkpoint-14500

    model.to(C.DEVICE)  # automatic if use the Trainer()
    print("\nRunning on", C.DEVICE, " with PyTorch", torch.__version__, "\n")

    # --- TRAINING & VALIDATION ---
    if config.train:
        model = run_training(model)

    # --- TESTING ---
    if config.test:
        submit_filename = run_testing(model)

    # Submit the predictions on the test set directly to the Kaggle's leaderboard
    if submit_to_kaggle:
        res = submit_preds_on_Kaggle(submit_filename, config.tag)
        send_discord_notif("Submitted results on Kaggle!", f"{res}", green, None)

    # Time that took the whole experiment to run
    time_run = time.time() - time_run
    print(f"The program took {str(time_run/60/60)[:6]} Hours or {str(time_run/60)[:6]} minutes to run.")