from transformers import AutoConfig
from transformers import AdamW

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import random

from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import pipeline
from datasets import load_metric
from transformers import TrainingArguments     # https://huggingface.co/transformers/v3.0.2/main_classes/trainer.html#transformers.TFTrainingArguments
from transformers import Trainer    # https://huggingface.co/transformers/v3.0.2/main_classes/trainer
import time
import requests
#

# Handle configurations
import transformer_config
from transformer_config import Configuration
from transformer_config import CONSTANTS as C

# DATASET CLASSES
class TrainDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
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
      return len(self.encodings["input_ids"])

# Constant Values

red = 15158332
green = 3066993
orange = 15105570

# CONSTANT VALUES

device = C.DEVICE #"cuda:0" if torch.cuda.is_available() else "cpu"
path = C.ABS_PATH # ""
model_path_checkpoints = C.model_path_checkpoints #"train_results/"
metric = load_metric("accuracy")

#discord notifs constants
discord_hook= C.discord_hook

bot_name = C.bot_name
# global vars
train_dataset,val_dataset = None,None




# DATA MANAGEMENT
def read_file(file_name_label_tuple,starting_line=0,end_line=0):
        fname,label = file_name_label_tuple
        tweets,labels = [],[]
        with open( fname, 'r', encoding='utf-8') as f:
            tweets = [line for line in f.readlines()[starting_line:end_line] ]
            labels = [label] * (end_line-starting_line)
        return(tweets,  labels   )

def load_train_data(amount_per_batch,iteration):
 
    if use_full_dataset == True:
        X_train_neg_path = path + "train_neg_full.txt"
        X_train_pos_path = path + "train_pos_full.txt"
        
    else:
        X_train_neg_path = path + "train_neg.txt"
        X_train_pos_path = path + "train_pos.txt"

    amount_per_batch=amount_per_batch//2
    

    starting_line = iteration*amount_per_batch
    end_line = starting_line+amount_per_batch
    print(f"Going to read {amount_per_batch*2} lines, starting_line:{starting_line}, end_line:{end_line}")
    tweets,  labels = read_file( (X_train_neg_path,0),starting_line=starting_line,end_line=end_line )

    tweets_2,  labels_2 = read_file( (X_train_pos_path,1),starting_line=starting_line,end_line=end_line )
    tweets+=tweets_2
    tweets_2=[]
    del(tweets_2)
    labels+=labels_2
    labels_2=[]
    del(labels_2)
    print(f"Loaded {len(tweets)} tweets !")
    return np.array(tweets),np.array(labels)

def load_test_data():
    filename = path + "test_data.txt"
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
    return X_train,X_val

def get_train_val_datasets(tweets,labels):
    nb_of_samples = len(tweets)
    shuffled_indices = np.random.permutation(nb_of_samples)
    split_idx = int(train_val_ratio * nb_of_samples)

    train_indices = shuffled_indices[:split_idx]
    val_indices = shuffled_indices[split_idx:]

    print("Number of indices for training: ", len(train_indices))
    print("Number of indices for validation: ", len(val_indices))

    if use_most_freq_words:
        X_train,X_val = vectorize_data(tweets,train_indices,val_indices)
    else:
        X_train,X_val = tweets[train_indices],tweets[val_indices]
    
    Y_train = labels[train_indices]
    Y_val = labels[val_indices]

    #tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    X_train = tokenizer(X_train.tolist(), max_length=256, padding="max_length", truncation=True)
    X_val = tokenizer(X_val.tolist(), max_length=256, padding="max_length", truncation=True)

    Y_train = torch.tensor(Y_train).clone().detach()
    Y_val = torch.tensor(Y_val).clone().detach()

    train_dataset = TrainDataset(X_train, Y_train)
    val_dataset = TrainDataset(X_val, Y_val)
    return train_dataset,val_dataset

def get_test_dataset(tweets):
    nb_of_samples = len(tweets)
    print(f'{nb_of_samples} tweets loaded for testing.\n')
    tweets = tokenizer(tweets, max_length=256, padding="max_length", truncation=True)
    tweets = TestDataset(tweets  )
    return tweets

def compute_metrics(eval_pred): 
      logits, labels = eval_pred
      predictions = np.argmax(logits, axis=-1)
      # print("Predictions: ", predictions)
      # print("Labels: ", labels)
      # input()

      return metric.compute(predictions=predictions, references=labels)
    
def train(model,train_dataset_param,val_dataset_param):
    training_args = TrainingArguments(output_dir="./train_results", 
                                    per_device_train_batch_size=bs_train, 
                                    per_device_eval_batch_size=bs_eval, 
                                    learning_rate=lr, 
                                    evaluation_strategy="epoch",
                                    gradient_accumulation_steps=4,
                                    gradient_checkpointing=True,
                                    overwrite_output_dir=True,
                                    save_total_limit=2,
                                    fp16=fp16,
                                    seed=SEED,
                                  #   warmup_steps=500,                # number of warmup steps for learning rate scheduler
                                    weight_decay=weight_decay,               # strength of weight decay
                                    logging_dir='./logs',            # directory for storing logs
                                    logging_steps=100, 
                                    num_train_epochs=n_epochs)
    if not (train_dataset_param is  None) and not (val_dataset_param is None):
        train_dataset,val_dataset = train_dataset_param,val_dataset_param

    trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=train_dataset,
      eval_dataset=val_dataset,
      compute_metrics=compute_metrics
  )
    trainer.train()

def load_model_from_checkpoint(selected_checkpoint):
    model_path = model_path_checkpoints+f"checkpoint-{selected_checkpoint}"
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
    print(f"Loaded model from: {model_path}")
    return model

def load_model_from_path(path):
    print(f"Loaded model from: {path}")
    return AutoModelForSequenceClassification.from_pretrained(path, num_labels=2)

def test(model):
    test_trainer = Trainer(model)
    tweets = get_test_dataset(load_test_data())

    raw_pred, _, _ = test_trainer.predict(tweets)     # only predictions to return, no label ids, no metrics; see HF Trainer doc
    Y_test_pred = np.argmax(raw_pred, axis=1)
    return Y_test_pred

def generate_submission(Y_preds):
    nb_of_samples=len(Y_preds)
    results = np.zeros((nb_of_samples, 2))

    results[:,0] = np.arange(1, nb_of_samples+1).astype(np.int32)  # save the ids
    results[:,1] = [-1 if elem == 0 else 1 for elem in Y_preds]  # save the test predictions

    experiment_id = int(time.time())
    experiment_date = time.ctime(experiment_id)
    experiment_date_for_file_name = experiment_date.replace(" ", "_").replace(":", "h")[:-8] + experiment_date[-8:-5].replace(":", "m") + "s"


    submission_results_path = experiment_date_for_file_name
    final_filename = f"{path}test_results/{submission_results_path}-submission.csv"
    np.savetxt( final_filename, results,fmt="%1d", delimiter=",", header = "Id,Prediction", comments = "")
    
    return final_filename

def load_and_train(model,amount_per_batch,iteration):
    # Load TRAINING-VALIDATION DATA
    if model is None:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    tweets,labels = load_train_data(amount_per_batch,iteration)
    train_dataset,val_dataset = get_train_val_datasets(tweets,labels)
    tweets,labels = [],[]

    del(tweets)
    del(labels)
    # TRAIN

    
    
    train(model,train_dataset,val_dataset)

    return model

### DISCORD ###  
def send_discord_notif(title,content,color,error=None):
    if not discord_enabled:
        return
    if error is None:
        msg = "Little Update on my status"
    else: msg = error

    data = {
        "username": bot_name,
        "content": msg,
    }
    data["embeds"] = [

        {
        "title":title,
        "description":content,
        "color":color
    }
]

    result = requests.post(discord_hook, json = data)
    try:
        result.raise_for_status()
    except requests.exceptions.HTTPError as err:
        print(err)
    #else:
    #    print("Payload delivered successfully, code {}.".format(result.status_code))

# KAGGLE API
def submit_tests(submit_filename,msg):
    import kaggle
    api = kaggle.api
    api.get_config_value("username")
    res = api.competition_submit(submit_filename,f"Automatic File submission test: {msg}","cil-text-classification-2022")
    print("res:",res)
    return res


if __name__ == "__main__":
    # FIRST INIT ALL GLOBAL VARS FROM CONFIG
    curr_config = Configuration.parse_cmd()
    #Randomness
    SEED = curr_config.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = False

<<<<<<< HEAD
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
    print(f"{amount_per_it} tweets will be used for each of the {number_of_iterations} subset iterations (i.e., in each subset that is split in training/validation).")

    # Model
    n_epochs = config.n_epochs
    bs_train = config.bs_train
    bs_eval = config.bs_eval
    lr = config.lr
    fp16 = config.fp16
    weight_decay = config.weight_decay
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
=======
    # Train Data
    use_full_dataset = curr_config.full_data 
    model_name = curr_config.model_name 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    use_most_freq_words = curr_config.freq_words
    train_val_ratio = curr_config.train_val_ratio 
>>>>>>> fa4c73560f960971e342156092056eb4554c77dc

    # Miscs
    submit_to_kaggle = curr_config.autosubmit
    discord_enabled =  curr_config.discord
    
    # Model
    n_epochs = curr_config.n_epochs
    bs_train = curr_config.bs_train
    bs_eval = curr_config.bs_eval
    lr = curr_config.lr
    fp16 = curr_config.fp16

    weight_decay = curr_config.weight_decay
    
    if not (curr_config.load_model is None ):
        model = load_model_from_checkpoint(curr_config.load_model)
        
    # Calculate values for the amount of tweet to work on during training.    
    amount_per_it = curr_config.amount_per_it
    if use_full_dataset:
        total_amout_of_tweets = 2500000
    else:total_amout_of_tweets =  200000

    if curr_config.amount_of_data !=0:
        total_amout_of_tweets = curr_config.amount_of_data
        
    # Training parameters:
    if curr_config.train:

        number_of_iterations = int(np.ceil(total_amout_of_tweets / amount_per_it))
        print(f"Going to do {number_of_iterations} iterations to do {total_amout_of_tweets} tweets in batch sizes of {amount_per_it}")

        send_discord_notif("Starting Training",f"Going to do {number_of_iterations} iterations\
        to do {total_amout_of_tweets} tweets in batch sizes of {amount_per_it}",orange,None)

        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        try:
            for iteration in range(number_of_iterations)[curr_config.start_at_it:]:
                model = load_and_train(model,amount_per_it,iteration)
                send_discord_notif("Continuing Training",f"currently finished iteration: {iteration+1}/{number_of_iterations} ",orange,None)
        except Exception as e:
            print("GOT ERROR:",str(e))
            send_discord_notif("ERROR WHILE TRAINING",str(e),red,f"Got the error at iteration:{iteration+1}/{number_of_iterations}")
            raise(e)
            
        send_discord_notif("Finished Training",f"Did {number_of_iterations}  in batch sizes of {amount_per_it} without problem",green,None)

    if curr_config.test:
        try:
            Y_test_pred = test(model)
            submit_filename = generate_submission(Y_test_pred)
        except Exception as e:
            print("GOT ERROR:",str(e))
            send_discord_notif("ERROR WHILE TESTING",str(e),red,f"Got the error while trying to test")
            raise(e)

        send_discord_notif("Finished Testing",f"Everything ran without issues !",green,None)

    if submit_to_kaggle:
        res = submit_tests(submit_filename,"")
        send_discord_notif("Uploaded results",f"{res}",green,None)


    
   
