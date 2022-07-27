import os
import sys
import time
import random
import glob
from datetime import datetime   
import time

import numpy as np
import matplotlib
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import torch.nn.functional as F
import torch.optim as optim

from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoTokenizer

import nltk

from tensorboardX import SummaryWriter


# Other files dependencies
from config import Configuration
from config import CONSTANTS as C
from models import create_model
from losses import compute_accuracy
import utils as U

# Global variables
train_dataset, val_dataset = None, None

# Constant values for Discord bot
discord_hook= C.discord_hook
bot_name = C.bot_name
red = 15158332
green = 3066993
orange = 15105570


# DATASET CLASSES
class TrainDataset(Dataset):
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
    with open(fname, 'r', encoding='utf-8') as f:
        tweets = [line for line in f.readlines()[starting_line:end_line]]
        labels = [label] * (end_line-starting_line)

    return tweets, labels

def load_train_data(amount_per_it, iteration):
 
    if use_full_dataset == True:
        X_train_neg_path = project_path + "train_neg_full.txt"
        X_train_pos_path = project_path + "train_pos_full.txt"
        
    else:
        X_train_neg_path = project_path + "train_neg.txt"
        X_train_pos_path = project_path + "train_pos.txt"

    amount_per_it = amount_per_it // 2
    
    starting_line = iteration * amount_per_it
    end_line = starting_line + amount_per_it
    print(f"Going to read {amount_per_it*2} lines ({amount_per_it} in each of the pos and neg datasets), starting_line:{starting_line}, end_line:{end_line}")
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

def get_train_val_data(tweets, labels):
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
    
    X_train = tokenizer(X_train.tolist(), max_length=C.MAX_VECTORIZE_LEN, padding="max_length", truncation=True)
    X_val = tokenizer(X_val.tolist(), max_length=C.MAX_VECTORIZE_LEN, padding="max_length", truncation=True)

    Y_train = torch.tensor(Y_train).clone().detach()
    Y_val = torch.tensor(Y_val).clone().detach()

    train_dataset = TrainDataset(X_train, Y_train)
    val_dataset = TrainDataset(X_val, Y_val)

    train_loader = DataLoader(train_dataset,
                              batch_size=config.bs_train,
                              shuffle=True,
                              num_workers=config.data_workers)
    val_loader = DataLoader(val_dataset,
                              batch_size=config.bs_eval,
                              shuffle=False,
                              num_workers=config.data_workers)

    
    # Free some memory
    tweets, labels = [], []
    del(tweets)
    del(labels)
    
    return train_loader, val_loader


def get_test_data(tweets):
    nb_of_samples = len(tweets)
    print(f'{nb_of_samples} tweets loaded for testing.\n')
    tweets = tokenizer(tweets, max_length=C.MAX_VECTORIZE_LEN, padding="max_length", truncation=True)
    test_dataset = TestDataset(tweets)

    test_loader = DataLoader(test_dataset,
                              batch_size=config.bs_eval,
                              shuffle=False,
                              num_workers=config.data_workers)

    return test_loader

# taken from MP2022 evaluate.py
def load_model_weights(checkpoint_file, model, state_key='model_state_dict'):
    """Loads a pre-trained model."""
    if not os.path.exists(checkpoint_file):
        raise ValueError("Could not find model checkpoint {}.".format(checkpoint_file))
    checkpoint = torch.load(checkpoint_file)
    ckpt = checkpoint[state_key]
    model.load_state_dict(ckpt)

# taken from MP2022 evaluate.py
def get_model_config(model_id):
    model_id = model_id
    model_dir = U.get_model_dir(experiment_path, model_id)  # model_id should be of the form: "experiment-" + experiment_date_name + "/checkpoints/" + "some_check_point"
    model_config = Configuration.from_json(os.path.join(model_dir, 'config.json'))
    return model_config, model_dir

# taken from MP2022 evaluate.py
def load_model(model_id):
    model_config, model_dir = get_model_config(model_id)
    model = create_model(model_config)

    model.to(C.DEVICE)
    print('Model created with {} trainable parameters'.format(U.count_parameters(model)))

    # Load model weights.
    checkpoint_file = os.path.join(model_dir, 'model.pth')
    load_model_weights(checkpoint_file, model)
    print('Loaded weights from {}'.format(checkpoint_file))

    return model, model_config, model_dir


def generate_submission(Y_preds):
    
    nb_of_samples = len(Y_preds)
    results = np.zeros((nb_of_samples, 2))

    results[:,0] = np.arange(1, nb_of_samples+1).astype(np.int32)  # save the ids
    results[:,1] = [-1 if elem == 0 else 1 for elem in Y_preds]  # save the test predictions

    test_results_path = experiments_results_path + "/test_results/"
    print("\nPredictions on the test set saved in: ", test_results_path)
    os.makedirs(test_results_path, exist_ok=True)    # create the folder(s) if needed
    final_filename = f"{experiment_date_for_folder_name}-submission.csv"
    np.savetxt(test_results_path + final_filename, results, fmt="%1d", delimiter=",", header = "Id,Prediction", comments = "")
    
    return final_filename


def validate(model, data_loader):
    """
    Evaluate a model on the given dataset. This computes the loss, but does not do any backpropagation or gradient
    update.
    :param model: The model to evaluate.
    :param data_loader: The dataset.

    :return: The loss value.
    """
    # Put the model in evaluation mode.
    model.eval()

    # Some book-keeping.
    val_losses_list = []
    val_acc_list = []
    n_samples = 0

    with torch.no_grad():
        for abatch in data_loader:

            inputs = abatch['input_ids']
            labels = abatch['labels']

            inputs = inputs.to(C.DEVICE)
            labels = labels.to(C.DEVICE)

            # Get the predictions.
            model_out = model(inputs)  # model_out has two fields: 'seed' and 'predictions'

            # Compute the loss, but does NOT backpropagate (see the backward() function in models.py)
            val_loss = model.backward(model_out, labels)
            # print(val_loss)
            # Accumulate the loss and multiply with the batch size (because the last batch might have different size).
            val_losses_list.append(val_loss * inputs.shape[0])

            val_acc = compute_accuracy(model_out, labels)
            # print(val_acc)
            val_acc_list.append(val_acc)
            # print(val_acc_list)

            n_samples += inputs.shape[0]    # shape[0] needs to be batch size
    

    return val_losses_list, val_acc_list, n_samples


def run_training(model, amount_per_it, iteration):
    
    # Load training & validation data
    tweets, labels = load_train_data(amount_per_it, iteration)
    train_loader, val_loader = get_train_val_data(tweets, labels)

    # Training loop.
    global_step = 0
    train_losses_list = []
    train_acc_list = []
    best_valid_loss = float('inf')
    n_samples = 0


    for epoch in range(config.n_epochs):

        train_epoch_loss = 0
        valid_epoch_loss = 0
        valid_loss = float('inf')
        valid_acc = 0
        nb_of_validated_batches_per_epoch = 0

        for i, abatch in enumerate(train_loader):

            optimizer.zero_grad()
            
            inputs = abatch['input_ids']
            labels = abatch['labels']

            # print(abatch)
            # print(inputs.shape)
            # print(labels.shape)

            # Move data to GPU.
            inputs = inputs.to(C.DEVICE)
            labels = labels.to(C.DEVICE)


            # Get the predictions.
            model_out = model(inputs)

            # Compute gradients.
            train_losses = model.backward(model_out, labels)
            # print(train_losses)
            train_losses_list.append(train_losses * inputs.shape[0])
            train_acc = compute_accuracy(model_out, labels)
            # print(train_acc)
            train_acc_list.append(train_acc)

            n_samples += inputs.shape[0]    # shape[0] needs to be batch size

            # Clip gradient if needed; can be useful for exploding gradients in RNNs
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

            # Update params.
            optimizer.step()

            # if global_step % (config.print_every - 1) == 0:
            if (i % 1000) == 0:
                loss_string = ' '.join(['train loss: {:.5f} | train accuracy: {:.3f}'.format(train_losses, train_acc)])
                print(f'[TRAIN--- epoch: {epoch + 1} | batch: {i + 1}] | {loss_string}')

            # if global_step % (config.eval_every - 1) == 0:      # we only eval every config.eval_every mini-batch; Note: global_step counts how many mini-batches have been used (right?)
            if (i % 1000) == 0:
                model.eval()
                val_losses_list, val_acc_list, n_samples = validate(model, val_loader)
                
                valid_epoch_loss += sum(val_losses_list) / n_samples    # NOT SURE HERE (???); Note however that we only get a new loss value added when we evaluate, so not all the mini-batches will be considered (which is not the case for the epoch training loss)
                valid_loss = sum(val_losses_list) / n_samples
                valid_acc = sum(val_acc_list) / len(val_acc_list)   # NOT SURE HERE (???); Note however that we only get a new loss value added when we evaluate, so not all the mini-batches will be considered (which is not the case for the epoch training loss)
                nb_of_validated_batches_per_epoch += 1                
                
                loss_string = ' '.join(['val loss: {:.5f} | val accuracy: {:.3f}'.format(valid_loss, valid_acc)])
                print(f'\t\t\t\t\t\t\t\t[VALID--- epoch: {epoch + 1} | batch: {i + 1}] | {loss_string}')
                
                # Save the current model if it's better than what we've seen before.
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    torch.save({
                        'iteration': i,
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': train_losses,
                        'valid_loss': valid_loss,
                    },  checkpoint_path + "model_checkpoint.pth")
                    print("Saving new checkpoint for best model!")

                # Make sure the model is in training mode again.
                model.train()

            # Accumulate the training loss over the mini-batches to get the epoch training loss
            train_epoch_loss += train_losses * inputs.shape[0]
            
            global_step += 1    # to count the total number of mini-batches seen from the very beginning
        
        lrs.step()

        # print("Number of samples (i.e., sequences) in train_loader: ", len(train_loader.dataset))   # BUT we do not have to use that to normalize the epoch training loss since train_losses contains the "total_loss" for a mini-batch already. Therefore we will simply have to normalize with the number of mini-batches.
        # print("Number of mini-batches in the train_loader: ", len(train_loader))    # However, see the second warning under the Dataloader class: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader

        print(f"Epoch {epoch+1} training loss: (normalized by the number of mini-batches seen this epoch) ", train_epoch_loss / len(train_loader))   # divide by the number of mini-batches all for which we computed the epoch partial loss
        print(f"Epoch {epoch+1} validation loss: (normalized by the number of mini-batches validated in this epoch)", valid_epoch_loss / nb_of_validated_batches_per_epoch, "\n\n")


def run_testing(model):

    # Load test data
    tweets = load_test_data()
    test_loader = get_test_data(tweets)

    Y_test_preds = []

    # Put the model in evaluation mode.
    model.eval()

    with torch.no_grad():
        for abatch in test_loader:

            inputs = abatch['input_ids']

            inputs = inputs.to(C.DEVICE)

            # Get the predictions.
            model_out = model(inputs)
            logits = model_out['predictions']
            predictions = np.argmax(logits.cpu().detach().numpy(), axis=-1)
            Y_test_preds.append(predictions)

    predictions = [x for batch in Y_test_preds for x in batch]

    return predictions



# DISCORD
def send_discord_notif(title, content, color, error=None):
    import requests

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


if __name__ == '__main__':

    torch.cuda.empty_cache()    

    # To time the duration of the experiment
    time_run = time.time()     # better to use perf_counter() than time()

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
    checkpoint_path = experiments_results_path + "/checkpoints/"
    os.makedirs(checkpoint_path, exist_ok=True)    # create the experiment folder(s) needed

    print("The project path is: ", project_path)
    print("The experiment path is: ", experiment_path)
    print("The model checkpoints will be saved at: ", checkpoint_path, "\n")


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

    tokenizer_name = "bert-base-uncased"# "xlnet-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    # Model
    n_epochs = config.n_epochs
    bs_train = config.bs_train
    bs_eval = config.bs_eval
    lr = config.lr
    weight_decay = config.weight_decay


    # Misc
    submit_to_kaggle = config.autosubmit
    discord_enabled = config.discord


    # Create the model
    model = create_model(config)

    def init_weights(model):
        for name, param in model.named_parameters():
            torch.nn.init.uniform_(param.data, -0.1, 0.1)
        
    # Initialize the model weights as preferred
    # model.apply(init_weights)

    # Load a pre-trained model or not
    if not(config.load_model is None):
        print("Past model will start to load...")
        model, _, _ = load_model(config.load_model)
        print("Model loading worked!")

    # Place model on gpu
    model.to(C.DEVICE) 
    
    print("Model running on", C.DEVICE)
    print('Model initialized with {} trainable parameters'.format(U.count_parameters(model)))

    # Create or a new experiment ID and folder where to store logs and config.
    experiment_id = int(time.time())
    # experiment_date_for_file_name = experiment_id
    experiment_date = time.ctime(experiment_id)
    experiment_date_for_file_name = experiment_date.replace(" ", "_").replace(":", "h")[:-8] + experiment_date[-8:-5].replace(":", "m") + "s"
    experiment_name = model.model_name()


    # Create Tensorboard logger.
    writer = SummaryWriter(os.path.join("./", 'model_logs'))



    # Define the optimizer.
    # optimizer = optim.SGD(model.parameters(), lr=config.lr)
    # optimizer = torch.optim.LBFGS(model.parameters(), history_size=10, max_iter=4, line_search_fn=True)     # -> need to pass a closure argument to the step() function. Not compatible with this current code + memory intensive. However, it is an optimizer that can be good for LSTM. 
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay = 5e-5, amsgrad=False)      # weight_decay = 5e-6
    lrs = optim.lr_scheduler.StepLR(optimizer, 50, 0.7, verbose=True)
    # lrs = optim.lr_scheduler.StepLR(optimizer, 12, 0.8, verbose=True)
    # lrs = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, verbose=True)
    # lrs = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0005, max_lr=0.01, verbose=True)
    # lrs = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.000001, verbose=True)     # T_max = 100, 200



    model.to(C.DEVICE)
    print("\nRunning on", C.DEVICE, " with PyTorch", torch.__version__, "\n")

    # --- TRAINING & VALIDATION ---
    if config.train:
        for iteration in range(number_of_iterations)[config.start_at_it:]:
            trained_model = run_training(model, amount_per_it, iteration)

            # torch.cuda.empty_cache()    # ???
            send_discord_notif("Continuing Training", f"currently finished subset iteration: {iteration+1}/{number_of_iterations} ", orange, None)
            print(f"{iteration+1} out of {number_of_iterations} subset iteration(s) done!")

    # --- TESTING ---
    if config.test:
        predictions = run_testing(model)
        submit_filename = generate_submission(predictions)
        send_discord_notif("Finished to predict on test set", f"Everything ran without issues!", green, None)


    # Submit the predictions on the test set directly to the Kaggle's leaderboard
    if submit_to_kaggle:
        res = submit_preds_on_Kaggle(submit_filename, config.tag)
        send_discord_notif("Submitted results on Kaggle!", f"{res}", green, None)

    # Time that took the whole experiment to run
    time_run = time.time() - time_run
    print(f"The program took {str(time_run/60)[:6]} minutes to run.")