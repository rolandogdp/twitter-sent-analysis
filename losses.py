# Losses for training 

import numpy as np
import torch.nn as nn
from datasets import load_metric

from sklearn.metrics import accuracy_score

def compute_accuracy(outputs, labels):
    logits = outputs['predictions']
    predictions = np.argmax(logits.cpu().detach().numpy(), axis=-1)
    # print("Predictions: ", predictions)
    # print("Labels: ", labels)
    return accuracy_score(labels.cpu(), predictions)

def criterion(predictions, labels):
    # print("HERE IN CRITERION---")
    # print(predictions)
    # print(labels)

    loss = nn.CrossEntropyLoss()

    return loss(predictions, labels)

