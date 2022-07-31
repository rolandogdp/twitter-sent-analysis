import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from transformers import Trainer

class MyScalTrainer(Trainer):


	def compute_loss(self, model, inputs, return_outputs=False):
		outputs = model(**inputs, return_dict = True)
		loss = outputs.get("loss")
		return (loss, outputs) if return_outputs else loss



	def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys = None):
		labels = inputs.get("labels")
		device = 'cuda' if torch.cuda.is_available() else 'cpu'
		if labels is not None:
			labels.to(device)
		#model.to(device)
		loss = None
		with torch.no_grad():
			logitsTrue = model.predict(**inputs)
			#print(logitsTrue)
			if labels is not None:
				loss_fce = nn.CrossEntropyLoss()
				loss = loss_fce(logitsTrue.view(-1, 2).to(device), labels.view(-1).to(device)).to(device)
				loss = loss.mean().detach()

		if prediction_loss_only:
			return (loss, None, None)

		return (loss, logitsTrue, labels)

