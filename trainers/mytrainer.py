import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from transformers import Trainer

class MyTrainer(Trainer):

	def compute_kl_loss(self, p, q, pad_mask=None):
    
		p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
		q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

		# pad_mask is for seq-level tasks
		if pad_mask is not None:
			p_loss.masked_fill_(pad_mask, 0.)
			q_loss.masked_fill_(pad_mask, 0.)

		# You can choose whether to use function "sum" and "mean" depending on your task
		p_loss = p_loss.mean()
		q_loss = q_loss.mean()

		loss = (p_loss + q_loss) / 2.0
		return loss

	def compute_loss(self, model, inputs, return_outputs=False): #use R-Drop
		labels = inputs.get("labels")
		#print(inputs)
		outputs1 = model(**inputs)
		outputs2 = model(**inputs)

		logits1 = outputs1.get("logits")
		logits2 = outputs2.get("logits")

		loss_fct = nn.CrossEntropyLoss()
		loss_ce1 = loss_fct(logits1.view(-1, self.model.config.num_labels), labels.view(-1))
		loss_ce2 = loss_fct(logits2.view(-1, self.model.config.num_labels), labels.view(-1))
		kl_loss = self.compute_kl_loss(logits1, logits2)
		loss = (loss_ce2+loss_ce1)*0.5 + 0.5*kl_loss; #1.0 taken from the paper

		return (loss, outputs1) if return_outputs else loss