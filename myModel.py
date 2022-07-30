import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from typing import Optional
from transformers import AutoModel
from transformers import AutoModelForSequenceClassification
from transformers import DebertaV2ForSequenceClassification
from transformers import DebertaV2Config
from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


def compute_kl_loss(p, q, pad_mask=None):
    
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

def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)

class MyModel(PreTrainedModel):
	def __init__(self, model_name = "vinai/bertweet-base" , config = None):
		model = AutoModel.from_pretrained(model_name, num_labels = 2)
		#self.config.layer_norm_eps = 1e-6
		super(MyModel, self).__init__(model.config)
		self.config = model.config
		self.model = model;
		self.features_dim = self.config.hidden_size
		self.dropoutProb = 0.3;
		self.classifier = nn.Sequential(
			nn.Linear(self.features_dim, 1024),
			nn.Dropout(self.dropoutProb),
			nn.Linear(1024, 1024),
			nn.Dropout(self.dropoutProb),
			nn.Linear(1024, 2)
			)
		self.classifier.apply(weights_init)

	def forward(
		self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict = True,
        ):
		#First run of true embedding and predictions
		if attention_mask is None:
			attention_mask = torch.ones_like(input_ids)
		if token_type_ids is None:
			token_type_ids = torch.zeros_like(input_ids)
		
		#with torch.enable_grad():
		outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
		)
		#print(self.model.modules)
		#print(outputs)
		sequence_output = outputs[0]
		#print(sequence_output.shape)
		logits1 = self.classifier(sequence_output[:,0])
		logits2 = self.classifier(sequence_output[:,0])
	

		logits = 0.5*(logits1+logits2)
		loss = torch.tensor(0.0)
		if labels is not None:
			loss_fce = nn.CrossEntropyLoss()
			kl_loss = compute_kl_loss(logits1, logits2)
			ce_loss = loss_fce(logits.view(-1,2), labels.view(-1))
			loss = ce_loss+kl_loss
		

		if not return_dict:
			output = (logits, ) + outputs[1:]
			return ((loss, ) + output) if loss is not None else output

		return SequenceClassifierOutput(
			loss = loss.mean(), 
			logits = logits,
			#logitsAdversarial = logitsAdversarial,
			hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
		)

	