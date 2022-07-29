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

def fgsm_attack(embeddings, epsilon, embeddings_grad):
	perturbed_embeddings = embeddings + epsilon*F.normalize(torch.sign(embeddings_grad), p=2.0, dim=1, eps=1e-9);
	return perturbed_embeddings

def fgm_attack(embeddings, epsilon, embeddings_grad):
	perturbed_embeddings = embeddings + epsilon*F.normalize(embeddings_grad, p=2.0, dim=1, eps=1e-9);
	return perturbed_embeddings

def loss_contrastive(h_true, h_adversarial, temperature):
	batch_size = h_true.shape[0]
	loss = 0.0
	eps = 1e-9
	cos = nn.CosineSimilarity(dim=1, eps=eps)
	for i in range(batch_size):
		s = cos(h_true[i], h_adversarial);
		s = s/temperature
		s = s - s.mean()
		l = F.log_softmax(s, dim=0, dtype=torch.float32)
		loss -= l[i]

	loss = loss/batch_size
	return loss

class ContextPooler(nn.Module):
    def __init__(self, hidden_size, dropout_prob = 0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.act = nn.GELU()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.

        context_token = hidden_states[:, 0]
        context_token = self.dropout(context_token)
        pooled_output = self.dense(context_token)
        pooled_output = self.act(pooled_output)
        return pooled_output

    @property
    def output_dim(self):
        return self.hidden_size

class MyScalModel(PreTrainedModel):
	def __init__(self, config = None, model_name = "microsoft/deberta-v3-base", num_labels = 2, temperature = 0.05, epsilon = 0.2, alpha = 0.5, beta = 1.0):
		model = AutoModel.from_pretrained(model_name, num_labels = num_labels)
		self.config = model.config
		#self.config.layer_norm_eps = 1e-6
		super(MyScalModel, self).__init__(self.config)

		self.num_labels = num_labels
		self.model_name = model_name
		self.features_dim = self.config.hidden_size
		self.embeddings = model.embeddings
		#For both true encoding and adversarial attack
		self.encoder = model.encoder
		self.pooler = ContextPooler(self.features_dim, 0.1)
		self.output_dim = self.features_dim
		self.classifier = nn.Linear(self.output_dim, self.num_labels)
		#For contrastive learning
		self.mlp = nn.Sequential(
			nn.Linear(self.features_dim, 1024),
			nn.GELU(),
			nn.Dropout(0.1),
			nn.Linear(1024, self.features_dim),
		)
		
		self.mlp.apply(weights_init)
		self.pooler.apply(weights_init)
		weights_init(self.classifier)
		self.temperature = temperature #temperature for ct loss, taken from the paper
		self.epsilon = epsilon
		self.alpha = alpha
		self.beta = beta

		del model

	def forward(
		self, 
		input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
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
		embedding_output = self.embeddings(input_ids.to(torch.long), token_type_ids.to(torch.long), 
			position_ids, mask = attention_mask, inputs_embeds = inputs_embeds);
		embeddingTrue = embedding_output

		encoder_outputs = self.encoder(
            embeddingTrue,
            attention_mask,
            output_hidden_states=True,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )
		encodedTrue_layers = encoder_outputs[1]
		sequenceTrue_output = encodedTrue_layers[-1]
		logitsTrue = self.classifier(self.pooler(sequenceTrue_output))
		loss = torch.tensor(0.)
		loss_ct = torch.tensor(0.)
		loss_adversarial = torch.tensor(0.)
		loss_kl = torch.tensor(0.)

		if labels is not None:
			loss_fce = nn.CrossEntropyLoss()
			loss = loss_fce(logitsTrue.view(-1, self.num_labels), labels.view(-1))

		if embeddingTrue.requires_grad == True:
			#FGM
			embeddingTrue.retain_grad()
			self.zero_grad();
			loss.backward(retain_graph = True);
			grad2 = embeddingTrue.grad.data
			#embeddingTrue.requires_grad = False
			self.zero_grad();
			grad2.requires_grad = False
			embeddingAdversarial = fgm_attack(embeddingTrue, self.epsilon, grad2)

			#do adversarial pass
			#embedding_output['embeddings'] = embeddingAdversarial
			encoder_outputs_adversarial = self.encoder(
				embeddingAdversarial,
				attention_mask,
				output_hidden_states=True,
	            output_attentions=output_attentions,
	            return_dict=return_dict,
			)
			encodedAdversarial_layers = encoder_outputs_adversarial[1]
			sequenceAdversarial_output = encodedAdversarial_layers[-1]
			logitsAdversarial = self.classifier(self.pooler(sequenceAdversarial_output))
			loss_adversarial = loss_fce(logitsAdversarial.view(-1, self.num_labels), labels.view(-1))
			#embedding_output['embeddings'] = embeddingTrue

			#contrastive loss
			h_true = self.mlp(sequenceTrue_output)
			h_adversarial = self.mlp(sequenceAdversarial_output)
			loss_ct = loss_contrastive(h_true, h_adversarial, self.temperature)

			#kl_loss
			loss_kl = compute_kl_loss(logitsTrue, logitsAdversarial)


		if not return_dict:
			output = (logitsTrue, ) + encoder_outputs[1:]
			return ((loss, ) + output) if loss is not None else output

		return SequenceClassifierOutput(
			loss = (loss+loss_adversarial+self.alpha*loss_ct+self.beta*loss_kl).mean(), 
			logits = logitsTrue,
			#logitsAdversarial = logitsAdversarial,
			hidden_states = encoder_outputs.hidden_states,
			attentions = encoder_outputs.attentions
		)

	def predict(
		self, 
		input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
		):
		device = 'cuda' if torch.cuda.is_available() else 'cpu'
		with torch.no_grad():
			if attention_mask is None:
				attention_mask = torch.ones_like(input_ids).to(device)
			if token_type_ids is None:
				token_type_ids = torch.zeros_like(input_ids).to(device)

			if input_ids is not None:
				input_ids = input_ids.to(torch.long).to(device)
			if token_type_ids is not None:
				token_type_ids = token_type_ids.to(torch.long).to(device)
			if attention_mask is not None:
				attention_mask = attention_mask.to(device)
			if inputs_embeds is not None:
				inputs_embeds = inputs_embeds.to(device)

			embedding_output = self.embeddings(input_ids,token_type_ids, 
				position_ids, mask = attention_mask, inputs_embeds = inputs_embeds);
			embeddingTrue = embedding_output

			encoder_outputs = self.encoder(
	            embeddingTrue,
	            attention_mask,
	            output_hidden_states=True,
	            output_attentions=output_attentions,
	            return_dict=return_dict,
	        )
			encodedTrue_layers = encoder_outputs[1]
			sequenceTrue_output = encodedTrue_layers[-1]
			logitsTrue = self.classifier(self.pooler(sequenceTrue_output))
		

		return logitsTrue



