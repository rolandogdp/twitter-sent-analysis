import torch
import torch.nn.functional as F
import torch.nn as nn


class MyClassifier(nn.Module):
	def __init__(self, feature_dim=768, num_labels = 2):
		super(MyClassifier, self).__init__()
		self.fc1 = nn.Sequential(
			nn.Linear(feature_dim, 1024),
			nn.LeakyReLU(),
			nn.LayerNorm(1024),
			nn.Dropout(0.5),
			nn.Linear(1024, 1024),
			nn.LeakyReLU(),
			nn.LayerNorm(1024),
			nn.Dropout(0.5),
			nn.Linear(1024, feature_dim),
			nn.LeakyReLU(),
			nn.LayerNorm(feature_dim),
			nn.Dropout(0.5),
		)
		self.fc2 = nn.Sequential(
			nn.Linear(feature_dim, num_labels)
		)
		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight, gain=0.01)
				if m.bias is not None:
					torch.nn.init.zeros_(m.bias)

	def forward(self, features):
		non_linear_features = self.fc1(features)
		features = features + non_linear_features
		outputs = self.fc2(features)
		return outputs