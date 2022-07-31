import pandas as pd
import numpy as np


models = pd.read_csv("votingModel.csv")
names = models['Name']
weights = np.array(models['Weight'])

preds = []

for name in names:
	preds.append(np.array(pd.read_csv(name)))

n = preds[0].shape[0]
m = len(preds)
pred_final = np.empty_like(preds[0])

for i in range(n):
	pred_final[i,0] = i+1
	v = 0
	for j in range(m):
		v += (preds[j])[i,1]*weights[j]
	if v > 0:
		pred_final[i,1] = 1
	else:
		pred_final[i,1] = -1

df = pd.DataFrame({'Id': pred_final[:,0], 'Prediction': pred_final[:,1]})
df.to_csv('votingPrediction.csv', index=False)
