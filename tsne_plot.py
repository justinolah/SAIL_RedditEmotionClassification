import pandas as pd
from helpers import *
from transformers import BertModel, BertTokenizerFast

from torch.utils.data import TensorDataset, DataLoader

from sklearn.manifold import TSNE

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

seed_value=42
np.random.seed(seed_value)
torch.manual_seed(seed_value)
random.seed(seed_value) 
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class BERT_Model(nn.Module):
	def __init__(self, bert, numEmotions):
		super(BERT_Model, self).__init__()
		self.bert = bert 
		self.fc = nn.Linear(768, numEmotions)

	def forward(self, sent_id, mask):
		_, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
		#out = self.fc(cls_hs)
		return cls_hs

def makeBERTDataset(data, tokenizer, max_length, emotions, new_emotions=None, idx_map=None):
	if idx_map is None:
		data.labels = data.labels.apply(lambda x: getYMatrix(x,len(emotions)))
	else:
		data.labels = data.labels.apply(lambda x: getYMatrixWithMap(x, len(new_emotions), idx_map))

	tokens = tokenizer.batch_encode_plus(
		data.text.tolist(),
		max_length = max_length,
		padding='max_length',
		truncation=True
	)

	seq = torch.tensor(tokens['input_ids'])
	mask = torch.tensor(tokens['attention_mask'])
	y = torch.tensor(data.labels.tolist())

	dataset = TensorDataset(seq, mask, y)
	return dataset

def main():
	if torch.cuda.is_available():
		print("Using cuda...")
		device = torch.device("cuda")
	else:
		print("Using cpu...")
		device = torch.device("cpu")

	max_length = 128
	batch_size = 16

	emotions = getEmotions()
	emotions.remove("neutral")

	train = getTrainSet()
	test = getTestSet()
	dev = getValSet()
	all_data = test #pd.concat([train, test, dev])

	Y = all_data.labels.apply(lambda x: emotions[x.split(',')[0]] if len(x.split(',')) > 0 else 'neutral')

	tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

	data_set = makeBERTDataset(all_data, tokenizer, max_length, emotions)
	dataloader = DataLoader(data_set, batch_size=batch_size)

	bert = BertModel.from_pretrained('bert-base-uncased')

	model = BERT_Model(bert, len(emotions))
	model = model.to(device)

	checkpoint = torch.load("bert_best.pt")
	model.load_state_dict(checkpoint['model_state_dict'])
	model.eval()

	outputs = []	

	for batch in tqdm(dataloader):
		seq, mask, _ = batch

		output = model(seq.to(device), mask.to(device))
		outputs.append(output.detach().cpu())

	vectors = torch.Tensor(len(dataloader), 768)
	torch.cat(outputs, out=vectors)

	tsne = TSNE(n_components = 2, perplexity = 30, random_state = 6, learning_rate = 500, n_iter = 1500, verbose=2, n_jobs=10)

	reduced = tsne.fit_transform(vectors)

	df = pd.DataFrame(reduced)
	df = pd.concat([df,Y], axis=1)
	sns.FacetGrid(df, hue="label" , size=6).map(plt.scatter, 0, 1).add_legend()
	plt.savefig("tsne.png", format="png")
	plt.show()





if __name__ == '__main__':
	main()