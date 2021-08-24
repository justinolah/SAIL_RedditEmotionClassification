import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from helpers import *
from collections import Counter, defaultdict

from wordcloud import WordCloud

from transformers import BertModel, BertTokenizerFast

seed_value=42
np.random.seed(seed_value)
torch.manual_seed(seed_value)
random.seed(seed_value) 
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

latex_special_token = ["!@#$%^&*()"]

common_words = ["the", "to", "that", "for", "in", "of", "and", "it", "be", "is", "them", "they", "we", "you", "about", "has", "have", "had", "this", "about", "on", "[UNK]", "at", "there", "my", "re", "made"]

class BERT_Model(nn.Module):
	def __init__(self, bert, numEmotions):
		super(BERT_Model, self).__init__()
		self.bert = bert 
		self.fc = nn.Linear(768, numEmotions)

	def forward(self, sent_id, mask):
		_, cls_hs, attentions = self.bert(sent_id, attention_mask=mask, output_attentions=True, return_dict=False)
		out = self.fc(cls_hs)
		return out, attentions

def KmaxelementsDict(dict, k):
	kdict = {}

	for i in range(0, k): 
		maxword = ""
		maxval = 0
		
		for (word, val) in dict.items():    
			if val > maxval and word not in kdict and len(word) > 1 and word not in '''[]'`’-*/\\:;~%,."()''' and word not in common_words:
				maxword, maxval = word, val

		kdict[maxword] = maxval
		
	return kdict 

def main():
	if torch.cuda.is_available():
		print("Using cuda...")
		device = torch.device("cuda")
	else:
		print("Using cpu...")
		device = torch.device("cpu")

	emotions = getEmotions()
	emotions.remove("neutral")

	counts = dict(zip(emotions,[Counter() for i in range(len(emotions))]))
	word_scores = dict(zip(emotions,[defaultdict(float) for i in range(len(emotions))]))

	train = getTrainSet()
	test = getTestSet()
	dev = getValSet()
	data = pd.concat([train, test, dev])

	max_length = 128

	tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
	bert = BertModel.from_pretrained('bert-base-uncased')
	model = BERT_Model(bert, len(emotions))
	model = model.to(device)

	checkpoint = torch.load("bert_best.pt")
	model.load_state_dict(checkpoint['model_state_dict'])
	model.eval()

	tokens = tokenizer.batch_encode_plus(
		data.text.tolist(),
		max_length = max_length,
		padding='max_length',
		truncation=True
	)

	seqs = torch.tensor(tokens['input_ids'])
	masks = torch.tensor(tokens['attention_mask'])
	labels = data.labels.tolist()

	for i in tqdm(range(len(data))):
		seq = torch.tensor(seqs[i])
		mask = torch.tensor(masks[i])

		tokens = tokenizer.convert_ids_to_tokens(seq)
		tokens = [token for token in tokens if token not in ['[PAD]','[CLS]','[SEP]']]

		if len(tokens) == 0:
			continue

		softmax = nn.Softmax(dim=0)

		output, attention = model(seq.unsqueeze_(0).to(device), mask.unsqueeze_(0).to(device))
		#attention.size() -> 1,12,128,128

		length = torch.sum(mask)
		output = output.cpu()

		output = (output > 0.5).int()

		actual_labels = [emotions[int(label)] for label in labels[i].split(',') if int(label) < len(emotions)]
		predicted_labels = [emotions[index] for index, val in enumerate(output[0].tolist()) if val == 1]
		
		layer = attention[-1].cpu()
		
		att = torch.zeros(len(tokens))

		for j in range(12):
			head = layer[0,j,:length,:length]

			vec = torch.sum(head, dim=0)
			vec = vec[1:-1] #remove first and last spaces for cls and sep tokens

			vec = softmax(vec)
			vec = vec.detach()

			att += vec

		att /= 12
		att = att.tolist()

		new_att = []
		new_tokens = []

		for j in reversed(range(len(tokens))):
			if "##" in tokens[j]:
				tokens[j-1] += tokens[j][2:]
				att[j-1] += att[j]
			else:
				new_att.insert(0,att[j])
				new_tokens.insert(0,tokens[j])

		for label in predicted_labels:
			counts[label].update(new_tokens)
			for (token, att) in zip(new_tokens, new_att):
				word_scores[label][token] += att

	avg_scores = word_scores.copy()

	max_words = 100

	freq = {}

	for emotion in emotions:
		for word in counts[emotion]:
			#avg_scores[emotion][word] /= counts[emotion][word]
			avg_scores[emotion][word] = int(500 * avg_scores[emotion][word]) #convert average attention to int because wordcloud needs frequencies

		freq[emotion] = KmaxelementsDict(avg_scores[emotion], max_words)
		print(f"{emotion}:")
		print(freq[emotion])
		print("")

		wc = WordCloud(background_color="white", width=800, height=400, max_words=max_words, collocations=False)
		wc.generate_from_frequencies(freq[emotion])

		plt.axis("off")
		plt.imshow(wc, interpolation="bilinear")
		plt.savefig(f"plots/attention/{emotion}.png", format="png")

if __name__ == '__main__':
	main()