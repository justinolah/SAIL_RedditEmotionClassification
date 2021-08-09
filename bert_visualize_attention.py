"""
@inproceedings{yang2018ncrf,  
 title={NCRF++: An Open-source Neural Sequence Labeling Toolkit},  
 author={Yang, Jie and Zhang, Yue},  
 booktitle={Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics},
 Url = {http://aclweb.org/anthology/P18-4013},
 year={2018}  
}
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from helpers import *

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

class BERT_Model(nn.Module):
	def __init__(self, bert, numEmotions):
		super(BERT_Model, self).__init__()
		self.bert = bert 
		self.fc = nn.Linear(768, numEmotions)

	def forward(self, sent_id, mask):
		_, cls_hs, attentions = self.bert(sent_id, attention_mask=mask, output_attentions=True, return_dict=False)
		out = self.fc(cls_hs)
		return out, attentions

def generate(text_list, attention_list, color='red', rescale_value = True):
	assert(len(text_list) == len(attention_list))
	if rescale_value:
		attention_list = rescale(attention_list)
	word_num = len(text_list)
	text_list = clean_word(text_list)
	
	string = r'''\begin{CJK*}{UTF8}{gbsn} {\setlength{\fboxsep}{0pt}\colorbox{white!0}{\parbox{0.9\textwidth}{'''+"\n"
	for idx in range(word_num):
		string += "\\colorbox{%s!%s}{"%(color, attention_list[idx])+"\\strut " + text_list[idx]+"} "
	string += "\n}}}\n\\end{CJK*}\n\n"

	return string



def rescale(input_list):
	the_array = np.asarray(input_list)
	the_max = np.max(the_array)
	the_min = np.min(the_array)
	rescale = (the_array - the_min)/(the_max-the_min)*100
	return rescale.tolist()


def clean_word(word_list):
	new_word_list = []
	for word in word_list:
		for latex_sensitive in ["\\", "%", "&", "^", "#", "_",  "{", "}"]:
			if latex_sensitive in word:
				word = word.replace(latex_sensitive, '\\'+latex_sensitive)
		new_word_list.append(word)
	return new_word_list


def main():
	string = ""

	if torch.cuda.is_available():
		print("Using cuda...")
		device = torch.device("cuda")
	else:
		print("Using cpu...")
		device = torch.device("cpu")

	emotions = getEmotions()
	emotions.remove("neutral")

	data = getTestSet()

	max_length = 128

	entries = [
		"edmxdm7",
		"ed8llym",
		"eey7nln",
		"effn3uq",
		"edjd969",
		"eefbq4h",
		"ee66rq1",
		"eeyda6m",
		"efba01h",
		"edclugt",
		"edf9nth",
		"eebfurv",
		"eea5j35",
		"ef6nz0w",
		"eecefqy",
		"edqdi7q",
		"ef1e1yw",
	]

	data = data[data.id.isin(entries)]

	texts = [
		"Germany is the first country in Europe I've been to! Planning on Ireland next, and maybe Croatia(hoping for Germany again, though :) )",
		"What's that like? Like what's the thought process? I dunno. I know what's a weird question..i just can't imagine",
		"> one of the better diss tracks out there Lol okay",
		"[NAME]... I'm sorry. This is just wrong. I, can't.",
		"I do feel sorry for the squirrel, but I wouldn't say the dog is a jerk for acting on natural instinct.",
		"You're in luck!",
		":) thank you!",
		"That made me smile. Thank you!! And yes, definitely replacing her on my reference list lol. ",
	]

	tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
	bert = BertModel.from_pretrained('bert-base-uncased')
	model = BERT_Model(bert, len(emotions))
	model = model.to(device)

	checkpoint = torch.load("bert.pt")
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

	for i in range(len(data)):
		seq = torch.tensor(seqs[i])
		mask = torch.tensor(masks[i])

		tokens = tokenizer.convert_ids_to_tokens(seq)
		tokens = [token for token in tokens if token not in ['[PAD]','[CLS]','[SEP]']]
		#sentence = " ".join(tokens)

		length = len(tokens)

		softmax = nn.Softmax(dim=0)

		output, attention = model(seq.unsqueeze_(0).to(device), mask.unsqueeze_(0).to(device))
		#attention.size() -> 1,12,128,128

		length = torch.sum(mask)
		output = output.cpu()

		output = (output > 0.5).int()

		actual_labels = [emotions[int(label)] for label in labels[i].split(',') if int(label) < len(emotions)]
		predicted_labels = [emotions[index] for index, val in enumerate(output[0].tolist()) if val == 1]

		string += r'''\textbf{Actual Labels:} ''' + ", ".join(actual_labels) + r'''\\''' + "\n"
		string += r'''\textbf{Predicted Labels:} ''' + ", ".join(predicted_labels) + r'''\\''' + "\n"
		
		#layer = attention[-1]
		for layer in attention:
			layer = layer.cpu()
			att = torch.zeros(len(tokens))

			for i in range(12):
				head_i = layer[0,i,:length,:length]

				vec = torch.sum(head_i, dim=0)
				vec = vec[1:-1] #remove first and last spaces for cls and sep tokens

				vec = softmax(vec)
				vec = vec.detach()

				att += vec

			att /= 12
			string += generate(tokens, att, 'red')
		
		string += "\\clearpage\n"

	with open("attention.tex",'w') as f:
		f.write(r'''\documentclass{article}
			\usepackage[utf8]{inputenc}
			\usepackage{color}
			\usepackage{tcolorbox}
			\usepackage[document]{ragged2e}
			\usepackage{CJK}
			\usepackage{adjustbox}
			\usepackage{textpos}
			\tcbset{width=0.9\textwidth,boxrule=0pt,colback=red,arc=0pt,auto outer arc,left=0pt,right=0pt,boxsep=5pt}
			\begin{document}''')

		f.write(string)

		f.write(r'''\end{document}''')
		f.write("\n")

if __name__ == '__main__':
	main()