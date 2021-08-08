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
	string += "\n}}}\n\\end{CJK*}\n"

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
	print(emotions)

	max_length = 128

	texts = ["As an anesthesia resident this made me blow air out my nose at an accelerated rate for several seconds. Take your damn upvote you bastard."]

	tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
	bert = BertModel.from_pretrained('bert-base-uncased')
	model = BERT_Model(bert, len(emotions))
	model = model.to(device)

	checkpoint = torch.load("bert.pt")
	model.load_state_dict(checkpoint['model_state_dict'])
	model.eval()

	tokens = tokenizer.batch_encode_plus(
		texts,
		max_length = max_length,
		padding='max_length',
		truncation=True
	)

	seq = torch.tensor(tokens['input_ids'])
	mask = torch.tensor(tokens['attention_mask'])

	tokens = tokenizer.convert_ids_to_tokens(seq[0])
	tokens = [token for token in tokens if token != '[PAD]']
	sentence = " ".join(tokens)
	print(sentence)

	length = len(tokens)

	print(seq)
	print(mask)

	softmax = nn.Softmax(dim=0)

	output, attention = model(seq.to(device), mask.to(device))

	length = torch.sum(mask)
	attention = attention[-1].cpu()
	output = output.cpu()

	output = (output > 0.5).int()

	print(output)
	print(attention)
	print(attention.size())

	for i in range(12):
		head_i = attention[0,i,:length,:length]

		vec = torch.sum(head_i, dim=0)
		vec = softmax(vec)
		vec = vec.detach()

		string += generate(tokens, vec, 'red')

	with open("sample.tex",'w') as f:
		f.write(r'''\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{color}
\usepackage{tcolorbox}
\usepackage{CJK}
\usepackage{adjustbox}
\tcbset{width=0.9\textwidth,boxrule=0pt,colback=red,arc=0pt,auto outer arc,left=0pt,right=0pt,boxsep=5pt}
\begin{document}''')

		f.write(string)

		f.write(r'''\end{document}''')

if __name__ == '__main__':
	main()