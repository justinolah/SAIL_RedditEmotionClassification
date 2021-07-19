from pytorch_model import *
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class UniLSTM(nn.Module):
	def __init__(self, embedding, embedding_dim, hidden_dim, output_dim, device, n_layers=1, dropout=0):
		super(UniLSTM, self).__init__()
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim
		self.embedding = embedding
		self.n_layers = n_layers
		self.device = device

		self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=False)
		self.lin = nn.Linear(hidden_dim, output_dim)

		self.dropout = nn.Dropout(dropout)

	def forward(self, x, lengths, hidden):
		batch_size = x.size(0)
		embeds = self.embedding[x]
		embeds = pack_padded_sequence(embeds, lengths, enforce_sorted=False)
	
		lstm_out, hidden = self.lstm(embeds, hidden)
		lstm_out, lengths = pad_packed_sequence(lstm_out)

		out = torch.zeros_like(lstm_out[0])

		for i, length in enumerate(lengths):
			out[i] = lstm_out[length-1,i]

		#lstm_out = lstm_out[-1]
		
		out = self.dropout(out)
		out = self.lin(out)

		return out, hidden

	def initHidden(self, batch_size):
		weight = next(self.parameters()).data
		hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device),
					  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device))
		return hidden

class BiLSTM(nn.Module):
	def __init__(self, embedding, embedding_dim, hidden_dim, output_dim, device, n_layers=1, dropout=0):
		super(BiLSTM, self).__init__()
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim
		self.embedding = embedding
		self.n_layers = n_layers
		self.device = device

		self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, bidirectional=True, batch_first=False)
		self.lin = nn.Linear(hidden_dim * 2, output_dim)

		self.dropout = nn.Dropout(dropout)

	def forward(self, x, lengths, hidden):
		batch_size = x.size(0)
		embeds = self.embedding[x]
		embeds = pack_padded_sequence(embeds, lengths, enforce_sorted=False)
	
		lstm_out, hidden = self.lstm(embeds, hidden)
		lstm_out, lengths = pad_packed_sequence(lstm_out)

		out = torch.zeros_like(lstm_out[0])

		for i, length in enumerate(lengths):
			out[i] = lstm_out[length-1,i]
		
		out = self.dropout(out)
		out = self.lin(out)

		return out, hidden

	def initHidden(self, batch_size):
		weight = next(self.parameters()).data
		hidden = (weight.new(self.n_layers * 2, batch_size, self.hidden_dim).zero_().to(self.device),
					  weight.new(self.n_layers *2, batch_size, self.hidden_dim).zero_().to(self.device))
		return hidden

def trainNN(model, trainloader, batch_size, devData, devLengths, devLabels, optimizer, loss_fn, threshold, device):
	counter = 0
	train_running_loss = 0.0
	sigmoid = nn.Sigmoid()

	model.train()

	allTargets = []
	allPredictions = []

	for i, batch in enumerate(trainloader):
		counter += 1
		h = model.initHidden(len(batch))
		h = tuple([e.data for e in h])
		inputs, lengths = batch.text
		labels = batch.labels.float()
		allTargets.append(labels.detach())

		inputs, labels = inputs.to(device), labels.to(device)
		optimizer.zero_grad()
		outputs, h = model(inputs, lengths, h)
		outputs = outputs.squeeze()
		allPredictions.append((outputs.cpu() > threshold).int().detach())

		loss = loss_fn(outputs, labels)
		loss.backward()
		optimizer.step()

		train_running_loss += loss.item()

	trainLoss = train_running_loss / counter

	model.eval()
	val_h = model.initHidden(len(devData[0]))
	val_h = tuple([e.data for e in val_h])
	devOutput, val_h = model(devData.to(device), devLengths, val_h)
	devOutput = devOutput.squeeze()
	devLoss = loss_fn(devOutput, devLabels.to(device)).item()
	devOutput = sigmoid(devOutput)

	allPredictions = np.concatenate(allPredictions)
	allTargets = np.concatenate(allTargets)

	f1_train = f1_score(allTargets, allPredictions, average='macro')
	f1_dev = f1_score(devLabels.detach(), (devOutput.cpu() > threshold).int().detach(), average='macro')

	return trainLoss, devLoss, f1_train, f1_dev

def main():
	emotions = getEmotions()
	emotions.remove("neutral")

	if torch.cuda.is_available():
		print("Using cuda...")
		device = torch.device("cuda")
	else:
		print("Using cpu...")
		device = torch.device("cpu")

	#parameters
	maxSentenceLength = 31
	embedding_dim = 200
	epochs = 7
	hidden_dim = 250
	droput = 0.5
	weight_decay = 0.01
	output_dim = len(emotions)
	batch_size = 256
	threshold = 0.5
	lr = 1e-2
	filename = "rnn"

	print("Loading glove..\n")
	rawEmbedding = GloVe(name='twitter.27B', dim=embedding_dim)
	#rawEmbedding = FastText(language='en')

	#get data
	train = getTrainSet()
	test = getTestSet()
	dev = getValSet()

	#clean text
	train.text = train.text.apply(cleanTextForEmbedding)
	test.text= test.text.apply(cleanTextForEmbedding)
	dev.text = dev.text.apply(cleanTextForEmbedding)

	#remove empty sentences
	train = train[train.text.str.len() > 0]
	test = test[test.text.str.len() > 0]
	dev = dev[dev.text.str.len() > 0]

	text_field = Field(
	    sequential=True,
	    tokenize='basic_english', 
	    fix_length=maxSentenceLength,
	    lower=True,
	    include_lengths=True,
	)

	#build vocab
	preprocessed_text = train.text.apply(
    	lambda x: text_field.preprocess(x)
	)

	#print("Max sentence length:", np.array([len(row) for row in preprocessed_text.to_list()]).max()) #gets max sentence length

	text_field.build_vocab(
    	preprocessed_text, 
    	vectors=rawEmbedding,
	)

	vocab = text_field.vocab

	#Make datasets
	dataset = makeDataset(train, emotions, text_field)
	testset = makeDataset(test, emotions, text_field)
	devset = makeDataset(dev, emotions, text_field)

	#pytorch model
	print("Training NN...")
	torch.manual_seed(42)
	rnn = BiLSTM(vocab.vectors.to(device), embedding_dim, hidden_dim, output_dim, device, n_layers=1, dropout=0)
	rnn.to(device)

	optimizer = torch.optim.Adam(rnn.parameters(), lr=lr, weight_decay=weight_decay)

	loss_fn= nn.BCEWithLogitsLoss(pos_weight= 8*torch.ones(len(emotions)).to(device))
	#loss_fn = wlsep

	#trainloader = BucketIterator(dataset, batch_size, sort_key=lambda x: len(x.text), repeat=True, shuffle=True, sort_within_batch=True) #runs too slow for some reason
	trainloader = Iterator(dataset, batch_size)

	#train model
	rnn.train()
	trainLoss = []
	devLoss = []
	trainF1 = []
	devF1 = []
	devbatch = next(iter(Iterator(devset, len(devset))))
	devData, devLengths = devbatch.text
	devLabels = devbatch.labels.float()

	for epoch in range(epochs):
		print("Epoch:", epoch)
		epoch_loss, dev_loss, f1_train, f1_dev = trainNN(rnn, trainloader, batch_size, devData, devLengths, devLabels, optimizer, loss_fn, threshold, device)
		trainLoss.append(epoch_loss)
		devLoss.append(dev_loss)
		trainF1.append(f1_train)
		devF1.append(f1_dev)
		print("Training Loss:", epoch_loss)
		print("Dev Loss:", dev_loss)
		print("Training Macro F1:", f1_train)
		print("Dev Macro F1:", f1_dev, "\n")

	print("Training complete\n")

	bestEpochDevF1 = np.argmax(np.array(devF1))
	print("Best Dev F1:", devF1[bestEpochDevF1], "at epoch", bestEpochDevF1, "\n")

	#learning curve 
	fig, (ax1, ax2) = plt.subplots(2, figsize=(11, 9))
	ax1.plot(trainLoss, color='b', label='Training loss')
	ax1.plot(devLoss, color='r', label='Dev loss')
	fig.suptitle(f"Hidden:{hidden_dim}, LR:{lr}, BS:{batch_size}, Embedding Size: {embedding_dim}")
	ax1.set(xlabel='Epochs', ylabel="Loss")
	ax1.legend()
	ax2.plot(trainF1, color='b', label='Training Macro F1')
	ax2.plot(devF1, color='r', label='Dev Macro F1')
	ax2.set(xlabel='Epochs', ylabel="Macro F1")
	ax2.legend()
	fig.savefig('plots/learningcurve_rnn.png')

	#Testing metrics
	rnn.eval()
	sigmoid = nn.Sigmoid()
	testbatch = next(iter(Iterator(testset, len(devset))))
	data, lengths = testbatch.text
	labels = testbatch.labels.float()
	h = rnn.initHidden(len(devset))
	h = tuple([each.data for each in h])

	outputs, h = rnn(data.to(device), lengths, h)
	outputs = sigmoid(outputs.squeeze())
	prediction = (outputs > threshold).int().cpu()

	accuracy = accuracy_score(labels, prediction)
	print("Subset Accuracy:", accuracy)
	print(classification_report(labels, prediction, target_names=emotions, zero_division=0, output_dict=False))
	print("Best Dev F1:", devF1[bestEpochDevF1], "at epoch", bestEpochDevF1, "\n")
	report = classification_report(labels, prediction, target_names=emotions, zero_division=0, output_dict=True)

	#export resuls to csv
	micro = list(report['micro avg'].values())
	micro.pop() 
	macro = list(report['macro avg'].values())
	macro.pop()
	scores = [accuracy, *micro, *macro]
	results = pd.DataFrame(data=[scores], columns=['accuracy', 'micro_precision', 'micro_recall', 'micro_f1', 'macro_precision', 'macro_recall', 'macro_f1'])
	results.to_csv("tables/" + filename + "_results.csv")

	#confusion matrix
	multilabel_confusion_matrix(np.array(labels), outputs.cpu().detach().numpy(), emotions, top_x=3, filename=filename)

if __name__ == "__main__":
	main()