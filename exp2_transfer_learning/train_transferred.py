import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torchvision # for data
from model import SimpleConvNet
import numpy as np
import PIL
from database_util import *

# hyper-parameters
batch_size = 100
epochs = 150
report_every = 10
conv = [3,16,32,64]
fc = [256]
n_classes = 5
dropout_rate = 0.5
size = 32 # update

# GPU related info
cuda = 1
device = torch.device("cuda" if torch.cuda.is_available() and cuda == 1 else "cpu") # default gpu
print("Device:", device)

if(torch.cuda.is_available() == False):
	model = torch.load("convnet_right.pt", map_location='cpu').to(device)
else:
	model = torch.load("convnet_right.pt").to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adagrad(model.parameters(), lr=0.005)

def freeze_layers(model):

	for conv_layer in model.conv_layers[:-3]:
		for params in conv_layer.parameters():
			params.requires_grad = False
		print('c')

	for fc_layer in model.fc_layers[:-1]:
		for params in fc_layer.parameters():
			params.requires_grad = False
		print('fc')

	for batchnorm_layer in model.batchnorm_layers[:-4]:
		for params in batchnorm_layer.parameters():
			params.requires_grad = False
		print('bn')

def train(model, optim, db):

	for epoch in range(1, epochs+1):

		train_loader = torch.utils.data.DataLoader(db['train'],batch_size=batch_size, shuffle=True)

		# Update (Train)
		model.train()
		for batch_idx, (data, target) in enumerate(train_loader):

			data, target = Variable(data.to(device)), Variable((target.float()).to(device))
			optimizer.zero_grad()
			output = model(data)
			loss = criterion(output,target.long())
			# print(output.size())
			pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
			correct = pred.eq(target.data.view_as(pred).long()).cpu().sum()
			loss.backward()
			optimizer.step()

			if batch_idx % report_every == 0:
				print('Train Epoch: {} [{}/{} ({:.0f}%)]  Loss: {:.6f}, Accuracy: {}/{} ({:.6f})'.format(
					epoch, batch_idx * len(data), len(train_loader.dataset),
					100.0 * batch_idx / len(train_loader), loss.item(), correct, len(data), float(correct)/float(len(data))))


		# Evaluate
		model.eval()
		eval_loss = float(0)
		correct = 0
		batch_count = 0
		eval_loader = torch.utils.data.DataLoader(db['eval'], batch_size=batch_size, shuffle=True)
		with torch.no_grad():
			for data, target in eval_loader:

				data, target = Variable(data.to(device)), Variable((target.float()).to(device))
				output = model(data)
				eval_loss += criterion(output, target.long()).item() # sum up batch loss
				pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
				correct += pred.eq(target.data.view_as(pred).long()).cpu().sum()
				batch_count += 1


		eval_loss /= batch_count
		accuracy = float(correct) / len(eval_loader.dataset)

		print('Eval set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f})\n'.format(
			eval_loss, correct, len(eval_loader.dataset),
			accuracy))		

def main():
	db = prepare_db()
	db_l, db_r = split_database(db,0.10)
	print("Database split done!")
	freeze_layers(model)
	train(model, optimizer, db_l)


if __name__ == '__main__':
	main()