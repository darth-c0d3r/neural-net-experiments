import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torchvision # for data
from model import SimpleConvNet
import numpy as np
import PIL

def display_sample_image(db, set, index):
	trans1 = torchvision.transforms.ToPILImage()
	img = output_image = trans1(db[set][index][0])
	img.show()

# hyper-parameters
batch_size = 1000
epochs = 50
report_every = 10
conv = [3,16,32,64]
fc = [256]
n_classes = 10
dropout_rate = 0.2
size = 32 # update

# GPU related info
cuda = 1
device = torch.device("cuda" if torch.cuda.is_available() and cuda == 1 else "cpu") # default gpu
print("Device:", device)

# return normalized dataset divided into two sets
def prepare_db():
	train_dataset = torchvision.datasets.CIFAR10('./data/cifar10', train=True, download=True,
											   transform=torchvision.transforms.Compose([
												   torchvision.transforms.ToTensor(),
												   # torchvision.transforms.Normalize((0.1307,), (0.3081,))
											   ]))

	eval_dataset = torchvision.datasets.CIFAR10('./data/cifar10', train=False, download=True,
											   transform=torchvision.transforms.Compose([
												   torchvision.transforms.ToTensor(),
												   # torchvision.transforms.Normalize((0.1307,), (0.3081,))
											   ]))
	return {'train':train_dataset,'eval':eval_dataset}

model = SimpleConvNet(size, conv, fc, n_classes, dropout_rate).to(device)
model.my_init(0.0, 0.01)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adagrad(model.parameters(), lr=0.01)

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

		# with open('results/one_hot.dat', 'a+') as file:
		# 	file.write(str(accuracy)+"\n")
		print('Eval set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f})\n'.format(
			eval_loss, correct, len(eval_loader.dataset),
			accuracy))					

def main():
	db = prepare_db()

	# display_sample_image(db, 'train', 0)
	# print(len(db['train']))

	train(model, optim, db)


if __name__ == '__main__':
	main()