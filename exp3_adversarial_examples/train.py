import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torchvision # for data
import model
import numpy as np

# hyper-parameters
batch_size = 500 # 60000/500 = 120 batches
epochs = 50
report_every = 30 # report 120/30 = 4 times per epoch

# for fc net
# conv = [1]
# fc = [128,64,32]

# for conv net 
conv = [1,16,32,64]
fc = []

size_output = 10
size = 28

# GPU related info
cuda = 1
gpu_id = 0
# device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() and cuda == 1 else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() and cuda == 1 else "cpu") # default gpu
print("Device:", device)

# return normalized dataset divided into two sets
def prepare_db():
	train_dataset = torchvision.datasets.MNIST('./data/mnist', train=True, download=True,
											   transform=torchvision.transforms.Compose([
												   torchvision.transforms.ToTensor()
											   ]))

	eval_dataset = torchvision.datasets.MNIST('./data/mnist', train=False, download=True,
											   transform=torchvision.transforms.Compose([
												   torchvision.transforms.ToTensor()
											   ]))
	return {'train':train_dataset,'eval':eval_dataset}

model = model.classifier(size, conv, fc, size_output).to(device)
criterion = nn.CrossEntropyLoss().to(device)
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adagrad(model.parameters(), lr=0.01)

def train(model, optim, db):

	for epoch in range(1, epochs+1):

		train_loader = torch.utils.data.DataLoader(db['train'],batch_size=batch_size, shuffle=True)

		# Update (Train)
		model.train()
		for batch_idx, (data, target) in enumerate(train_loader):

			data, target = Variable(data.to(device)), Variable((target).to(device))
			optimizer.zero_grad()
			output = model(data)
			loss = criterion(output,target)
			pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
			correct = pred.eq(target.data.view_as(pred)).cpu().sum()
			loss.backward()
			optimizer.step()

			if batch_idx % report_every == 0:
				print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}, Accuracy: {}/{} ({:.6f})'.format(
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

				data, target = Variable(data.to(device)), Variable((target).to(device))
				output = model(data)
				eval_loss += criterion(output, target).item() # sum up batch loss
				pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
				correct += pred.eq(target.data.view_as(pred)).cpu().sum()
				batch_count += 1

		eval_loss /= batch_count
		accuracy = float(correct) / len(eval_loader.dataset)

		print('Eval set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f})\n'.format(
			eval_loss, correct, len(eval_loader.dataset),
			accuracy))					

	torch.save(model, 'saved_models/model_convnet.pt')

def main():
	db = prepare_db()
	train(model, optim, db)


if __name__ == '__main__':
	main()