import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torchvision # for data
import model
import numpy as np

# hyper-parameters
batch_size = 128
epochs = 50
report_every = 64
fc = [256]
size_output = 10 # ceil(log(10))
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
												   torchvision.transforms.ToTensor(),
												   torchvision.transforms.Normalize((0.1307,), (0.3081,))
											   ]))

	eval_dataset = torchvision.datasets.MNIST('./data/mnist', train=False, download=True,
											   transform=torchvision.transforms.Compose([
												   torchvision.transforms.ToTensor(),
												   torchvision.transforms.Normalize((0.1307,), (0.3081,))
											   ]))
	return {'train':train_dataset,'eval':eval_dataset}

model = model.fc_net(size, fc, size_output).to(device)
criterion = nn.MSELoss().to(device)
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adagrad(model.parameters(), lr=0.01)

def train(model, optim, db):

	for epoch in range(1, epochs+1):

		train_loader = torch.utils.data.DataLoader(db['train'],batch_size=batch_size, shuffle=True)

		# Update (Train)
		model.train()
		for batch_idx, (data, target) in enumerate(train_loader):

			target = target.numpy()
			tgx = np.zeros((len(target), size_output))
			idx = [(i, target[i]) for i in range(len(target))]
			for i in idx:
				for j in get_indices(i):
					tgx[j]=1.0
			target = torch.tensor(tgx)

			data, target = Variable(data.to(device)), Variable((target.float()).to(device))
			optimizer.zero_grad()
			output = model(data)
			loss = criterion(output,target)
			loss.backward()
			optimizer.step()

			if batch_idx % report_every == 0:
				print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
					epoch, batch_idx * len(data), len(train_loader.dataset),
					100.0 * batch_idx / len(train_loader), loss.item()))


		# Evaluate
		model.eval()
		eval_loss = float(0)
		correct = 0
		batch_count = 0
		eval_loader = torch.utils.data.DataLoader(db['eval'], batch_size=batch_size, shuffle=True)
		with torch.no_grad():
			for data, target in eval_loader:

				target = target.numpy()
				tgx = np.zeros((len(target), size_output))
				idx = [(i, target[i]) for i in range(len(target))]
				for i in idx:
					for j in get_indices(i):
						tgx[j]=1.0
				target = torch.tensor(tgx)

				data, target = Variable(data.to(device)), Variable((target.float()).to(device))
				output = model(data)
				eval_loss += criterion(output, target).item() # sum up batch loss

				# change here to convert from binary
				pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
				target = target.data.max(1, keepdim=True)[1]

				correct += pred.eq(target.data.view_as(pred)).cpu().sum()
				batch_count += 1

		eval_loss /= batch_count
		accuracy = float(correct) / len(eval_loader.dataset)

		print('\nEval set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f})\n'.format(
			eval_loss, correct, len(eval_loader.dataset),
			accuracy))					

def main():
	db = prepare_db()
	# print(db['train'][0][1])
	train(model, optim, db)


if __name__ == '__main__':
	main()