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
size = 28*28

# GPU related info
cuda = 1
gpu_id = 0
# device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() and cuda == 1 else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() and cuda == 1 else "cpu") # default gpu
print("Device:", device)

# return the binary indices to be set to 1
# n : [0,9] => max length
def get_indices(n):
	ret = []
	curr = 0
	while n != 0:
		if (n%2 == 1):
			ret += [curr]
		n = n//2
		curr += 1
	return ret

def do_thresh(output):
	output[output >= 0.5] = 1
	output[output < 0.5] = 0
	return output.long()

def get_num(output):
	ret = []
	exp = torch.tensor([1,2,4,8])
	for i in range(output.size()[0]):
		ret += [[torch.sum(exp.long() * output[i].long().cpu())]]
	return torch.tensor(ret)


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

model_10 = model.fc_net(size, fc, 10).to(device)
model_4 = model.fc_net(10,[],4).to(device)
criterion = nn.MSELoss().to(device)
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer_10 = optim.Adagrad(model_10.parameters(), lr=0.01)
optimizer_4 = optim.Adagrad(model_4.parameters(), lr=0.01)

def train(model_10, model_4, optimizer_10, optimizer_4, db):

	for epoch in range(1, epochs+1):

		train_loader = torch.utils.data.DataLoader(db['train'],batch_size=batch_size, shuffle=True)

		# Update (Train)
		model_10.train()
		model_4.train()
		for batch_idx, (data, target) in enumerate(train_loader):

			target_4 = target.numpy()
			tgx = np.zeros((len(target_4), 4))
			idx = [(i, target_4[i]) for i in range(len(target_4))]
			for i in idx:
				for j in get_indices(i[1]):
					tgx[i[0],j]=1.0
			target_4 = torch.tensor(tgx)

			target_10 = target.numpy()
			tgx = np.zeros((len(target_10), 10))
			idx = [(i, target_10[i]) for i in range(len(target_10))]
			for i in idx:
				tgx[i]=1.0
			target_10 = torch.tensor(tgx)

			data, target_10 = Variable(data.to(device)), Variable((target_10.float()).to(device))
			optimizer_10.zero_grad()
			output_10 = model_10(data)
			loss_10 = criterion(output_10,target_10)
			loss_10.backward()
			optimizer_10.step()

			data, target_4 = Variable(output_10.to(device)), Variable((target_4.float()).to(device))
			optimizer_4.zero_grad()
			output_4 = model_4(data)
			loss_4 = criterion(output_4,target_4)
			loss_4.backward()
			optimizer_4.step()

			if batch_idx % report_every == 0:
				print('Train Epoch [10]: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
					epoch, batch_idx * len(data), len(train_loader.dataset),
					100.0 * batch_idx / len(train_loader), loss_10.item()))
				print('Train Epoch [04]: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
					epoch, batch_idx * len(data), len(train_loader.dataset),
					100.0 * batch_idx / len(train_loader), loss_4.item()))


		# Evaluate
		model_10.eval()
		model_4.eval()
		eval_loss_10 = float(0)
		eval_loss_4 = float(0)
		correct_10 = 0
		correct_4 = 0
		batch_count = 0
		eval_loader = torch.utils.data.DataLoader(db['eval'], batch_size=batch_size, shuffle=True)
		with torch.no_grad():
			for data, target in eval_loader:

				target_4 = target.numpy()
				tgx = np.zeros((len(target_4), 4))
				idx = [(i, target_4[i]) for i in range(len(target_4))]
				for i in idx:
					for j in get_indices(i[1]):
						tgx[i[0],j]=1.0
				target_4 = torch.tensor(tgx)

				target_10 = target.numpy()
				tgx = np.zeros((len(target_10), 10))
				idx = [(i, target_10[i]) for i in range(len(target_10))]
				for i in idx:
					tgx[i]=1.0
				target_10 = torch.tensor(tgx)

				data, target_10 = Variable(data.to(device)), Variable((target_10.float()).to(device))
				output_10 = model_10(data)
				eval_loss_10 += criterion(output_10, target_10).item() # sum up batch loss
				pred = output_10.data.max(1, keepdim=True)[1] # get the index of the max log-probability
				target_10 = target_10.data.max(1, keepdim=True)[1]
				correct_10 += pred.eq(target_10.data.view_as(pred)).cpu().sum()

				data, target_4 = Variable(output_10.to(device)), Variable((target_4.float()).to(device))
				output_4 = model_4(data)
				eval_loss_4 += criterion(output_4, target_4).item() # sum up batch loss
				pred = get_num(do_thresh(output_4.data))
				target_4 = get_num(target_4.data)
				correct_4 += pred.eq(target_4).cpu().sum()

				batch_count += 1

		eval_loss_10 /= batch_count
		eval_loss_4 /= batch_count

		accuracy_10 = float(correct_10) / len(eval_loader.dataset)
		accuracy_4 = float(correct_4) / len(eval_loader.dataset)

		with open('results/one_hot_joined.dat', 'a+') as file:
			file.write(str(accuracy_10)+"\n")

		with open('results/binary_encoding_joined.dat', 'a+') as file:
			file.write(str(accuracy_4)+"\n")

		print('\nEval set [10]: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f})\n'.format(
			eval_loss_10, correct_10, len(eval_loader.dataset),
			accuracy_10))	
		print('Eval set [04]: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f})\n'.format(
			eval_loss_4, correct_4, len(eval_loader.dataset),
			accuracy_4))

	with torch.no_grad():
		print(model_4(torch.eye(10)))

def main():
	db = prepare_db()
	# print(db['train'][0][1])
	train(model_10, model_4, optimizer_10, optimizer_4, db)

if __name__ == '__main__':
	main()