import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torchvision # for data
import model
import numpy as np
import sys

# hyper-parameters
batch_size = 500 # 60000/500 = 120 batches
epochs = 10000
report_every = epochs/100 # report 120/30 = 4 times per epoch
learning_rate = 10.0
num_sched_lr = 3
num_trials = 5

size_output = 10
size_input = 28

# GPU related info
cuda = 1
gpu_id = 0
# device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() and cuda == 1 else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() and cuda == 1 else "cpu") # default gpu
print("Device:", device)

# return normalized dataset divided into two sets
def prepare_db():
	eval_dataset = torchvision.datasets.MNIST('./data/mnist', train=False, download=True,
											   transform=torchvision.transforms.Compose([
												   torchvision.transforms.ToTensor()
											   ]))
	return {'eval':eval_dataset}

if(torch.cuda.is_available() == False):
	model = torch.load("saved_models/model_convnet.pt", map_location='cpu').to(device)
else:
	model = torch.load("saved_models/model_convnet.pt").to(device)

def generate_non_targeted(model, lr, label, n_try):

	if n_try == 0:
		return

	lr_orig = lr
	criterion = nn.MSELoss().to(device)

	data = torch.randn((1,1,size_input,size_input))*0.01
	target = torch.zeros((1,size_output))
	target[0][label] = 1

	data, target = data.to(device), target.to(device)

	for i in range(epochs):

		data = Variable(data.data, requires_grad=True)
		output = torch.sigmoid(model(data))
		pred = output.max(1, keepdim=True)[1]

		if i % report_every == 0 or i == epochs-1:
			print(output.data.cpu().numpy()[0])

		if (i + 1) % ( epochs / (num_sched_lr + 1) ) == 0 and label == pred:
			lr /= 10.0
			print("LR sched lap")

		loss = criterion(output, target)
		model.zero_grad()
		loss.backward()

		data = data - lr * data.grad.data

	if model(data).max(1, keepdim=True)[1] == label:
		print("Success. " + str(label))
	else:
		print("Fail. Retry. " + str(label))
		return generate_non_targeted(model, lr_orig, label, n_try-1)

	data = data.view(1,size_input,size_input).detach().cpu()
	print(data)
	trans = torchvision.transforms.ToPILImage()
	data = trans(data)
	data.save("nti_"+str(label)+".jpg")	

def main():
	db = prepare_db()
	label = int(sys.argv[1])
	generate_non_targeted(model, learning_rate, label, num_trials)


if __name__ == '__main__':
	main()