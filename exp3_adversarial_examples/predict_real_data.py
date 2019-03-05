import sys
import torch
import torchvision
import numpy as np
from PIL import Image
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="Path to model")
args = parser.parse_args()

# return normalized dataset divided into two sets
def prepare_db():
	eval_dataset = torchvision.datasets.MNIST('./data/mnist', train=False, download=True,
											   transform=torchvision.transforms.Compose([
												   torchvision.transforms.ToTensor()
											   ]))
	return {'eval':eval_dataset}


cuda = 1
gpu_id = 0
device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() and cuda == 1 else "cpu")
print("Device:", device)

# model on which to test
if(torch.cuda.is_available() == False):
	model = torch.load("saved_models/"+args.model, map_location='cpu').to(device)
else:
	model = torch.load("saved_models/"+args.model).to(device)

model.eval()

scores = [float(0)] * 10
for i in range(len(scores)):
	scores[i] = [float(0)]*10
scores = np.array(scores)

counts = [0] * 10


# not the best way to do
# more efficient would be to pass all images at once
db = prepare_db()
for (data, target) in db['eval']:
	data = data.view((1,1,28,28)).to(device)
	target = target.item()
	output = torch.sigmoid(model(data))
	counts[target] += 1
	scores[target] += output.view(10).detach().cpu().numpy()

for i in range(10):
	scores[i] /= counts[i]
	print(i, max(enumerate(scores[i]), key=lambda x: x[1])[0])