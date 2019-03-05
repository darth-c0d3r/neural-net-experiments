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
root_dir_input = "adversarial_outputs_conv/targeted/"
files = os.listdir(root_dir_input)
files.sort()

inp_out_pairs = torch.zeros((len(files), 2)).to(device)

with torch.no_grad():
	i = 0
	for file in files:
		inp = Image.open(root_dir_input + file)
		trans = torchvision.transforms.ToTensor()
		tensor = trans(inp)
		tensor = tensor.view(1,tensor.shape[0],tensor.shape[1],tensor.shape[2]).to(device)
		# output = torch.softmax(model(tensor), dim=1)
		output = torch.sigmoid(model(tensor))
		inp_out_pairs[i][0], inp_out_pairs[i][1] = int(file[-5:-4]), output.data.cpu()[0].max(0, keepdim=True)[1] 
		i += 1
		#int(max(enumerate(output.data.cpu().numpy()[0]), key=lambda x: x[1])[0])

all_pairs = len(files)%10 == 0
if all_pairs:
	acc_array = (inp_out_pairs[:,0] == inp_out_pairs[:,1]).reshape((len(inp_out_pairs)//10), 10)
	for i in range(acc_array.shape[0]):
		print(int(torch.sum(acc_array, 1)[i])/acc_array.shape[1])
else:
	print(len(files))