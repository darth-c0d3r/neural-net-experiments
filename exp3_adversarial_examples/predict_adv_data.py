import sys
import torch
import torchvision
import numpy as np
from PIL import Image
import os

cuda = 1
gpu_id = 0
device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() and cuda == 1 else "cpu")
print("Device:", device)

# model on which to test
if(torch.cuda.is_available() == False):
	model = torch.load("saved_models/model_convnet.pt", map_location='cpu').to(device)
else:
	model = torch.load("saved_models/model_convnet.pt").to(device)

model.eval()
root_dir_input = "adversarial_outputs_conv/targeted/"
files = os.listdir(root_dir_input)
files.sort()


with torch.no_grad():
	for file in files:
		inp = Image.open(root_dir_input + file)
		trans = torchvision.transforms.ToTensor()
		tensor = trans(inp)
		tensor = tensor.view(1,tensor.shape[0],tensor.shape[1],tensor.shape[2]).to(device)
		# output = torch.softmax(model(tensor), dim=1)
		output = torch.sigmoid(model(tensor))
		print(file[-5:-4], output.data.cpu().numpy()[0])