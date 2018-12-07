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

cuda = 1
device = torch.device("cuda" if torch.cuda.is_available() and cuda == 1 else "cpu")
print("Device:", device)

db = prepare_db()
db_l, db_r = split_database(db, 0.75)

model = torch.load("convnet_right.pt", map_location='cpu').to(device)
model.eval()
correct = 0
eval_loader = torch.utils.data.DataLoader(db_l['eval'], batch_size=len(db_l['eval']), shuffle=True)

with torch.no_grad():
	for data, target in eval_loader:

		data, target = Variable(data.to(device)), Variable((target.float()).to(device))
		output = model(data)
		pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
		correct += pred.eq(target.data.view_as(pred).long()).cpu().sum()

accuracy = float(correct) / len(eval_loader.dataset)

print('Accuracy: {}/{} ({:.6f})'.format(
	correct, len(eval_loader.dataset),
	accuracy))

# --------------------------------------------------------------------------------------------------

correct = 0
eval_loader = torch.utils.data.DataLoader(db_r['eval'], batch_size=len(db_r['eval']), shuffle=True)

with torch.no_grad():
	for data, target in eval_loader:

		data, target = Variable(data.to(device)), Variable((target.float()).to(device))
		output = model(data)
		pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
		correct += pred.eq(target.data.view_as(pred).long()).cpu().sum()

accuracy = float(correct) / len(eval_loader.dataset)

print('Accuracy: {}/{} ({:.6f})'.format(
	correct, len(eval_loader.dataset),
	accuracy))