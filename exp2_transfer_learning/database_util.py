import torchvision
import torch
import torch.utils.data as utils

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

def split_database(db, frac):
	db_l_train_data = []
	db_l_train_target = []
	db_r_train_data = []
	db_r_train_target = []
	db_l_test_data = []
	db_l_test_target = []
	db_r_test_data = []
	db_r_test_target = []

	for data, target in db['train']:
		if target < 5:
			db_l_train_data += [data]
			db_l_train_target += [torch.tensor(target)]
		else:
			db_r_train_data += [data]
			db_r_train_target += [torch.tensor(target-5)]


	for data, target in db['eval']:
		if target < 5:
			db_l_test_data += [data]
			db_l_test_target += [torch.tensor(target)]
		else:
			db_r_test_data += [data]
			db_r_test_target += [torch.tensor(target-5)]

	frac_train = int(frac * len(db_l_train_data))

	tdb_l_train_data = torch.stack([point for point in db_l_train_data[:frac_train]])
	tdb_r_train_data = torch.stack([point for point in db_r_train_data[:frac_train]])
	tdb_l_train_target = torch.stack([point for point in db_l_train_target[:frac_train]])
	tdb_r_train_target = torch.stack([point for point in db_r_train_target[:frac_train]])
	tdb_l_test_data = torch.stack([point for point in db_l_test_data])
	tdb_r_test_data = torch.stack([point for point in db_r_test_data])
	tdb_l_test_target = torch.stack([point for point in db_l_test_target])
	tdb_r_test_target = torch.stack([point for point in db_r_test_target])

	# my_dataset = utils.TensorDataset(tensor_x,tensor_y)
	db_l_train = utils.TensorDataset(tdb_l_train_data, tdb_l_train_target)
	db_l_test = utils.TensorDataset(tdb_l_test_data, tdb_l_test_target)
	db_r_train = utils.TensorDataset(tdb_r_train_data, tdb_r_train_target)
	db_r_test = utils.TensorDataset(tdb_r_test_data, tdb_r_test_target)

	db_l = {'train':db_l_train, 'eval':db_l_test}
	db_r = {'train':db_r_train, 'eval':db_r_test}
	return db_l, db_r
