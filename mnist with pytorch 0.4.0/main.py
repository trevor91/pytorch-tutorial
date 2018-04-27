# pytorch 0.4.0 in windows 10
# cuda 9.1
# 1080*4

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

#model
class Net(nn.Module):
	def __init__(self):
		super(Net,self).__init__()
		self.conv1 = nn.Conv2d(1,10,kernel_size=5)
		self.conv2 = nn.Conv2d(10,20, kernel_size=5)
		self.conv2_drop = nn.Dropout2d()
		self.fc1 = nn.Linear(320,50)
		self.fc2 = nn.Linear(50,10)

	def forward(self, x):
		x = F.relu(F.max_pool2d(self.conv1(x), 2))
		x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)),2))
		
		x = x.view(-1,320)
		
		x = F.relu(self.fc1(x))
		x = F.dropout(x, training=self.training)
		x = self.fc2(x)
		return F.log_softmax(x, dim=1)

if __name__ == '__main__':
	torch.manual_seed(2018)
	# use cuda
	usd_cuda = torch.cuda.is_available()
	device = torch.device("cuda")
	kwargs = {"num_workers":1, 'pin_memory':True}

	#data load
	train_loader = torch.utils.data.DataLoader(
		datasets.MNIST('./data', train = True, download = True,
			transform = transforms.Compose([
				transforms. ToTensor(),
				transforms.Normalize((0.1307,), (0.3081,))
				])),
		batch_size = 64, shuffle = True, **kwargs)

	test_loader =  torch.utils.data.DataLoader(
		datasets.MNIST('./data', train = False,
			transform = transforms.Compose([
				transforms. ToTensor(),
				transforms.Normalize((0.1307,), (0.3081,))
				])),
		batch_size = 64, shuffle = True, **kwargs)

	

	model = Net().to(device)

	optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

	def train(epoch):
		model.train()
		for batch_idx, (data, target) in enumerate(train_loader):
			data, target = data.to(device), target.to(device)
			optimizer.zero_grad()
			output = model(data)
			loss = F.nll_loss(output, target)
			loss.backward()
			optimizer.step()
			if batch_idx % 1000:
				print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'. format(
					epoch, batch_idx * len(data), len(train_loader.dataset),
					100.*batch_idx / len(train_loader), loss.item()))

	def test():
		model.eval()
		test_loss = 0
		correct = 0
		with torch.no_grad():
			for data, target in test_loader:
				data, target = data.to(device), target.to(device)
				output = model(data)
				test_loss += F.nll_loss(output, target, size_average=False).item()
				pred = output.max(1, keepdim = True)[1]
				correct += pred.eq(target.view_as(pred)).sum().item()

			test_loss /= len(test_loader.dataset)
			print('\nTest set: average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
				test_loss, correct, len(test_loader.dataset),
				100. * correct / len(test_loader.dataset)))

	for epoch in range(1,11):
		train(epoch)
		test()
	else:
		torch.save(model, 'mnist_model.pt')