import torch
import torchvision

class Cifar10SearchDataset(datasets.CIFAR10):

    def __init__(self, root="~/data", train=True, download=True, transform=None):

        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):

        image, label = self.data[index], self.targets[index]

        if self.transform is not None:

            transformed = self.transform(image=image)

            image = transformed["image"]

        return image, label

def load(train_transform,test_transform,seed=1,batch_size=128,num_workers=4,pin_memory=True):
	

	#Get the Train and Test Set
  train = Cifar10SearchDataset(root='./data', train=True,download=True, transform=train_transforms)
  test = Cifar10SearchDataset(root='./data', train=False,download=True, transform=test_transforms)
  SEED = 1

	# CUDA?
  cuda = torch.cuda.is_available()


	# For reproducibility
  torch.manual_seed(SEED)

  if cuda:
			torch.cuda.manual_seed(SEED)

	# dataloader arguments - something you'll fetch these from cmdprmt
  dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory) if cuda else dict(shuffle=True, batch_size=64)

  trainloader = torch.utils.data.DataLoader(train, **dataloader_args)
  testloader = torch.utils.data.DataLoader(test, **dataloader_args)

  classes = ('plane', 'car', 'bird', 'cat',
    	       'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

  return classes, trainloader, testloader