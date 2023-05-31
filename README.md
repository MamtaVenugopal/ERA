# ERA S5 - Classification of MNIST 
This is the assignment given in 5th Session, ERA batch of School of AI

In this session, the MNIST dataset has been used. It consists of 60,000 training images and 10,000 testing image.
It is a large database of handwritten digits that is commonly used for training various image processing systems.
It has all the same square size images of 28×28 pixels, and they are of grayscale.
# Contents
```
.
├── README.md
├── S5.ipynb
├── model.py
└── utils.py
```

## S5.ipynb 
This juputer notebook is the main program from where all other files like utils.py and models.py are called from. 
Code is written for classification of MNIST dataset using pytorch. PyTorch is a fully featured framework for building deep 
learning models, which is a type of machine learning that's commonly used in applications like image recognition and 
language processing. Model optimization used in this notebook Stochastic Gradient Descent (SGD), variant of the Gradient Descent
algorithm is used for optimizing model build with learning rate of 0.01 and scheduled learning rate of 0.1 every 20 epochs. 
By comparing the actual labels to the projected values, loss is computed. This loss is backprogragated by updating the weights with
gradients such that next iteration, the loss is reduced.

## utils.py
This file contains torch.utils.data.DataLoader and torch.utils.data.Dataset that allow you to use 
pre-loaded datasets as well as your own data. Dataset stores the samples and their corresponding labels, 
and DataLoader wraps an iterable around the Dataset to enable easy access to the samples.
They are implemented in function named Dataloader that returns train dataset, test dataset, train dataloader and test dataloader.
Along with function utils.py contains function for plotting the data.
![image](https://github.com/MamtaVenugopal/ERA/assets/42015212/cae992fc-3e41-4175-9464-b97570034845)


## model.py

This file contains the model used for predicting the MNIST dataset. The model summary is described below:
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 26, 26]             320
            Conv2d-2           [-1, 64, 24, 24]          18,496
            Conv2d-3          [-1, 128, 10, 10]          73,856
            Conv2d-4            [-1, 256, 8, 8]         295,168
            Linear-5                   [-1, 50]         204,850
            Linear-6                   [-1, 10]             510
================================================================
Total params: 593,200
Trainable params: 593,200
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.67
Params size (MB): 2.26
Estimated Total Size (MB): 2.94
----------------------------------------------------------------
```
This model has been built with 2 blocks of 2 convoluted layer followed by maxpooling and finally with two layers of fully 
connected. 
```
    def forward(self, x):
        x = F.relu(self.conv1(x), 2) # 28>26 | 1>3 | 1>1
        x = F.relu(F.max_pool2d(self.conv2(x), 2)) #26>24>12 | 3>5>6 | 1>1>2
        x = F.relu(self.conv3(x), 2) # 12>10 | 6>10 | 2>2
        x = F.relu(F.max_pool2d(self.conv4(x), 2)) # 10>8>4 | 10>14>16 | 2>2>4
        x = x.view(-1, 4096) # 4*4*256 = 4096
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
```
## Results

![image](https://github.com/MamtaVenugopal/ERA/assets/42015212/59d381fd-f0d1-4781-b9eb-cefc479d02a3)

        

