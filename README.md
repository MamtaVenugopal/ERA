# ERA S10 - Classification of CIFAR10
This is the assignment given in 10th Session, ERA batch of School of AI

In this session, the CIFAR10 dataset has been used. It consists of 60,000 training images and 10,000 testing image.
It is a large database of handwritten digits that is commonly used for training various image processing systems.
It has all the same square size images of 32X32 pixels, and they are colored images.
# Contents
```
.
├── README.md
├── S10.ipynb
├── new_model.py
└── show_images.py
└── train_test.py
└── train_test_loader.py
```

## S10.ipynb 
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


## new_model.py

This file contains the model used for predicting the MNIST dataset. The model summary is described below:
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,728
       BatchNorm2d-2           [-1, 64, 32, 32]             128
              ReLU-3           [-1, 64, 32, 32]               0
            Conv2d-4          [-1, 128, 32, 32]          73,728
         MaxPool2d-5          [-1, 128, 16, 16]               0
       BatchNorm2d-6          [-1, 128, 16, 16]             256
              ReLU-7          [-1, 128, 16, 16]               0
            Conv2d-8          [-1, 128, 16, 16]         147,456
       BatchNorm2d-9          [-1, 128, 16, 16]             256
             ReLU-10          [-1, 128, 16, 16]               0
           Conv2d-11          [-1, 128, 16, 16]         147,456
      BatchNorm2d-12          [-1, 128, 16, 16]             256
             ReLU-13          [-1, 128, 16, 16]               0
           Conv2d-14          [-1, 256, 16, 16]         294,912
        MaxPool2d-15            [-1, 256, 8, 8]               0
      BatchNorm2d-16            [-1, 256, 8, 8]             512
             ReLU-17            [-1, 256, 8, 8]               0
           Conv2d-18            [-1, 512, 8, 8]       1,179,648
        MaxPool2d-19            [-1, 512, 4, 4]               0
      BatchNorm2d-20            [-1, 512, 4, 4]           1,024
             ReLU-21            [-1, 512, 4, 4]               0
           Conv2d-22            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-23            [-1, 512, 4, 4]           1,024
             ReLU-24            [-1, 512, 4, 4]               0
           Conv2d-25            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-26            [-1, 512, 4, 4]           1,024
             ReLU-27            [-1, 512, 4, 4]               0
        MaxPool2d-28            [-1, 512, 1, 1]               0
           Linear-29                   [-1, 10]           5,120
================================================================
Total params: 6,573,120
Trainable params: 6,573,120
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 6.44
Params size (MB): 25.07
Estimated Total Size (MB): 31.53
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
# Augmentated Images
![image](https://github.com/MamtaVenugopal/ERA/assets/42015212/5c91e610-1fa7-4d9b-86f2-66dd6562a79c)

## Results
MaxLR = 4.51E-02 with steepest gradient using LRFinder
![image](https://github.com/MamtaVenugopal/ERA/assets/42015212/79ae17b7-e22b-488d-8961-8cf5dcea6b8e)

Using One Cycle Policy such that:
Total Epochs = 24
Max at Epoch = 5
LRMIN = 0.000451
LRMAX = 0.002353737996777662
NO Annihilation
        

