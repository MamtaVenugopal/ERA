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
Code is written for classification of CIFAR10 dataset using pytorch. PyTorch is a fully featured framework for building deep 
learning models, which is a type of machine learning that's commonly used in applications like image recognition and 
language processing. Model optimization used in this notebook Adam.  
Resnet like model used
1. PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
2. Layer1 -
   X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
   R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
   Add(X, R1)
3. Layer 2 - Conv 3x3 [256k],
   MaxPooling2D
   BN
   ReLU
4. Layer 3 -
   X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
   R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
   Add(X, R2)
5. MaxPooling with Kernel Size 4
6. FC Layer 
7. SoftMax
One cycle used for building model with accuracy above 90%

## Augmentated Images

![image](https://github.com/MamtaVenugopal/ERA/assets/42015212/0ccb3eb4-68b5-4bf8-9f7c-7fc98d024eff)


![image](https://github.com/MamtaVenugopal/ERA/assets/42015212/1443f03b-ba75-4a09-8f12-262bc1ff5b84)












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
This model has been built with custom resnet model
```
    def forward(self, x):

        preplayer = self.preplayer(x) 
        x1 = self.x1(preplayer)
        R1 = self.R1(x1)
        layer1 = x1+R1
        layer2 = self.layer2(layer1)
        x2 = self.x2(layer2)
        R2 = self.R2(x2)
        layer3 = R2+x2
        maxpool = self.pool(layer3)
        x = maxpool.view(maxpool.size(0),-1)
        fc = self.fc(x)
        return F.log_softmax(fc.view(-1,10), dim=-1)
```


## Results
MaxLR = 4.51E-02 with steepest gradient using LRFinder
![image](https://github.com/MamtaVenugopal/ERA/assets/42015212/79ae17b7-e22b-488d-8961-8cf5dcea6b8e)

Using One Cycle Policy such that:
Total Epochs = 24
Max at Epoch = 5
LRMIN = 0.000451
LRMAX = 0.002353737996777662
NO Annihilation

Accuracy without augmentation = 70.50% for training and 72% for testing dataset
Accuracy with augmentation    = 91.82% for training and 90.51% for testing dataset
        

