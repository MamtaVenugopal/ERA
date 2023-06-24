import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary

class model_1(nn.Module):
    def __init__(self):
        super(model_1, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            if norm == 'bn':
                self.n1 = nn.BatchNorm2d(output_size)
            elif norm == 'gn':
                self.n1 = nn.GroupNorm(GROUP_SIZE, output_size)
            elif norm == 'ln':
                self.n1 = nn.GroupNorm(1, output_size)
        ) # output_size = 28

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU()
        ) # output_size = 28
        # self.convblock3 = nn.Sequential(
        #     nn.Conv2d(in_channels=12, out_channels=20, kernel_size=(3, 3), padding=1, bias=False),
        #     nn.ReLU()
        # ) # output_size = 28

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 14
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 14

        # CONVOLUTION BLOCK 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 12
        # self.convblock6 = nn.Sequential(
        #     nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
        #     nn.ReLU()
        # ) # output_size = 7

        # OUTPUT BLOCK
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 12
        self.convblock8 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1))
            #nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(12, 12), padding=0, bias=False),
            # nn.ReLU() NEVER!
        ) # output_size = 1 7x7x10 | 7x7x10x10 | 1x1x10

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        #x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        #x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = x.squeeze()
        return F.log_softmax(x, dim=-1)


class model_2(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(model_2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),  #28x28x3 28x28x8
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 8, 3, padding=1),  #28x28x8 28x28x8

            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 8, 3, padding=1),  #28x28x8 28x28x8

            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2, 2), #28x28x8 14x14x8
            nn.Dropout(p=0.01)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 12, 3, padding=1),  #14x14x8 14x14x12
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Conv2d(12, 12, 3, padding=1),  #14x14x12 14x14x12
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Conv2d(12, 12, 3, padding=1),  #14x14x12 14x14x12
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.MaxPool2d(2, 2), #12x12x12 12x12x12
            nn.Dropout(p=0.01)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(12, 15, 3, padding=1),  #12x12x12 12x12x15
            nn.ReLU(),
            nn.BatchNorm2d(15),
            nn.Conv2d(15, 10, 3, padding=1),  #12x12x15 12x12x10
            nn.ReLU(),
            nn.BatchNorm2d(10),
            )
        self.pool = nn.AdaptiveAvgPool2d((1,1)) # 1x1x10


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = x.squeeze()
        return F.log_softmax(x, dim=1)

def model_summary(model,input_size):
    summary(model, input_size)

class model_3(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(model_3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),  #28x28x3 28x28x8
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 8, 3, padding=1),  #28x28x8 28x28x8

            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 8, 3, padding=1),  #28x28x8 28x28x8

            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2, 2), #28x28x8 14x14x8
            nn.Dropout(p=0.01)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 12, 3, padding=1),  #14x14x8 14x14x12
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Conv2d(12, 12, 3, padding=1),  #14x14x12 14x14x12
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Conv2d(12, 12, 3, padding=1),  #14x14x12 14x14x12
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.MaxPool2d(2, 2), #12x12x12 12x12x12
            nn.Dropout(p=0.01)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(12, 15, 3, padding=1),  #12x12x12 12x12x15
            nn.ReLU(),
            nn.BatchNorm2d(15),
            nn.Conv2d(15, 10, 3, padding=1),  #12x12x15 12x12x10
            nn.ReLU(),
            nn.BatchNorm2d(10),
            )
        self.pool = nn.AdaptiveAvgPool2d((1,1)) # 1x1x10


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = x.squeeze()
        return F.log_softmax(x, dim=1)

class model_4(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(model_4, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),  #28x28x3 28x28x8
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 8, 3, padding=1),  #28x28x8 28x28x8

            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 8, 3, padding=1),  #28x28x8 28x28x8

            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2, 2), #28x28x8 14x14x8
            nn.Dropout(p=0.01)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 12, 3, padding=1),  #14x14x8 14x14x16
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Conv2d(12, 12, 3, padding=1),  #14x14x16 14x14x16
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Conv2d(12, 12, 3, padding=1),  #14x14x16 14x14x16
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.MaxPool2d(2, 2), #12x12x16 12x12x16
            nn.Dropout(p=0.01)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(12, 10, 3, padding=1),  #14x14x32 12x12x32
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(p=0.01),
            nn.Conv2d(10, 10, 3, padding=1),  #14x14x32 12x12x32
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(p=0.01),
            nn.Conv2d(10, 10, 3, padding=1),  #12x12x32 12x12x10
            nn.ReLU(),
            nn.BatchNorm2d(10),
            )
        self.pool = nn.AdaptiveAvgPool2d((1,1)) # 1x1x10


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = x.squeeze()

        #x = x.squeeze()




        return F.log_softmax(x, dim=1)
