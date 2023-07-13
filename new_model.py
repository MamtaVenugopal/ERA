import torch.nn.functional as F
dropout_value = 0.1
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block C1
        # PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
        self.preplayer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1,stride=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) # output_size = 32

        # Layer 1
        # X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
        # R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k]
        # Add(X, R1)
        self.x1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3),stride=1, padding=1, bias=False),
            nn.MaxPool2d(2, 2), # output_size = 16
            nn.ReLU(),
            nn.BatchNorm2d(128),
            #nn.Dropout(dropout_value)
        ) # output_size = 32

        self.R1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3),stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3),stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value)
        ) # # output_size = 16

        # Layer 2
        # Conv 3x3 (s1, p1) [256k]>> MaxPool2D >> BN >> RELU [256k]
        self.layer2  = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1,stride=1, bias=False),
            nn.MaxPool2d(2, 2), # output_size = 8
            nn.ReLU(),
            nn.BatchNorm2d(256),
            #nn.Dropout(dropout_value)
         ) # output_size = 8


        # Layer 3
        # X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
        # R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
        # Add(X, R2)
        self.x2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3),stride=1, padding=1, bias=False),
            nn.MaxPool2d(2, 2), # output_size = 4
            nn.ReLU(),
            nn.BatchNorm2d(512),
            #nn.Dropout(dropout_value)
        ) # output_size = 32

        # CONVOLUTION BLOCK C4
        self.R2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3),stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3),stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Dropout(dropout_value)
        ) # # output_size = 4

        # MaxPooling with Kernel Size 4
        self.pool = nn.MaxPool2d(4, 4) # output_size = 2


       # FC Layer
        self.fc1 = nn.Linear(512, 10,bias=False)
      
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
