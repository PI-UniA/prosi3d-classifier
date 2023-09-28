import torch
import torch.nn as nn
import torch.nn.functional as F


class ProsiNN(nn.Module):
    def __init__(self,errorclasses = 10,dropout = 0.2):
        super().__init__()
        self.inn = ImageNN(3)
        self.ann = AudioNN()

        self.maxpool1d = nn.MaxPool1d(kernel_size=2,stride=2)

        self.cls0 = nn.Sequential(
            nn.Linear(64,96),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(96,192),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(192,384),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(384,768),
            nn.Dropout(dropout),
            nn.ReLU()
        )

        self.cls1 = nn.Sequential(
            nn.Linear(786,786),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(786,768),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(786,384),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(384,64),
            nn.ReLU(),
            nn.Linear(64,errorclasses),
            nn.Softmax()
        )


    def forward(self, x, y ):
        x = self.inn(x)
        y = self.ann(y)
        z = torch.cat((x,y),1)
        z = self.cls0(z)
        z = self.maxpool1d(z)
        z = self.cls1(z)

        return z




class ProsiSingleNN(nn.Module):
    def __init__(self, errorclasses=10, dropout=0.2):
        super().__init__()
        self.sinn = ImageNN2()


        self.maxpool1d = nn.MaxPool1d(kernel_size=2, stride=2)

        self.cls0 = nn.Sequential(
            nn.Linear(400, 200),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(100, errorclasses),

        )

        self.cls1 = nn.Sequential(
            nn.Linear(768, 768),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(768, 768),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(768, 384),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(384, 64),
            nn.ReLU(),
            nn.Linear(64, errorclasses),

        )

    def forward(self, x):
        z = self.sinn(x)
        z = self.cls0(z)
      #  z = self.maxpool1d(z)
      #  z = self.cls1(z)

        return z




class ImageNN(nn.Module):
    def __init__(self, in_channel ,output_layer= 32, out_channels=[64,96,192,384,768,1536]):
        super().__init__()

        self.maxpool2d = nn.MaxPool2d(kernel_size=2,stride=2)
        self.flatten = nn.Flatten()

        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channel,in_channel,kernel_size=7, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(in_channel, out_channels[0], kernel_size=5,stride=2, padding=2 ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(out_channels[0],out_channels[1],kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels[1],out_channels[2],kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels[2],out_channels[3],kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
        )

        self.mlp0 = nn.Sequential(
            nn.Linear(out_channels[3],out_channels[4]),
            nn.ReLU(),
            nn.Linear(out_channels[4],out_channels[4]),
            nn.ReLU()
        )

        self.mlp1 = nn.Sequential(
            nn.Linear(out_channels[4],out_channels[3]),
            nn.ReLU(),
            nn.Linear(out_channels[3],out_channels[2]),
            nn.ReLU(),
            nn.Linear(out_channels[2],output_layer),
            nn.ReLU()
        )


    def forward(self,x):

        y = self.conv0(x)
        y = self.conv1(y)
        y = self.maxpool2d(y)
        y = self.flatten(y)
        y = self.mlp0(y)
        y = self.mlp1(y)

        return y

class ImageNN2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,6,kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(6,16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2,2)

        )
        self.mlp0 = nn.Sequential(
            nn.Linear(400,800),
            nn.ReLU(),
            nn.Linear(800,800),
            nn.ReLU(),
            nn.Linear(800,400)

        )
    def forward(self,x):
        x = self.conv(x)
        x = torch.flatten(x,1)
        x = self.mlp0(x)
        return x



#RNN?
class AudioNN(nn.Module):
    def __init__(self,output_layer = 32):
        super().__init__()

        self.flatten = nn.Flatten()

        self.ftr = nn.Sequential(
            nn.Conv2d(3,32, kernel_size=5,stride=1,padding=0),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=5,stride=1,padding=0),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
        )

        self.mlp0 = nn.Sequential(
            nn.Linear(51136,512),
            nn.ReLU(),
            nn.Linear(512,output_layer),
            nn.ReLU()
        )

    def forward(self,x):
        x = self.ftr(x)
        x = self.flatten(x)
        x = self.mlp0(x)
        return x




class simpleNN(nn.Module):
    def __init__(self,outputlayer = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, outputlayer)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


