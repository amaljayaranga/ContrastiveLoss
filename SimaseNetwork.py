import torch.nn as nn

class SimaseNet(nn.Module):

    def __init__(self):
        super(SimaseNet,self).__init__()

        self.conv = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=2, stride=2),
                                  nn.Conv2d(in_channels=32,out_channels=64, kernel_size=5),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=2, stride=2)
                                  )
        #output 64,4,4

        self.fc = nn. Sequential(nn.Linear(64*4*4,256),
                                 nn.ReLU(),
                                 nn.Linear(256,256),
                                 nn.ReLU(),
                                 nn.Linear(256,2)
                                 )

    def forward_once(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x

    def forward(self, in1, in2):
        out1 = self.forward_once(in1)
        out2 = self.forward_once(in2)
        return out1, out2



