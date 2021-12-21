import torch.nn as nn
import torch.nn.functional as F
import torch

class LGF(nn.Module):
    def __init__(self):
        super(LGF, self).__init__()
        self.fc1 = nn.Linear(54, 54, bias = True)
        self.fc2 = nn.Linear(54, 54)
        self.fc3 = nn.Linear(54, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=.3)
        x = F.relu(self.fc3(x))

        #return F.softmax(x, dim=1)
        return x



class LGF2d(nn.Module):
    def __init__(self, training = False):
        super(LGF2d, self).__init__()
        self.conv1 = nn.Conv1d(6, 6, 2, padding=1)
        self.training = training
        self.fc1 = nn.Linear(330, 150, bias = True)
        self.fc2 = nn.Linear(150, 50)
        self.fc3 = nn.Linear(50, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        #x = x.view(-1, 6*9*6)   #6x9 face, 6 channels
        #print(x.shape())
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=.3, training=self.training)
        x = F.relu(self.fc2(x))
        
        x = self.fc3(x)

        return x



class LGF3d(nn.Module):
    def __init__(self, training = False):
        super(LGF3d, self).__init__()
        self.conv1 = nn.Conv3d(6, 12, 2, padding=(1,1,1))
        self.conv2 = nn.Conv3d(12, 24, 2, padding=(1,1,1))
        self.conv3 = nn.Conv3d(24, 36, 2, padding=(1,1,1))
        self.training = training
        self.fc1 = nn.Linear(4800, 2000, bias = True)
        self.fc2 = nn.Linear(2000, 800, bias = True)
        self.fc3 = nn.Linear(800, 400, bias = True)
        self.fc4 = nn.Linear(400, 100, bias = True)
        self.final = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        #x = x.view(-1, 6*9*6)   #6x9 face, 6 channels
        #print(x.shape())
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=.33, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        #x = F.relu(self.fc5(x))
        
        x = self.final(x)

        return x