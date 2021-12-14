import torch.nn as nn
import torch.nn.functional as F


class LGF(nn.Module):
    def __init__(self):
        self.fc1 = nn.Linear(54, 54, bias = True)
        self.fc2 = nn.Linear(54, 54)
        self.fc3 = nn.Linear(54, 1)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=.3)
        x = F.relu(self.fc3(x))

        return F.softmax(x)

