import sys
sys.path.append('..')
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F

class Discriminator(nn.Module):

    def __init__(self, hidden_size=230, num_labels=2, N=2*5*2): #batch * N * 2
        nn.Module.__init__(self)
        # self.hidden_size = 768*3
        # hidden_size = 768*3
        self.num_labels = num_labels
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu1 = nn.LeakyReLU()
        self.drop0 = nn.Dropout(0.5)
        self.drop = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_size, 2)
        k = 5
        p = 2
        self.conv1 = nn.Sequential(  #
            nn.Conv1d(1, 256, k, padding=p),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(256, 64, k, padding=p),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            # nn.MaxPool1d(2),
            nn.Conv1d(64, 1, k, padding=p),
            nn.LeakyReLU(),
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.drop(x)
        x = x.squeeze(1)

        # x = self.fc1(x)
        # x = self.relu1(x)
        # x = self.drop0(x)

        logits = self.fc2(x)
        # print(logits.shape)
        return logits

    # def forward(self, x):
    #     x = self.fc1(x)
    #     # x = self.bn1(x)
    #     x = self.relu1(x)
    #     x = self.drop(x)
    #     # x = x.squeeze(1)
    #     logits = self.fc2(x)
    #     # print(logits.shape)
    #     return logits

# class Discriminator(nn.Module):
#   def __init__(self,hidden_size=230, num_labels=2):
#     super(Discriminator,self).__init__()
#     self.layer1 = nn.Sequential(nn.Linear(hidden_size,hidden_size),nn.ReLU())
#     self.layer2 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU())
#     self.layer3 = nn.Sequential(nn.Linear(hidden_size,num_labels))
#     self.drop = nn.Dropout()
#
#   def forward(self,x):
#     x = self.layer1(x)
#     x = self.drop(x)
#     x = self.layer2(x)
#     x = self.drop(x)
#     x = self.layer3(x)
#     return x