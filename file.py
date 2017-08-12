from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.utils.data as utils_data
import numpy
from random import randrange
# fix random seed for reproducibility
numpy.random.seed(7)

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10000000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

class MyDataset():
    def __init__(self):
        datasetWin = numpy.genfromtxt("playerWinPreFlop.txt", delimiter=",")
        datasetLose = numpy.genfromtxt("playerLosePreFlop.txt", delimiter=",")

        self.winFeatures = datasetWin[:,0:36]
        self.winLabels = datasetWin[:,37]
        
        self.loseFeatures = datasetLose[:,0:36]
        self.loseLabels = datasetLose[:,37]
        
        if (len(self.winLabels)>len(self.loseLabels)):
            self.size = len(self.loseLabels)
        else:
            self.size = len(self.winLabels)

    def __getitem__(self, index):
        # choose random index in features
        winPick = randrange(0,self.size)
        losePick = randrange(0,self.size)

        if (winPick > losePick):#this is just to mix it to wins and losses to be on both sides
            #winner on left so zero
            data = numpy.array([(self.winFeatures[winPick], self.loseFeatures[losePick])])    
            target = numpy.array([(0)])
        else:
            #winner on the right so one
            data = numpy.array([(self.loseFeatures[losePick], self.winFeatures[winPick])])
            target = numpy.array([(1)])        

        data = data.reshape((72))
        data = torch.from_numpy(data).float()
        target = torch.from_numpy(target).float()

        
        #print('data', data)
        #print('target', target)
        return data, target

    def __len__(self):
        return self.size

train_loader = torch.utils.data.DataLoader(MyDataset(), batch_size=250)
test_loader = torch.utils.data.DataLoader(MyDataset(), batch_size=250)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        #self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        #self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(72, 72)
        self.fc2 = nn.Linear(72, 1)

    def forward(self, x):
        #print('x')
        #print(x)
        #x = F.relu(F.max_pool2d(self.conv1(x), 2))
        #x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        #x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.sigmoid(x)

model = Net()
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        #print(batch_idx)
        #print(data)
        #print(target)
        if args.cuda:
            print('cuda')
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        #print('output', output)
        #print('target', target)
        loss = F.binary_cross_entropy(output, target)
        #print('loss', loss)
        loss.backward()
        optimizer.step()
        if batch_idx % 1000000 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
            torch.save(model.state_dict(), "pokerNetSettings")

model.load_state_dict(torch.load('pokerNetSettings'))
for epoch in range(1, args.epochs + 1):
    train(epoch)
