import torch
from torch.utils import data
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import csv
from progressbar import *               # just a simple progress bar


import my2_classes # With that way we import the my_classes code inside and we
# have the ability to run my_classes.customDatasetVideos in the next block

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # set up the GPU
print(device)


class Net(nn.Module):

    def __init__(self):  # Initialize the parameters of the model

        super(Net, self).__init__()

        # SOS SOS SOS SOS
        # On the paper we needed to pad in order to have the same size for input and output layers
        # But we don't do that here

        # SOS SOS
        # Now I am using padding 2 which is wrong, I need to have padding 1

        #         self.conv1 = nn.Conv3d(3,64,kernel_size=3,padding=2)
        #         self.conv2 = nn.Conv3d(64,128,kernel_size=3,padding=2)
        #         self.conv3 = nn.Conv3d(128,256,kernel_size=3,padding=2)
        #         self.conv4 = nn.Conv3d(256,256,kernel_size=3,padding=2)
        #         self.conv5 = nn.Conv3d(256,256,kernel_size=3,padding=2)
        self.conv1 = nn.Conv3d(3, 10, kernel_size=3, padding=2)
        self.conv2 = nn.Conv3d(10, 20, kernel_size=3, padding=2)
        self.conv3 = nn.Conv3d(20, 25, kernel_size=3, padding=2)
        self.conv4 = nn.Conv3d(25, 30, kernel_size=3, padding=2)
        self.conv5 = nn.Conv3d(30, 35, kernel_size=3, padding=2)

        self.mp222 = nn.MaxPool3d(2)  # This means that the size of the window will me 2 so the whole image will be devidd by 2
        self.mp122 = nn.MaxPool3d((1, 2, 2))  # This means that the size of the window will me 2 so the whole image will be devidd by 2

        # SOS SOS
        # The fully connected layers goes from something really large to
        # Something really small in only one layer

        self.fc1 = nn.Linear(1400, 500)  # 17920
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, 46)  # need to 45 because the classes are 45

    def forward(self, x):
        in_size = x.size(0)
        x = self.conv1(x)
        x = self.mp122(x)

        x = F.relu(x)
        x = F.relu(self.mp222(self.conv2(x)))
        x = F.relu(self.mp222(self.conv3(x)))
        x = F.relu(self.mp222(self.conv4(x)))
        x = F.relu(self.mp222(self.conv5(x)))

        x = x.view(in_size, -1)  # flatten the tensor

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return F.log_softmax(x)


model = Net()
model = model.to(device)

optimizer = optim.SGD(model.parameters(), lr=0.007, momentum=0.5)


custom_mnist_from_csv = my2_classes.customDatasetVideos('videoPath_labels.csv', frameStart=5, frames=15, startH=30, startW=30, heightVid=80, widthVid=100)

mn_dataset_loader = torch.utils.data.DataLoader(dataset=custom_mnist_from_csv, batch_size=1, shuffle=True, num_workers=7)


model.load_state_dict(torch.load("../../modelParameters/testBC4.pth"))
model.eval()

#ArrI = []
#ArrLoss = []
i = 0

for epoch in range(50):  # An epoch is basically when we finish the wholde data set and we want to train again from the start

    model.train()
    for vid, labels in mn_dataset_loader:  # This makes many itterations until finishes the whole data set which is really long about 6000 videos this menas the the whole system is really slow and we need to chagne something in the future
        i = i + 1

        vid = vid.type('torch.FloatTensor')

        vid, labels = vid.to(device), labels.to(device)  # With this way we transfer all the arrays to the GPU in order to make the processes there


        optimizer.zero_grad()
        output = model(vid)

        loss = F.nll_loss(output, labels)
        loss.backward()

        optimizer.step()


        print(epoch,i,loss.data[0])#, loss.data[0])


    torch.save(model.state_dict(), "../../modelParameters/testBC4.pth")

#np.save('LossArray',ArrLoss)
#np.save('ArrI',ArrI)
