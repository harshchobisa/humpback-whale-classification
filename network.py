'''
---Overview---
consider doing an encoding - take our image, calculate an encoding for that image that is smaller than original tensor 
but captures all key info (ie is it blue, raw shape?). our siamese network will calculate an encoding for our image. 
we kind of do this already in a neural network - each layer will break down the image more, then pass it in to the next one. moves image vector
to a space that is easier to work with. can more easily draw conclusions (ie logistic regression)
or w/ CNN's, each filter is classifying increasingly more complex features. first layers see low level features, later layers will look at more abstract ideas
triplet loss is a mechanism to calculate useful embeddings. siamese network generates embeddings. very cool
'''

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class SiameseNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #consider adding conv before 
        resnet_full = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True) #transfer learning. 
        self.resnet = torch.nn.Sequential(*list(resnet_full.children())[:-1]) #take everything except the last layer, combine w/ torch.nn.Sequential?
        self.resnet.eval() #batch norm, dropout, etc have diff behavior at test time. this tells it to use test time behavior 
        self.fc1 = nn.Linear(512, 256) #resnet outputs size 512
        self.fc2 = nn.Linear(256, 64) #get a size 64 encoding at the end 
        self.fc3 = nn.Linear(64, 2) 
        # self.relu = F.relu()
    
    def forward(self, x): 
        with torch.no_grad():
            out = self.resnet(x) #freeze weights for just resnet. still need to tune our output layers! 
        out = out.reshape((len(x), 512))
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
