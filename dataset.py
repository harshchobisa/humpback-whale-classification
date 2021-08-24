import os
import urllib.parse

from PIL import Image
import numpy as np
import pandas as pd
import torch
from torchvision import transforms

#goal: generate a vector for each image in our database. take an input image, run through model, compare that vector w/ database to find match 
#focus of this dataset is for calculating embeddings/training the Siamese Network 
class WhaleDataset(torch.utils.data.Dataset):
    #try to get every possible positive pair - for loop pairing each whale w/ itself as many times as you can
    #then, append negative images? 
    #two options. find every possible positive pair or 1 for each? keep in mind: might overfit to jimmy
    #
    
    def __init__(self, data, image_path):
        self.data = pd.read_csv(data) #file names. open the csv file 
        self.image_path = image_path #path to images themselves
        self.device = None

    def to(self, device):
        self.device = device
        return self

    #return triplets - represented as three individual tensors 
    def __getitem__(self, index):
        #1) go to row indicated by index
        triplet = self.data.iloc[index] 
        
        #in that row in the dataframe, convert to tensors. 

        
        anchor = Image.open(self.image_path + triplet.anchor)
        anchor = anchor.convert("RGB") #convert to RGB 
        positive = Image.open(self.image_path + triplet.positive)
        positive = positive.convert("RGB")
        negative = Image.open(self.image_path + triplet.negative)
        negative = negative.convert("RGB")

        #2) convert to tensor and resize so that every image is consistent. compose combines operations
        process = transforms.Compose([
            transforms.Resize((200, 400)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) #I'm not sure if these are the right values - Frank, Mar 20. guess we'll find out
            ])

        anchor = process(anchor)
        positive = process(positive)
        negative = process(negative)

        anchor, positive, negative = anchor.to(self.device), positive.to(self.device), negative.to(self.device)

        #3) return
        triplet_tensors = (anchor, positive, negative) # () tuples. {} dictionary. [] lists
        return (triplet_tensors)

    def __len__(self):
        return len(self.data)

#make sure to make a WhaleIDDataset of ONLY test whales and one of ONLY train whales.
class WhaleIDDataset(torch.utils.data.Dataset):
    def __init__(self, data, image_path):
        self.data = pd.read_csv(data) #only train.csv so hardcode is find
        # self.data = self.data.drop_duplicates(subset = "Id") #subset = "Id" - means we elim duplicates of the Id's so we get one image of each whale. no duplicates
        self.image_path = image_path
        self.device = None

    def to(self, device):
        self.device = device
        return self

    def __getitem__(self, index):
        #1) get the indiv image and its id
        row = self.data.iloc[index]
        image_name = row.Image
        image_id = row.Id
        image = Image.open(self.image_path + image_name)
        
        #2) process the image
        image = image.convert("RGB")
        process = transforms.Compose([
            transforms.Resize((200, 400)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) #I'm not sure if these are the right values - Frank, Mar 20. guess we'll find out
            ])

        image = process(image)
        image = image.to(self.device)

        #3) return image and its corresponding id
        return (image, image_id)

    def __len__(self):
        return len(self.data)


class PairsDataset(torch.utils.data.Dataset):
    def __init__(self, data, image_path):
        self.data = pd.read_csv(data) #only train.csv so hardcode is find
        self.image_path = image_path
        self.device = None

    def to(self, device):
        self.device = device
        return self

    def __getitem__(self, index):
        #1) get the indiv image and its id
        row = self.data.iloc[index]
        image_name_one = row.image_one
        image_id_one = row.id_one
        image_name_two = row.image_two
        image_id_two = row.id_two
        same_whale = row.same_whale
        image_one = Image.open(self.image_path + image_name_one)
        image_two = Image.open(self.image_path + image_name_two)

        
        #2) process the image
        image_one = image_one.convert("RGB")
        process = transforms.Compose([
            transforms.Resize((200, 400)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) #I'm not sure if these are the right values - Frank, Mar 20. guess we'll find out
            ])

        image_one = process(image_one)
        image_one = image_one.to(self.device)

        #2) process the image
        image_two = image_two.convert("RGB")
        process = transforms.Compose([
            transforms.Resize((200, 400)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) #I'm not sure if these are the right values - Frank, Mar 20. guess we'll find out
            ])

        image_two = process(image_two)
        image_two = image_two.to(self.device)

        #3) return image and its corresponding id
        return (image_one, image_two, same_whale)

    def __len__(self):
        return len(self.data)