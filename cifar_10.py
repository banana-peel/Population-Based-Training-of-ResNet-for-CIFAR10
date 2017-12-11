#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 18:29:30 2017

@author: saurabh
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np

import timeit

import matplotlib.pyplot as plt
import load_cifar_data as ld
from torchvision.models import resnet
import cifar_resnet as cr

#Load images

X_train, y_train, X_val, y_val, X_test, y_test = ld.get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000,
                     subtract_mean=True)

    
for i in range(4):
    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.axis('off')
    plt.imshow(X_train[i])
    
plt.show()


#Create Dataset

class cifar_data(Dataset):
    '''
    Creates the dataset for the cifar data
    initiates the __len__ and __getitem__
    '''
    def __init__(self,images,labels, transform = None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,indx):
        imgs = self.images[indx]
        labls = self.labels[indx]
        
        if self.transform:
            imgs, labls = self.transform((imgs,labls))
        return (imgs,labls)
    
req_dataset = cifar_data(X_train,y_train)

# Create the transforms

class ToTensor(object):
    '''Convert ndarrays in sample to Tensors
        We don't need to make a tensor out of labels
        as it is only 1 dim array
        '''

    def __call__(self, inpt):

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image,label = inpt
        image = image.transpose((2, 0, 1))
        return (torch.from_numpy(image).float(),label)

trfm = ToTensor()

tfm_img, tfm_lab = trfm(req_dataset[10])

# Test the Dataset

transformed_dataset = cifar_data(X_train,y_train,transform=ToTensor())
validation_data = cifar_data(X_val,y_val,transform=ToTensor())
test_data = cifar_data(X_test,y_test,transform=ToTensor())

#for i in range(3,7):
#    sample = transformed_dataset[i]
#    print(i, sample[0].size(), sample[1],'\n')
    
# Create the data loader and test it
    
dataloader = DataLoader(transformed_dataset, batch_size = 10, shuffle = True, num_workers=1)
val_dataloader = DataLoader(validation_data, batch_size = 100, shuffle = True, num_workers=0)
test_dataloader = DataLoader(test_data, batch_size = 100, shuffle = True, num_workers=0)
#g= 0
#for samples in dataloader:
#    img,labels = samples
#    print(labels)
#    g += 1
#    if g == 3:
#        break
    
#Load the model
#model = resnet.ResNet(BasicBlock, [2, 2, 2, 2], num_classes = 10)
model = cr.ResNet56()
    
#for i,data in enumerate(dataloader):
#    inp = Variable(data[0])
#    y = model(inp)
#    print(y)
#    break

# Defining Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)

# Training the network

val_img, val_label = next(iter(val_dataloader))
correct = 0
total = 0
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 1000 == 999:
            val_outputs = model(Variable(val_img))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            print("Validation accuracy = ", 100*correct/total)
            total=0
            correct=0
        if i % 100 == 99:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished Training')




