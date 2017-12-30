#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 14:25:04 2017

@author: saurabh
"""
import subprocess
import os
from six.moves import cPickle as pickle
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torch


def download_data(filename):
    '''
    This will take the shell script filename as input and
    run the shell script to download the data
    '''
    subprocess.call([filename])
    
def load_data_from_file(filename):
    '''
    Load data from the individual files
    '''
    with open(filename,'rb') as f:
        inp_data = pickle.load(f, encoding='latin1')
        X = inp_data['data'].reshape(-1,3,32,32).transpose(0,2,3,1).astype('float')
        Y = np.array(inp_data['labels'])
        return X,Y
    
def combine_data(filename):
    '''
    This will combine/concatenate data from the batch files
    '''
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(filename, 'data_batch_%d' % (b, ))
        X, Y = load_data_from_file(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_data_from_file(os.path.join(filename, 'test_batch'))
    return Xtr, Ytr, Xte, Yte

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000,
                     subtract_mean=True):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for classifiers. These are the same steps as we used for the SVM, but
    condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'cifar-10-batches-py'
    X_train, y_train, X_test, y_test = combine_data(cifar10_dir)

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    
    if subtract_mean == True:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image

    return X_train, y_train, X_val, y_val, X_test, y_test





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


# Create the Dataset and the Dataloader

def create_dataloader(train_batch=10, validation_batch=100, test_batch=100):
    '''
    This will return dataloader with training, validatin and testing data respectively
    '''
    X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000,
                     subtract_mean=True)

    transformed_dataset = cifar_data(X_train,y_train,transform=ToTensor())
    validation_data = cifar_data(X_val,y_val,transform=ToTensor())
    test_data = cifar_data(X_test,y_test,transform=ToTensor())
    
        
    dataloader = DataLoader(transformed_dataset, batch_size = train_batch, shuffle = True, num_workers=0)
    val_dataloader = DataLoader(validation_data, batch_size = validation_batch, shuffle = True, num_workers=0)
    test_dataloader = DataLoader(test_data, batch_size = test_batch, shuffle = True, num_workers=0)
    
    return dataloader, val_dataloader, test_dataloader



