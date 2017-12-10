#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 17:44:38 2017

@author: saurabh
"""
import torch
from torchvision.models import resnet
import torch.nn as nn
import math

# Have to use resnet.BasicBlock

class cifarResnet(nn.Module):
    '''
    Resnet class tailored for CIFAR dataset.
    The netowork is split into the following graph, all Convolution filters have size 3x3
    unless specified otherwise
    input -> Conv_layer1(3,16,32,32) -> Conv_block1(16,32,32) -> Conv_block2(32,16,16)
    -> Conv_block3(64,8,8) -> Avg_pooling(64,1,1) -> FC_layer(10)
    '''
    def __init__(self, block, layers, num_classes=10):
        self.inplanes = 16
        super(cifarResnet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8,stride = 1)
        self.fc = nn.Linear(64,num_classes)
        
        #Adding modified form of Xavier weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
    def _make_layer(self, block, planes, blocks, stride=1):
        '''
        This takes in the number of filters for input layer and number of filters for
        output layer and creates a sequential block. Also performs downsampling appropriately
        
        Input: block-> the basic building block. This is an nn.Module class
        planes-> Number of output filters
        blocks-> Number of basic elements in a block
        stride-> 1 by default. Require 2 only at the begining of the block
        
        Output: A sequential block
        '''
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
                    
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
        
    def forward(self, x):
        '''
        Puts the blocks created in a sequential format
        Note that forward function seems to subtitute the default function call
        for the nn.Module
        '''
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def ResNet56(**kwargs):
    '''
    Creates a 56 layer model
    1 + 2n + 2n + 2n + 1
    '''
    model = cifarResnet(resnet.BasicBlock, [9,9,9], **kwargs)
    return model
