#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 18:29:30 2017

@author: saurabh
"""

#import torch
#import torch.nn as nn
#import torch.optim as optim
#from torch.autograd import Variable
#from torch.utils.data import DataLoader, Dataset
#from torch.utils.data import sampler
#
#import torchvision.datasets as dset
#import torchvision.transforms as T
#
#import numpy as np
#
#import timeit
#
#import matplotlib.pyplot as plt
#import load_cifar_data as ld
#from torchvision.models import resnet
#import cifar_resnet as cr
import torch
import torch.multiprocessing as mp
from torch.multiprocessing import set_start_method
import pickle

#Load images



    

#
## Defining Loss function and optimizer
#criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)

# Training the network



def training_cifar_multi(train_state_dict, val_acc_dict, net_acc_dict ,name,return_top_arg, learn_rate):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.autograd import Variable
#    from torch.utils.data import DataLoader, Dataset
#    from torch.utils.data import sampler
    
#    import torchvision.datasets as dset
#    import torchvision.transforms as T
    
    import numpy as np
    
#    import timeit
    
#    import matplotlib.pyplot as plt
    import load_cifar_data as ld
    import cifar_resnet as cr
    
#    import torch.multiprocessing as mp
    
    model = cr.ResNet56()
    model.cuda(rank)
    
    net_acc_dict[name]
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learn_rate, momentum=0.9, weight_decay=0.0001)
    
    dataloader, val_dataloader, test_dataloader = ld.create_dataloader()
    epoch = 0
    

    while epoch <= 13:
        model.train(True)
        total = 0
        correct = 0
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
            if i % 1000 == 999:    # print every 2000 mini-batches
                print('process =', name, '[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                temp_dict = net_acc_dict[name]
                temp_dict.append(running_loss / 1000)
                net_acc_dict[name] = temp_dict
                running_loss = 0.0
            if i==3:
                break
    
    #Saving model to manager
        train_state_dict[name] = {'state_dict': model.state_dict(), 'optimizer': 
                        optimizer.state_dict(), 'epoch':epoch}
        
        model.eval()
        for ix, (val_img,val_label) in enumerate(val_dataloader):
            val_outputs = model(Variable(val_img))
            _, predicted = torch.max(val_outputs.data, 1)
            total += val_label.size(0)
            correct += (predicted == val_label).sum()
        valid_accuracy = 100*correct/total
        print("Validation accuracy = ", valid_accuracy)
        val_acc_dict[name] = valid_accuracy
        total=0
        correct=0
    
        flag = return_top_arg(val_acc_dict, valid_accuracy)
        if flag:
            model.load_state_dict(train_state_dict[flag]['state_dict'])
            optimizer.load_state_dict(train_state_dict[flag]['optimizer'])
            epoch = train_state_dict[flag]['epoch']
            for param_group in optimizer.param_groups:
                param_group['lr'] = (np.random.uniform(0.5,2,1)[0])*param_group['lr']
        
        epoch += 1
            
    for ix, (test_img,test_label) in enumerate(test_dataloader):
        test_outputs = model(Variable(test_img))
        _, predicted = torch.max(test_outputs.data, 1)
        total += test_label.size(0)
        correct += (predicted == test_label).sum()
    test_accuracy = 100*correct/total
    print("Testing accuracy = ", test_accuracy)
        
        
        
        
        
        
        
def return_top_arg(inp_dict,val_acc):
    '''
    Input: a dictionary with the process name and corresponding validatin accuracy
    Output: a process name randomly taken from the top 20% of the processes if
            the calling process itself is not in top 20. Else return None
    '''
    import operator
    import numpy as np
    req_range = sorted(inp_dict.values())[int(len(inp_dict)*0.8):]
    if val_acc >= req_range[0]:
        return None
    sorted_dict = sorted(inp_dict.items(), key=operator.itemgetter(1))
    indx = np.random.randint(0,len(req_range))
    req_key = sorted_dict[-indx-1][0]
    return req_key
    
#correct = 0
#total = 0
#for epoch in range(2):  # loop over the dataset multiple times
#
#    running_loss = 0.0
#    for i, data in enumerate(dataloader, 0):
#        # get the inputs
#        inputs, labels = data
#
#        # wrap them in Variable
#        inputs, labels = Variable(inputs), Variable(labels)
#
#        # zero the parameter gradients
#        optimizer.zero_grad()
#
#        # forward + backward + optimize
#        outputs = model(inputs)
#        loss = criterion(outputs, labels)
#        loss.backward()
#        optimizer.step()
#
#        # print statistics
#        running_loss += loss.data[0]
#        if i % 1000 == 999:
#            val_outputs = model(Variable(val_img))
#            _, predicted = torch.max(outputs.data, 1)
#            total += labels.size(0)
#            correct += (predicted == labels).sum()
#            print("Validation accuracy = ", 100*correct/total)
#            total=0
#            correct=0
#        if i % 100 == 99:    # print every 2000 mini-batches
#            break
#            print('[%d, %5d] loss: %.3f' %
#                  (epoch + 1, i + 1, running_loss / 100))
#            running_loss = 0.0
#    break
#
#print(optimizer.state_dict())
#
#print('Finished Training')

if __name__ == "__main__":
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    train_state_dict = mp.Manager().dict()
    val_acc_dict = mp.Manager().dict()
    net_acc_dict = mp.Manager().dict()
    print(torch.cuda.device_count())
    processes = []
    for rank in range(4):
        net_acc_dict[rank] = []
        learning_rate = [0.01, 0.06, 0.001, 0.008]
        p = mp.Process(target=training_cifar_multi, \
            args = (train_state_dict, val_acc_dict, net_acc_dict ,rank, \
                    return_top_arg, learning_rate[rank]))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

# Plotting Graphs
#    plt.figure()
#    for pros in net_acc_dict.keys():
#        plt.plot(net_acc_dict[pros], label = 'process: '+str(pros))
#    plt.legend(loc='best')
#    plt.savefig('Loss_vs_iteration.png')

# Saving Variables
    with open("loss_iteration.txt", "wb") as myFile:
        pickle.dump(net_acc_dict, myFile)
