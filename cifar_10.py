#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 18:29:30 2017

@author: saurabh
"""

import torch.multiprocessing as mp
import pickle




def training_cifar_multi(train_state_dict, val_acc_dict, net_acc_dict ,name,return_top_arg,learn_rate):
    '''
    This is the main training code. It will use the definitions present in cifar_resnet
    and load_cifar_data to initialize the model and load the datasets into the input 
    pipeline. It will run for 13 epochs. At the end of each epoch it will calculate
    the accuracy on the validation set and depending on the performance explore and 
    exploit other models. Test error is calculated at the end.
    '''
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.autograd import Variable
    import numpy as np
    import load_cifar_data as ld
    import cifar_resnet as cr
    

    
    model = cr.ResNet56()
    net_acc_dict[name] = []
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learn_rate , momentum=0.9, weight_decay=0.0001)
    
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
            if i % 1000 == 999:    # print every 1000 mini-batches
                print('process =', name, '[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                temp_list  = net_acc_dict[name]
                temp_list.append(running_loss / 1000)
                net_acc_dict[name] = temp_list
                running_loss = 0.0
            

    
        #Saving model to manager
        model.eval()
        train_state_dict[name] = {'state_dict': model.state_dict(), 'optimizer': 
                        optimizer.state_dict(), 'epoch':epoch}
        
        #Getting validation accuracy   
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
        
        #Depending on the performance explore and exploit other models randomly
        #selecting from the top 20% best performers
        flag = return_top_arg(val_acc_dict, valid_accuracy)
        if flag:
            model.load_state_dict(train_state_dict[flag]['state_dict'])
            optimizer.load_state_dict(train_state_dict[flag]['optimizer'])
            epoch = train_state_dict[flag]['epoch']
            for param_group in optimizer.param_groups:
                param_group['lr'] = (np.random.uniform(0.5,2,1)[0])*param_group['lr']
        
        epoch += 1
    
    #Calculate Test Accuracy       
    for ix, (test_img,test_label) in enumerate(test_dataloader):
        test_outputs = model(Variable(test_img))
        _, predicted = torch.max(test_outputs.data, 1)
        total += test_label.size(0)
        correct += (predicted == test_label).sum()
    test_accuracy = 100*correct/total
    print("Testing accuracy = ", test_accuracy)        
    
    #Save the final model
    with open("model_dict_"+str(rank)+".txt", "wb") as myFile:
        pickle.dump(model.state_dict(), myFile)

        
        
        
        
        
        
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
    


if __name__ == "__main__":
    train_state_dict = mp.Manager().dict()
    val_acc_dict = mp.Manager().dict()
    net_acc_dict = mp.Manager().dict()
    learn_rate = [0.01, 0.06, 0.001, 0.008]
    processes = []
    for rank in range(4):
        p = mp.Process(target=training_cifar_multi, \
            args = (train_state_dict, val_acc_dict, net_acc_dict ,rank,return_top_arg, learn_rate[rank]))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

# Saving files for plotting later
    with open("loss_iteration.txt", "wb") as myFile:
        pickle.dump(net_acc_dict, myFile)
