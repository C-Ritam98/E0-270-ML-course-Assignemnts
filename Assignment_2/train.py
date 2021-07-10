# import standard PyTorch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets

import pandas as pd
import numpy as np
from functools import reduce



torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)     # On by default, here for clarity

from pprint import pprint


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class ConvNet(nn.Module):
    
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 0)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.fc1 = nn.Linear(in_features = 64*6*6, out_features = 600)
        self.fc2 = nn.Linear(in_features = 600, out_features = 150)
        self.fc3 = nn.Linear(in_features = 150, out_features = 10)
        
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        #x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
        
def Custom_train_val_split(split,length,Data):
  indices = list(range(length))
  np.random.shuffle(indices)

  split = int(np.floor(split * len(Data)))
  train_sampler , valid_sampler = torch.utils.data.SubsetRandomSampler(indices[split:]), torch.utils.data.SubsetRandomSampler(indices[:split])

  return torch.utils.data.DataLoader(Data,batch_size = 64, sampler = train_sampler), torch.utils.data.DataLoader(Data, batch_size = 64, sampler = valid_sampler)

def index_to_name(index):
    name = {0: 'T-shirt/top',
        1: 'Trouser',
        2: 'Pullover',
        3: 'Dress',
        4: 'Coat',
        5: 'Sandal',
        6: 'Shirt',
        7: 'Sneaker',
        8: 'Bag',
        9: 'Ankle-boot'}

    return name[index]



if __name__ == '__main__':

    # convert to tensor and normalise the data
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])


    # download data, if not downloaded previously
    mnist_train = torchvision.datasets.FashionMNIST(root='./data/',train=True,download = True,transform = transform)

    mnist_test = torchvision.datasets.FashionMNIST(root='./data/',train=False,download = True,transform = transform)




    # train-validation split
    validation_split = 0.2
    train_loader, valid_loader = Custom_train_val_split(validation_split,len(mnist_train),mnist_train)
    #test_loader = torch.utils.data.DataLoader(mnist_test,batch_size = 64)


    #model creation

    model = ConvNet()
    model.to(device)

    loss = nn.CrossEntropyLoss()

    l_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=l_rate)
    print(model)  # cross check the parameters


    num_epochs = 35
    count = 0

    # Lists for visualization of loss and accuracy
    loss_list = []
    iteration_list = []
    accuracy_list = []


    # for plotting
    losses = []
    epochs = []

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Transfering images and labels to GPU if available
            images, labels = images.to(device), labels.to(device)
        
            # train = images.view(64, 1, 28, 28)
            # labels = labels.view(64,)
            
            # Forward pass 
            outputs = model(images)
            error = loss(outputs, labels)
            
            # Initializing a gradient as 0 so there is no mixing of gradient among the batches
            optimizer.zero_grad()
            
            #Propagating the error backward
            error.backward()
            
            # Optimizing the parameters
            optimizer.step()
        
            count += 1
        
        # Testing the model on validation set after backpropping over a single batch of train data
        
            if not (count % 64):    # single batches
                total = 0
                correct = 0
                accuracy_temp = []
                for i,(images,labels) in enumerate(valid_loader):
                    
                    images, labels = images.to(device), labels.to(device)
                                    
                    outputs = model(images)
                    predictions = torch.max(outputs, 1)[1]
                    
                    correct += (predictions == labels).sum()
                    total += len(labels)
                    
                    accuracy = correct * 100 / total
                    accuracy_temp.append(accuracy)
                
                # store accuracy (averaged over all batches) of the validation set
                iteration_list.append(count/64)
                accuracy_list.append(reduce(lambda x,y : x+y, accuracy_temp)/len(accuracy_temp))
            
            # store the accuracy vs epoch count at the end of single epochs
            if not (count % 750):  # one epoch has 750 batches 
                print("Epoch: {}, Loss: {},".format(count//750, error.data))
                epochs.append(count/750)
                losses.append(error.data)

    print("Training Done!!")
    # save model
    torch.save(model,"./Model") # torch.save(model,PATH)
    print("Model saved!!")


    # epoch vs loss for the trained model while training
    plt.plot(epochs,losses)
    plt.title("Epoch vs loss plot")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()


    # iteration vs accuracy on validation set while training
    plt.plot(iteration_list, accuracy_list)
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title("Validation_set accuracy after each backpropagation")
    plt.show()

    '''
    with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_predict = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for i,(images,labels) in enumerate(test_loader):  # predict in batches of 64
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)

        prediction = torch.max(outputs,1)[1]

        n_samples += len(labels) #.size(0)
        n_correct += (prediction == labels).sum()

        for i in range(len(labels)):
        label = labels[i]
        pred = prediction[i]
        n_class_samples[label]+=1
        n_class_predict[label] += int(pred == label)

    print(f'Accuracy = {100* n_correct/n_samples}')
    print(n_class_predict)
    print(n_class_samples)

    for i in range(10):
        print(name[i], end = ' ')
    '''


